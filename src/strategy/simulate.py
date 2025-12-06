import pandas as pd
import lightgbm as lgb
from src.features.engineer import FeatureEngineer
from src.strategy.betting import BettingStrategy
from src.utils.logger import setup_logger
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import json
import os
import numpy as np

logger = setup_logger(__name__)

class Backtester:
    def __init__(self, data_path='data/common/raw_data/results.csv'):
        self.data_path = data_path
        self.fe = FeatureEngineer()
        self.strategy = BettingStrategy(ev_threshold=1.5)

    def prepare_features(self, df):
        """
        Generates or loads features from cache.
        """
        cache_path = 'data/cache/features_sim.pkl'
        if os.path.exists(cache_path):
            logger.info("Loading features from cache...")
            cached_df = pd.read_pickle(cache_path)
            # Simple validation: size match? or just trust user.
            # Trust user for now as per plan.
            return cached_df
        
        logger.info("Generating features for entire dataset...")
        fe = FeatureEngineer()
        # Transform full dataset (encoding_only=False) -> generates raw numeric features
        # Note: df is already sorted by date in run_walk_forward
        df_features = fe.transform(df, encoding_only=False)
        
        # Save cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_features.to_pickle(cache_path)
        
        return df_features

    def run_walk_forward(self, start_year=2023):
        """
        Performs Walk-Forward Validation and returns predictions DataFrame.
        Trains on data < year, tests on data == year.
        Uses cached features to speed up simulation.
        """
        logger.info("Loading data for simulation...")
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            return pd.DataFrame()

        try:
            # Handle date parsing strictly for correct yearly split
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
            else:
                df['date_dt'] = df['date']
            df['year'] = df['date_dt'].dt.year
        except Exception as e:
            logger.error(f"Failed to parse dates: {e}")
            return pd.DataFrame()

        # Filter for JRA (Central Racing) only - place codes 01-10
        jra_places = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        df['place_code'] = df['race_id'].astype(str).str[4:6]
        original_count = len(df)
        df = df[df['place_code'].isin(jra_places)].copy()
        logger.info(f"Filtered to JRA only: {len(df)}/{original_count} records")

        # Sort strictly
        df = df.sort_values(['date_dt', 'race_id'])

        # Prepare Features (Cached)
        df_features = self.prepare_features(df)
        
        if 'year' not in df_features.columns:
            # Ensure year column is preserved or restored
            df_features['year'] = df.loc[df_features.index, 'year']

        years = sorted(df['year'].unique())
        test_years = [y for y in years if y >= start_year]
        
        if not test_years:
            logger.warning(f"No data found for years >= {start_year}.")
            return pd.DataFrame()

        all_predictions = []
        
        # Load Best Params
        best_params_path = 'models/best_params.json'
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 1000,
            'verbose': -1,
            'learning_rate': 0.05,
            'num_leaves': 31
        }
        if os.path.exists(best_params_path):
            try:
                with open(best_params_path, 'r') as f:
                    best_params = json.load(f)
                params.update(best_params)
            except:
                pass

        for test_year in test_years:
            logger.info(f"=== Training for Year {test_year} ===")
            
            # Split Data using Pre-calculated Features
            train_mask = df_features['year'] < test_year
            test_mask = df_features['year'] == test_year
            
            if not train_mask.any():
                continue
            
            df_train = df_features[train_mask].copy()
            df_test = df_features[test_mask].copy()
            
            # Target Encoding (Fit on Train, Apply to Both)
            # We must re-fit encoders for each fold to avoid leakage
            logger.info("Encoding features...")
            fe_iter = FeatureEngineer()
            
            # Use fit_transform for Train (Apply K-Fold TE to prevent leakage)
            # skip_feature_generation=True because df_train already has raw features
            df_train_enc = fe_iter.fit_transform(df_train, encoding_only=True, skip_feature_generation=True)
            
            # Use transform for Test (Apply Global TE from fit)
            df_test_enc = fe_iter.transform(df_test, encoding_only=True)
            
            # Define Features
            future_info = [
                'rank', 'time', 'time_seconds', 'target', 'prize', 'log_prize',
                'passing_order', 'agari_3f', 'margin', 'margin_val',
                'corner_1コーナー', 'corner_2コーナー', 'corner_3コーナー', 'corner_4コーナー', 'race_laps',
                'running_style', 'position_gain',
                'odds', 'popularity', 'log_odds', 
                'is_win', 'is_place', 'is_show'
            ]
            
            meta_cols = [
                'race_id', 'date', 'date_dt', 'horse_name', 'jockey', 'trainer', 
                'horse_id', 'jockey_id', 'trainer_id', 'owner', 'owner_id', 'year',
                'sire_id', 'dam_id', 'bms_id', 'breeder_id',
                'jockey_trainer_pair', 'sire_name', 'dam_name', 'bms_name', 'owner_name', 'breeder_name'
            ]
            
            raw_interactions = [
                'sire_surface', 'sire_distance', 'bms_surface', 'bms_distance',
                'jockey_place', 'jockey_surface', 'jockey_distance',
                'trainer_place', 'trainer_surface', 'trainer_distance',
                'owner_surface', 'horse_jockey',
                'age_gender', 'class_distance', 'condition_surface'
            ]
            
            exclude_cols = set(future_info + meta_cols + raw_interactions)
            feature_cols = [c for c in df_train_enc.columns if c not in exclude_cols]
            
            # Final numeric check
            final_feature_cols = []
            for c in feature_cols:
                if pd.api.types.is_numeric_dtype(df_train_enc[c]):
                    final_feature_cols.append(c)
            
            X_train = df_train_enc[final_feature_cols]
            y_train = df_train_enc['target']
            X_test = df_test_enc[final_feature_cols]
            
            # Train
            logger.info(f"Training model for {test_year} with {len(final_feature_cols)} features...")
            
            base_model = LGBMClassifier(**params)
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
            calibrated_model.fit(X_train, y_train)
            
            # Predict
            preds = calibrated_model.predict_proba(X_test)[:, 1]
            
            # Store
            test_df_copy = df_test.copy()
            test_df_copy['pred_prob'] = preds
            
            # Ensure essential columns for simulation
            # We need odds and horse_num. They might be in excludes but should be in df_test (raw features)
            # If df_features preserved them (it should as transform(encoding_only=False) keeps columns)
            
            all_predictions.append(test_df_copy)
            
        if not all_predictions:
            return pd.DataFrame()
            
        return pd.concat(all_predictions, ignore_index=True)

    def simulate_with_threshold(self, df, ev_threshold, verbose=False):
        """
        Runs betting simulation on pre-calculated predictions.
        """
        self.strategy.ev_threshold = ev_threshold
        total_return = 0
        total_bet_amount = 0
        
        # Group by year for logging
        years = sorted(df['year'].unique())
        
        for year in years:
            year_df = df[df['year'] == year]
            year_return = 0
            year_bet_amount = 0
            
            for race_id, race_data in year_df.groupby('race_id'):
                # Ensure odds are present
                if 'odds' not in race_data.columns:
                    continue
                    
                odds_data = dict(zip(race_data['horse_num'], race_data['odds']))
                bets = self.strategy.decide_bet(race_data[['horse_num', 'pred_prob']], odds_data)
                
                for bet in bets:
                    amount = bet['amount']
                    horse_num = bet['horse_num']
                    year_bet_amount += amount
                    
                    # Check result
                    result_row = race_data[race_data['horse_num'] == horse_num].iloc[0]
                    if result_row['rank'] == 1:
                        payout = amount * result_row['odds']
                        year_return += payout
            
            total_return += year_return
            total_bet_amount += year_bet_amount
            
            if verbose:
                recovery = (year_return / year_bet_amount * 100) if year_bet_amount > 0 else 0
                logger.info(f"Year {year}: Bet {year_bet_amount} -> Return {int(year_return)} | Recovery: {recovery:.1f}%")

        total_recovery = (total_return / total_bet_amount * 100) if total_bet_amount > 0 else 0
        return total_return, total_bet_amount, total_recovery

    def optimize_thresholds(self, start_year=2023):
        """
        Performs Grid Search efficiently.
        """
        # 1. Generate Predictions (Slow, but done once)
        logger.info("Generating predictions for optimization...")
        pred_df = self.run_walk_forward(start_year=start_year)
        
        if pred_df.empty:
            logger.error("No predictions generated.")
            return
            
        # Save predictions cache
        pred_df.to_csv('simulation_predictions.csv', index=False)

        logger.info("Starting Threshold Optimization (Grid Search)...")
        
        results = []
        thresholds = [round(x * 0.1, 1) for x in range(10, 21)] 
        
        best_recovery = 0
        best_threshold = 0
        
        for th in thresholds:
            # 2. Simulate (Fast)
            ret, bet, recovery = self.simulate_with_threshold(pred_df, ev_threshold=th, verbose=False)
            logger.info(f"Threshold {th}: Recovery {recovery:.1f}% (Bet {bet})")
            results.append({'threshold': th, 'recovery': recovery, 'bet': bet, 'return': ret})
            
            if recovery > best_recovery and bet > 0:
                best_recovery = recovery
                best_threshold = th
                
        logger.info(f"=== Optimization Complete ===")
        logger.info(f"Best Threshold: {best_threshold}")
        logger.info(f"Best Recovery: {best_recovery:.1f}%")
        
        return best_threshold

if __name__ == "__main__":
    backtester = Backtester()
    # Run Optimization
    best_th = backtester.optimize_thresholds(start_year=2023)
    
    # Run detailed simulation with best threshold
    if best_th:
        logger.info(f"Running detailed simulation with Best Threshold: {best_th}")
        # Re-load predictions if possible or use logical flow
        # In optimized function we saved to csv, let's just reload to simulate?
        # Or Backtester doesn't state data, pass df.
        
        if os.path.exists('simulation_predictions.csv'):
             pred_df = pd.read_csv('simulation_predictions.csv')
             backtester.simulate_with_threshold(pred_df, ev_threshold=best_th, verbose=True)
