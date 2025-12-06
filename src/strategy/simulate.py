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

    def run_walk_forward(self, start_year=2023):
        """
        Performs Walk-Forward Validation and returns predictions DataFrame.
        Trains on data < year, tests on data == year.
        Uses train_v2.py logic for leakage prevention.
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
        # race_id format: YYYYPPKKDDRR where PP is place code
        jra_places = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        df['place_code'] = df['race_id'].astype(str).str[4:6]
        original_count = len(df)
        df = df[df['place_code'].isin(jra_places)].copy()
        logger.info(f"Filtered to JRA only: {len(df)}/{original_count} records")

        # Sort strictly
        df = df.sort_values(['date_dt', 'race_id'])

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
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            params.update(best_params)

        for test_year in test_years:
            logger.info(f"=== Training for Year {test_year} ===")
            
            # Split Data
            train_df_raw = df[df['year'] < test_year].copy()
            test_df_raw = df[df['year'] == test_year].copy()
            
            if len(train_df_raw) == 0:
                continue
            
            # Feature Engineering: FIT on ALL data first to generate raw features?
            # NO, we should follow train_v2 logic: generate raw features, THEN split, THEN encode.
            # But here we need to be careful not to use future data in rolling windows.
            # Ideally, rolling windows are causal (shift(1)), so generating on full df is technically safe 
            # as long as we don't use global stats that include future.
            # FeatureEngineer v2 uses shift(1), so it's causal.
            # BUT, race-relative stats (z-score) are within-race. That's fine.
            # The only risk is target encoding (handled by split) and global aggregations.
            
            # To be safe, generate on combined [train + test] for this iteration?
            # Or generate on FULL logic but filter?
            # Let's generate on full DF up to this point?
            # Actually, simply calling FE on proper subsets is safer but might break rolling windows at the boundary.
            # Better approach: Pass full DF, but marking the boundary?
            # Or standard: Generate features on FULL HISTORY (df), then slice.
            
            # Using transform on FULL df to preserve rolling window continuity
            # (Assuming FE handles causality correctly)
            current_df = df[df['year'] <= test_year].copy()
            
            # 1. Generate Raw Features
            logger.info("Generating raw features...")
            # Re-initialize FE to avoid state carryover issues
            fe_iter = FeatureEngineer()
            df_generated = fe_iter.transform(current_df, encoding_only=False)
            
            train_mask = df_generated['year'] < test_year
            test_mask = df_generated['year'] == test_year
            
            df_train = df_generated[train_mask].copy()
            df_test = df_generated[test_mask].copy()
            
            # 2. Target Encoding (Fit on Train, Apply to Both)
            logger.info("Encoding features...")
            df_train_enc = fe_iter.fit_transform(df_train, encoding_only=True)
            df_test_enc = fe_iter.transform(df_test, encoding_only=True)
            
            # 3. Define Features
            # Same exclude list as train_v2.py (Production Logic)
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
            
            # 4. Train
            logger.info(f"Training model for {test_year} with {len(final_feature_cols)} features...")
            
            # Base Model
            base_model = LGBMClassifier(**params)
            
            # Calibrated Model (Isotonic)
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
            calibrated_model.fit(X_train, y_train)
            
            # 5. Predict
            preds = calibrated_model.predict_proba(X_test)[:, 1]
            
            # Store
            test_df_copy = df_test.copy() # Use the generated one to keep metadata if needed, but safe to use original test_df_raw part if indices match?
            # Better to use df_test_enc index alignment
            test_df_copy['pred_prob'] = preds
            
            # We need raw odds for simulation, ensure they are preserved or retrieve from raw
            # feature engineering might have dropped/transformed odds?
            # 'odds' is in exclude_cols, so it should be in df_test_enc (just not used for training)
            # Check if columns are present
            if 'odds' not in test_df_copy.columns:
                 # Restore from raw using index
                 test_df_copy['odds'] = test_df_raw.loc[test_df_copy.index, 'odds']
            if 'horse_num' not in test_df_copy.columns:
                 test_df_copy['horse_num'] = test_df_raw.loc[test_df_copy.index, 'horse_num']
                 
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
        thresholds = [round(x * 0.1, 1) for x in range(10, 31)] 
        
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
