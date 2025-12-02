import pandas as pd
import lightgbm as lgb
from experiments.v2_feature_expansion.src.feature_engineering import FeatureEngineer
from experiments.v2_feature_expansion.src.betting import BettingStrategy
from common.src.utils.logger import setup_logger
from sklearn.metrics import accuracy_score, log_loss

logger = setup_logger(__name__)

class Backtester:
    def __init__(self, data_path='common/data/rawdf/results.csv'):
        self.data_path = data_path
        self.fe = FeatureEngineer()
        self.strategy = BettingStrategy(ev_threshold=1.5)

    def run_walk_forward(self, start_year=2021):
        """
        Performs Walk-Forward Validation and returns predictions DataFrame.
        Trains on data < year, tests on data == year.
        """
        logger.info("Loading data for simulation...")
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            return pd.DataFrame()

        try:
            df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日')
            df['year'] = df['date_dt'].dt.year
        except Exception as e:
            logger.error(f"Failed to parse dates: {e}")
            return pd.DataFrame()

        years = sorted(df['year'].unique())
        test_years = [y for y in years if y >= start_year]
        
        if not test_years:
            logger.warning(f"No data found for years >= {start_year}.")
            return pd.DataFrame()

        all_predictions = []
        
        # Load Best Params
        import json
        import os
        best_params_path = 'models/best_params.json'
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 1000,
            'verbose': -1
        }
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
            params.update(best_params)

        from sklearn.calibration import CalibratedClassifierCV
        from lightgbm import LGBMClassifier

        for test_year in test_years:
            logger.info(f"=== Training for Year {test_year} ===")
            
            train_df = df[df['year'] < test_year]
            test_df = df[df['year'] == test_year]
            
            if len(train_df) == 0:
                continue
            
            # Feature Engineering
            fe = FeatureEngineer()
            train_X_all = fe.fit_transform(train_df)
            test_X_all = fe.transform(test_df)
            
            # Define features (exclude target and non-feature columns)
            exclude_cols = ['race_id', 'date', 'time', 'rank', 'target', 'date_dt', 'year',
                            'horse_name', 'jockey', 'trainer', 'horse_id', 'jockey_id', 'trainer_id', 'time_seconds',
                            'prize', 'surface', 'distance', 'weather', 'condition', 'jockey_trainer_pair', 'winner_time']
            feature_cols = [c for c in train_X_all.columns if c not in exclude_cols]
            
            X_train = train_X_all[feature_cols]
            y_train = train_X_all['target']
            X_test = test_X_all[feature_cols]
            
            base_model = LGBMClassifier(**params)
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
            calibrated_model.fit(X_train, y_train)
            
            preds = calibrated_model.predict_proba(X_test)[:, 1]
            
            # Store predictions with necessary info for simulation
            test_df_copy = test_df.copy()
            test_df_copy['pred_prob'] = preds
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
                odds_data = dict(zip(race_data['horse_num'], race_data['odds']))
                bets = self.strategy.decide_bet(race_data[['horse_num', 'pred_prob']], odds_data)
                
                for bet in bets:
                    amount = bet['amount']
                    horse_num = bet['horse_num']
                    year_bet_amount += amount
                    
                    result_row = race_data[race_data['horse_num'] == horse_num].iloc[0]
                    if result_row['rank'] == 1:
                        payout = amount * result_row['odds']
                        year_return += payout
            
            total_return += year_return
            total_bet_amount += year_bet_amount
            
            if verbose:
                recovery = (year_return / year_bet_amount * 100) if year_bet_amount > 0 else 0
                logger.info(f"Year {year}: Bet {year_bet_amount} -> Return {year_return} | Recovery: {recovery:.1f}%")

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
        pred_df = backtester.run_walk_forward(start_year=2023)
        backtester.simulate_with_threshold(pred_df, ev_threshold=best_th, verbose=True)
