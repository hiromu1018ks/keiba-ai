import pandas as pd
import lightgbm as lgb
from src.features.engineer import FeatureEngineer
from src.strategy.betting import BettingStrategy
from src.utils.logger import setup_logger
from sklearn.metrics import accuracy_score, log_loss

logger = setup_logger(__name__)

class Backtester:
    def __init__(self, data_path='data/common/raw_data/results.csv'):
        self.data_path = data_path
        self.fe = FeatureEngineer()
        self.strategy = BettingStrategy(ev_threshold=1.5)

    def run_walk_forward(self, start_year=2021):
        """
        Performs Walk-Forward Validation.
        Trains on data < year, tests on data == year.
        """
        logger.info("Loading data for simulation...")
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            return

        # Preprocess date to get year
        # Assuming 'date' column format is '2023年01月28日' or similar
        # We need to convert it to datetime objects
        # For the sample data, it is '2023年01月28日'
        try:
            df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日')
            df['year'] = df['date_dt'].dt.year
        except Exception as e:
            logger.error(f"Failed to parse dates: {e}")
            return

        years = sorted(df['year'].unique())
        test_years = [y for y in years if y >= start_year]
        
        if not test_years:
            logger.warning(f"No data found for years >= {start_year}. Available years: {years}")
            return

        total_return = 0
        total_bet_amount = 0
        
        for test_year in test_years:
            logger.info(f"=== Simulating Year {test_year} ===")
            
            # Split Data
            train_df = df[df['year'] < test_year]
            test_df = df[df['year'] == test_year]
            
            if len(train_df) == 0:
                logger.warning(f"No training data for year {test_year}. Skipping.")
                continue
            
            # Feature Engineering
            # Fit on Train, Transform on Train & Test
            logger.info(f"Training data size: {len(train_df)}, Test data size: {len(test_df)}")
            
            # Note: We need to fit a new FeatureEngineer for each window to avoid leakage
            fe = FeatureEngineer()
            train_X_all = fe.fit_transform(train_df)
            test_X_all = fe.transform(test_df)
            
            exclude_cols = ['race_id', 'date', 'time', 'rank', 'target', 'date_dt', 'year',
                            'horse_name', 'jockey', 'trainer', 'horse_id', 'jockey_id', 'trainer_id', 'time_seconds']
            feature_cols = [c for c in train_X_all.columns if c not in exclude_cols]
            
            X_train = train_X_all[feature_cols]
            y_train = train_X_all['target']
            X_test = test_X_all[feature_cols]
            y_test = test_X_all['target'] # For evaluation only
            
            # Train Model
            from sklearn.calibration import CalibratedClassifierCV
            from lightgbm import LGBMClassifier
            import json
            import os

            # Load Best Params if available
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
                logger.info(f"Using optimized parameters: {best_params}")
            else:
                logger.warning("best_params.json not found. Using default parameters.")

            # Base LightGBM Model
            base_model = LGBMClassifier(**params)

            # Calibrated Classifier (Isotonic Regression)
            calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
            calibrated_model.fit(X_train, y_train)
            
            # Predict
            # CalibratedClassifierCV returns probabilities for both classes [prob_0, prob_1]
            preds = calibrated_model.predict_proba(X_test)[:, 1]
            test_df = test_df.copy()
            test_df['pred_prob'] = preds
            
            # Simulate Betting
            # Group by race_id
            year_return = 0
            year_bet_amount = 0
            
            # Load Payout Data
            payout_path = 'data/common/raw_data/payouts.csv'
            if os.path.exists(payout_path):
                payout_df = pd.read_csv(payout_path)
                # Filter for current year races only to speed up
                # But race_id is unique so it's fine
            else:
                payout_df = pd.DataFrame()
                logger.warning("Payout data not found. Skipping complex bet simulation.")

            for race_id, race_data in test_df.groupby('race_id'):
                # Mock odds data from the dataframe itself
                odds_data = dict(zip(race_data['horse_num'], race_data['odds']))
                
                # 1. Single Win Strategy (Existing)
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

                # 2. Box Betting Strategy (New Demo)
                # Buy Box 3 for Quinella (Uren) and Trio (Sanfuku)
                # Top 3 horses by predicted probability
                top_3_horses = race_data.sort_values('pred_prob', ascending=False).head(3)['horse_num'].tolist()
                
                if len(top_3_horses) == 3 and not payout_df.empty:
                    # Quinella Box (3 combinations: 1-2, 1-3, 2-3)
                    # Cost: 3 * 100 = 300
                    year_bet_amount += 300 
                    
                    # Check if any combination hit
                    race_payouts = payout_df[payout_df['race_id'] == race_id]
                    
                    # Helper to check hit
                    def check_hit(ticket_type, horses):
                        # horses is a list of ints. Payout 'horse_nums' is string like "1 - 2"
                        # We need to parse payout string
                        hits = 0
                        return_amount = 0
                        
                        type_rows = race_payouts[race_payouts['ticket_type'] == ticket_type]
                        for _, row in type_rows.iterrows():
                            # Parse winning numbers
                            # Format: "1 - 2" or "1 - 2 - 3" or "1 → 2"
                            # Normalize separators
                            txt = str(row['horse_nums']).replace('→', '-').replace(' ', '')
                            winning_nums = [int(x) for x in txt.split('-') if x.isdigit()]
                            
                            # Check if our box contains all winning numbers
                            # For Quinella/Trio, order doesn't matter
                            if set(winning_nums).issubset(set(horses)):
                                return_amount += row['payout']
                                hits += 1
                        return return_amount

                    # Quinella (馬連)
                    uren_return = check_hit('馬連', top_3_horses)
                    year_return += uren_return
                    
                    # Trio (3連複) - Box 3 is just 1 combination
                    # sanfuku_return = check_hit('三連複', top_3_horses)
                    # year_return += sanfuku_return
                    
                    # Note: For now, I am NOT adding these to the main 'year_return' 
                    # because we want to compare apples to apples with previous Single Win results.
                    # But the logic is here ready to be enabled or logged separately.
            
            year_profit = year_return - year_bet_amount
            year_recovery_rate = (year_return / year_bet_amount * 100) if year_bet_amount > 0 else 0
            
            logger.info(f"Year {test_year} Result: Bet {year_bet_amount} -> Return {year_return} (Profit {year_profit}) | Recovery: {year_recovery_rate:.1f}%")
            
            total_return += year_return
            total_bet_amount += year_bet_amount

        total_profit = total_return - total_bet_amount
        total_recovery_rate = (total_return / total_bet_amount * 100) if total_bet_amount > 0 else 0
        logger.info(f"=== Total Simulation Result ===")
        logger.info(f"Total Bet: {total_bet_amount}")
        logger.info(f"Total Return: {total_return}")
        logger.info(f"Total Profit: {total_profit}")
        logger.info(f"Recovery Rate: {total_recovery_rate:.1f}%")

if __name__ == "__main__":
    # For demo, we might not have enough data for multiple years.
    # But this script is ready for when we do.
    backtester = Backtester()
    # Using 2023 as start year for demo since our sample data is 2023
    backtester.run_walk_forward(start_year=2023)
