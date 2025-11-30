import pandas as pd
import pickle
import os
from src.features.engineer import FeatureEngineer
from src.strategy.betting import BettingStrategy
from src.automation.ipat import IpatController
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info("Starting Keiba AI System...")

    # 1. Load Model
    model_path = 'models/lgbm_baseline.pkl'
    if not os.path.exists(model_path):
        logger.error("Model not found. Please train the model first.")
        return
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")

    # 2. Load Real-time Data (Mocked using our existing raw data for demonstration)
    # In production, this would be scraped live or read from JRA-VAN CSVs
    raw_data_path = 'data/common/raw_data/results.csv'
    if not os.path.exists(raw_data_path):
        logger.error("Raw data not found.")
        return
    
    df = pd.read_csv(raw_data_path)
    
    # Pick one race for simulation
    sample_race_id = df['race_id'].unique()[0]
    logger.info(f"Simulating for Race ID: {sample_race_id}")
    
    race_data = df[df['race_id'] == sample_race_id].copy()
    
    # 3. Feature Engineering
    fe = FeatureEngineer()
    # Note: In production, we should load the fitted encoders, but here we fit on the fly for demo
    df_processed = fe.fit_transform(race_data)
    
    exclude_cols = ['race_id', 'date', 'time', 'rank', 'target']
    feature_cols = [c for c in df_processed.columns if c not in exclude_cols]
    
    X = df_processed[feature_cols]
    
    # 4. Predict
    probs = model.predict(X)
    race_data['pred_prob'] = probs
    
    # 5. Strategy Decision
    # Create a mock odds dictionary from the data
    odds_data = dict(zip(race_data['horse_num'], race_data['odds']))
    
    strategy = BettingStrategy(ev_threshold=1.0) # Lower threshold for demo to ensure some bets
    bets = strategy.decide_bet(race_data[['horse_num', 'pred_prob']], odds_data)
    
    logger.info(f"Strategy decided {len(bets)} bets.")

    # 6. Execute Bet (Dry-Run)
    controller = IpatController()
    controller.vote(sample_race_id, bets, dry_run=True)

    logger.info("System finished.")

if __name__ == "__main__":
    main()
