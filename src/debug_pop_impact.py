import pandas as pd
import numpy as np
import pickle
import os
import sys
from src.data.parser_shutsuba import ShutsubaParser
from src.features.engineer import FeatureEngineer
from src.utils.logger import setup_logger

logger = setup_logger('debug_inspect')

def main():
    # 1. Simulate the environment of predict_today (Shortened)
    # We will just load the saved output csv if possible? No, we need inputs.
    # We need to run the scraping/parsing part or mock it.
    # Since we can't easily standard input into predict_today, I'll copy the logic.
    
    # Load History
    logger.info("Loading history (head)...")
    df_history = pd.read_csv('data/common/raw_data/results.csv', nrows=5)
    # Fix IDs in history as per my fix
    id_cols = ['horse_id', 'jockey_id', 'trainer_id', 'owner_id', 'sire_id', 'dam_id', 'bms_id', 'breeder_id']
    for col in id_cols:
        if col in df_history.columns:
            df_history[col] = pd.to_numeric(df_history[col], errors='coerce')
            
    # Load FE
    with open('models/feature_engineer.pkl', 'rb') as f:
        fe = pickle.load(f)
        
    # Mock "Today" Data
    # I'll manually create a row that looks like the problem cases (Race 11)
    # From user output:
    # 4: キタノズエッジ, Odds 5.9
    # I don't have all IDs, but I can check how FE handles missing IDs (Target Enc -> Global Mean).
    # Critical part is Popularity.
    
    today_data = [{
        'race_id': '202512079999',
        'horse_num': 4,
        'horse_name': 'Test Horse',
        'date': '2025-12-07',
        'odds': 5.9,
        # popularity is MISSING (simulating parser failure)
        'horse_weight': 500,
        'weight': 56,
        'jockey_id': 99999, # Unknown
        'trainer_id': 99999
    }, {
        'race_id': '202512079999',
        'horse_num': 8,
        'horse_name': 'Test Horse 2',
        'date': '2025-12-07',
        'odds': 4.9,
        'popularity': 1.0, # Explicitly valid
        'horse_weight': 500,
        'weight': 56,
        'jockey_id': 99999,
        'trainer_id': 99999
    }]
    
    df_today = pd.DataFrame(today_data)
    
    # Combine
    df_combined = pd.concat([df_history, df_today], axis=0, ignore_index=True)
    # Mixed date formats handling
    def parse_date(x):
        try:
            return pd.to_datetime(x)
        except:
            try:
                # Japanese format: 2018年12月15日 => 2018-12-15
                return pd.to_datetime(x, format='%Y年%m月%d日')
            except:
                return pd.NaT

    df_combined['date_dt'] = df_combined['date'].apply(parse_date)
    df_combined = df_combined.sort_values('date_dt')
    
    # Transform
    logger.info("Transforming...")
    df_processed = fe.transform(df_combined)
    
    # Check "Test Horse" (Missing Pop)
    row_missing = df_processed[df_processed['horse_name'] == 'Test Horse'].iloc[0]
    row_valid = df_processed[df_processed['horse_name'] == 'Test Horse 2'].iloc[0]
    
    print("\n--- Feature Inspection ---")
    print(f"Missing Pop - 'popularity' feature value: {row_missing.get('popularity')}")
    print(f"Valid Pop   - 'popularity' feature value: {row_valid.get('popularity')}")
    
    # Load Model and Predict
    with open('models/lgbm_calibrated.pkl', 'rb') as f:
        model = pickle.load(f)
        
    # Prepare X
    # Extract features from model
    model_features = []
    try:
        base = model.calibrated_classifiers_[0].estimator
        model_features = base.booster_.feature_name()
    except:
        pass
        
    cols = model_features if model_features else [c for c in df_processed.columns if pd.api.types.is_numeric_dtype(df_processed[c])]
    
    # Prediction
    X_missing = pd.DataFrame([row_missing])[cols]
    X_valid = pd.DataFrame([row_valid])[cols]
    
    p_missing = model.predict_proba(X_missing)[:, 1][0]
    p_valid = model.predict_proba(X_valid)[:, 1][0]
    
    print(f"Pred Prob (Missing Pop): {p_missing:.4f}")
    print(f"Pred Prob (Valid Pop=1): {p_valid:.4f}")

if __name__ == "__main__":
    main()
