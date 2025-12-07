import pandas as pd
import numpy as np
import pickle
import os
import sys
from src.features.engineer import FeatureEngineer

# Setup logger to stdout
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('debug')

def main():
    # 1. Load History (Train Data Sample)
    logger.info("Loading history...")
    df_history = pd.read_csv('data/common/raw_data/results.csv', nrows=1000)
    
    # Mock 'df_today' by taking a recent race from history and pretending it's today
    # (stripping result columns)
    last_date = df_history['date'].max()
    # Actually let's just use the history dataframe as 'combined' to see what the features look like
    # vs what they look like if we pass a "dummy" row.
    
    # Better yet, let's look at the ACTUAL df_today if possible.
    # But I can't scrape right now easily.
    # I'll rely on inspecting the FeatureEngineer object and the logic.

    model_dir = 'models'
    fe_path = os.path.join(model_dir, 'feature_engineer.pkl')
    
    if not os.path.exists(fe_path):
        logger.error("No FE found")
        return

    with open(fe_path, 'rb') as f:
        fe = pickle.load(f)

    logger.info("Loaded FeatureEngineer.")
    
    # Check what encodings are stored
    # FE likely has 'target_enc_map' or similar
    if hasattr(fe, 'te_maps'):
        logger.info(f"TE Maps keys: {fe.te_maps.keys()}")
        # Inspect a few values
        for col in list(fe.te_maps.keys())[:3]:
            logger.info(f"TE Map for {col} (sample): {list(fe.te_maps[col].items())[:5]}")
    
    # Check if there are any default values that are high
    
    # Also Check Model Features
    model_path = os.path.join(model_dir, 'lgbm_calibrated.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Extract features used by model
    model_features = []
    try:
        if hasattr(model, 'calibrated_classifiers_'):
            base = model.calibrated_classifiers_[0].estimator
            model_features = base.booster_.feature_name()
    except:
        pass
        
    logger.info(f"Model uses {len(model_features)} features.")
    
    # 2. Check if specific columns could cause high probs
    # e.g. if 'jockey_id' is target encoded, and we pass a new jockey ID, what does it get?
    # Usually global mean.
    
    # Let's try to simulate a 'prediction' on a dummy row
    dummy_row = {
        'race_id': '202599999999',
        'horse_id': '9999999999',
        'date': pd.to_datetime('2025-12-07'),
        # Add minimal columns required by FE
    }
    # We need to know what columns FE expects.
    # It expects raw columns.
    
    # I suspect the issue is simply that the Binary Model is uncalibrated for the "Field Strength".
    # But why did simulate.py work?
    
if __name__ == "__main__":
    main()
