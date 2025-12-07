import pandas as pd
import numpy as np
import pickle
import os
import sys
from src.features.engineer import FeatureEngineer
from src.utils.logger import setup_logger

logger = setup_logger('deep_debug')

def main():
    # 1. Load Model and FE
    with open('models/lgbm_calibrated.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_engineer.pkl', 'rb') as f:
        fe = pickle.load(f)
        
    # 2. Get Feature Names
    if hasattr(model, 'calibrated_classifiers_'):
        base = model.calibrated_classifiers_[0].estimator
        model_features = base.booster_.feature_name()
    else:
        model_features = model.booster_.feature_name()
    
    logger.info(f"Model checks {len(model_features)} features.")
    
    # 3. Load History (Simulation Context)
    # We take a sample from 2023 (likely trained/val on)
    df_history = pd.read_csv('data/common/raw_data/results.csv')
    # Parse dates with support for Japanese format
    def parse_date(x):
        try:
            return pd.to_datetime(x)
        except:
            try:
                # Japanese format: 2018年12月15日 => 2018-12-15
                return pd.to_datetime(x, format='%Y年%m月%d日')
            except:
                return pd.NaT

    df_history['date_dt'] = df_history['date'].apply(parse_date)
    # Filter JRA
    df_history['place_code'] = df_history['race_id'].astype(str).str[4:6]
    df_sim_sample = df_history[df_history['date_dt'] > '2023-11-01'].head(500).copy() # A specific race set
    
    # Ensure IDs numeric (as we fixed in predict) - check existence first
    id_cols = ['horse_id', 'jockey_id', 'trainer_id', 'owner_id', 'sire_id', 'dam_id', 'bms_id', 'breeder_id']
    for col in id_cols:
        if col in df_sim_sample.columns:
            df_sim_sample[col] = pd.to_numeric(df_sim_sample[col], errors='coerce')
        
    # Transform Simulation Sample
    logger.info("Transforming Sim Sample...")
    df_sim_feat = fe.transform(df_sim_sample)
    
    # Pick a specific race from history to analyze
    sim_race_id = df_sim_sample['race_id'].unique()[0]
    sim_race = df_sim_feat[df_sim_feat['race_id'] == sim_race_id]
    
    # Ensure all model features exist
    for c in model_features:
        if c not in sim_race.columns:
            sim_race[c] = np.nan
            
    sim_probs = model.predict_proba(sim_race[model_features])[:, 1]
    logger.info(f"Sim Race {sim_race_id}: Prob Sum = {sim_probs.sum():.4f}")
    
    # 5. Create Mock Bad Row (mimic Today's prediction)
    # R11 Horse 4: Odds 5.9, Pop 99 (Missing).
    
    mock_row = sim_race.iloc[0].copy()
    mock_row['odds'] = 5.9
    mock_row['popularity'] = 99
    
    # Manually set related features
    if 'log_odds' in model_features or 'log_odds' in mock_row:
        mock_row['log_odds'] = np.log1p(5.9)
    
    # Create DataFrame
    mock_df = pd.DataFrame([mock_row])
    # Ensure columns
    for c in model_features:
        if c not in mock_df.columns:
            mock_df[c] = 0
            
    mock_prob = model.predict_proba(mock_df[model_features])[:, 1][0]
    logger.info(f"Mock Row (Odds=5.9, Pop=99): Prob = {mock_prob:.4f}")
    
    # Feature Contribution Analysis
    logger.info("Analyzing Feature Contributions...")
    try:
        # Access the underlying LGBM Booster
        if hasattr(model, 'calibrated_classifiers_'):
            booster = model.calibrated_classifiers_[0].estimator.booster_
        else:
            booster = model.booster_
            
        contribs = booster.predict(mock_df[model_features], pred_contrib=True)
        # contribs is (n_samples, n_features + 1). Last one is bias.
        
        # Map back to feature names
        feature_contribs = list(zip(model_features, contribs[0][:-1]))
        feature_contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        bias = contribs[0][-1]
        logger.info(f"Base Score (Bias): {bias:.4f}")
        logger.info("Top 10 Contributors:")
        for name, score in feature_contribs[:10]:
            val = mock_row.get(name, 'N/A')
            logger.info(f"  {name}: {score:.4f} (Value: {val})")
            
    except Exception as e:
        logger.error(f"Failed to calculate contributions: {e}")

    # Try with Pop = 3 (Typical for Odds 5.9)
    mock_row_2 = mock_row.copy()
    mock_row_2['popularity'] = 3
    mock_prob_2 = model.predict_proba(pd.DataFrame([mock_row_2])[model_features])[:, 1][0]
    logger.info(f"Mock Row (Odds=5.9, Pop=3): Prob = {mock_prob_2:.4f}")

             
if __name__ == "__main__":
    main()
