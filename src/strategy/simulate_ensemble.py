"""
Simulation with Ensemble Model (LightGBM + CatBoost)
"""
import pandas as pd
import os
import pickle
import numpy as np
from src.features.engineer import FeatureEngineer
from src.strategy.betting import BettingStrategy
from src.model.ensemble import EnsembleModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_ensemble_simulation(start_year=2023):
    """Run simulation using the ensemble model."""
    
    # Load predictions cache
    pred_path = 'simulation_predictions.csv'
    if not os.path.exists(pred_path):
        logger.error("simulation_predictions.csv not found. Run simulate.py first.")
        return
    
    df = pd.read_csv(pred_path)
    
    # Load ensemble model
    model_path = 'models/ensemble_model.pkl'
    if not os.path.exists(model_path):
        logger.error("Ensemble model not found.")
        return
        
    with open(model_path, 'rb') as f:
        ensemble = pickle.load(f)
    
    # Load feature engineer
    fe_path = 'models/feature_engineer_ensemble.pkl'
    with open(fe_path, 'rb') as f:
        fe = pickle.load(f)
    
    # Get features used by ensemble (same as train)
    future_info = [
        'rank', 'time', 'time_seconds', 'target', 'prize', 'log_prize',
        'passing_order', 'agari_3f', 'margin', 'margin_val',
        'corner_1コーナー', 'corner_2コーナー', 'corner_3コーナー', 'corner_4コーナー', 'race_laps',
        'running_style', 'position_gain', 'first_corner_pos', 'final_corner_pos',
        'odds', 'popularity', 'log_odds', 
        'is_win', 'is_place', 'is_show', 'pred_prob'
    ]
    
    meta_cols = [
        'race_id', 'date', 'date_dt', 'horse_name', 'jockey', 'trainer', 
        'horse_id', 'jockey_id', 'trainer_id', 'owner', 'owner_id', 
        'sire_id', 'dam_id', 'bms_id', 'breeder_id', 'year', 'place_code',
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
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    final_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    # Predict with ensemble
    X = df[final_feature_cols].fillna(-999)
    ensemble_probs = ensemble.predict_proba(X)[:, 1]
    df['ensemble_prob'] = ensemble_probs
    
    # Run simulation with EV > 2.0
    logger.info("=== Ensemble Model Simulation (EV > 2.0) ===")
    
    for year in sorted(df['year'].unique()):
        if year < start_year:
            continue
            
        year_df = df[df['year'] == year].copy()
        year_df['ev'] = year_df['ensemble_prob'] * year_df['odds']
        
        # Bets: EV > 2.0, odds <= 100, prob >= 0.05
        bets = year_df[(year_df['ev'] > 2.0) & (year_df['odds'] <= 100) & (year_df['ensemble_prob'] >= 0.05)]
        
        bet_count = len(bets)
        bet_amount = bet_count * 100
        
        wins = bets[bets['rank'] == 1]
        return_amount = (wins['odds'] * 100).sum()
        
        recovery = (return_amount / bet_amount * 100) if bet_amount > 0 else 0
        
        logger.info(f'{int(year)}: Bet {bet_amount:,.0f}円 → Return {return_amount:,.0f}円 | 回収率 {recovery:.1f}%')
    
    # Compare with baseline
    logger.info("\n=== Baseline (LightGBM Only) ===")
    for year in sorted(df['year'].unique()):
        if year < start_year:
            continue
            
        year_df = df[df['year'] == year].copy()
        year_df['ev'] = year_df['pred_prob'] * year_df['odds']
        
        bets = year_df[(year_df['ev'] > 2.0) & (year_df['odds'] <= 100) & (year_df['pred_prob'] >= 0.05)]
        
        bet_count = len(bets)
        bet_amount = bet_count * 100
        
        wins = bets[bets['rank'] == 1]
        return_amount = (wins['odds'] * 100).sum()
        
        recovery = (return_amount / bet_amount * 100) if bet_amount > 0 else 0
        
        logger.info(f'{int(year)}: Bet {bet_amount:,.0f}円 → Return {return_amount:,.0f}円 | 回収率 {recovery:.1f}%')


if __name__ == "__main__":
    run_ensemble_simulation()
