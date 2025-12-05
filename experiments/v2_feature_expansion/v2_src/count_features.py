import pandas as pd
import numpy as np
from experiments.v2_feature_expansion.src.feature_engineering import FeatureEngineer
from experiments.v2_feature_expansion.src.simulate import Backtester

def count_features():
    # Create dummy data with all necessary columns
    data = {
        'race_id': ['202301010101'],
        'date': ['2023年1月5日'],
        'horse_id': ['H1'],
        'rank': [1],
        'prize': [100.0],
        'margin': ['1/2'],
        'jockey_id': ['J1'],
        'trainer_id': ['T1'],
        'place': ['Tokyo'],
        'distance': [1600],
        'surface': ['芝'],
        'weather': ['晴'],
        'condition': ['良'],
        'horse_weight': [480],
        'weight_change': [0],
        'age': [3],
        'odds': [2.5],
        'popularity': [1],
        'impost': [56.0],
        'passing_order': ['1-1'],
        'bracket': [1],
        'horse_num': [1],
        'gender': ['牡'],
        'around': ['右'],
        'race_class': ['OP']
    }
    df = pd.DataFrame(data)
    
    fe = FeatureEngineer()
    # Fit transform to generate all columns
    df_transformed = fe.fit_transform(df)
    
    # Define exclude_cols as in simulate.py
    exclude_cols = ['race_id', 'date', 'time', 'rank', 'target', 'date_dt', 'year',
                    'horse_name', 'jockey', 'trainer', 'horse_id', 'jockey_id', 'trainer_id', 'time_seconds',
                    'prize', 'surface', 'distance', 'weather', 'condition', 'jockey_trainer_pair', 'winner_time',
                    'around', 'race_class', 'place', 'passing_order', 'margin', 'margin_val', 'running_style', 'log_prize']
    
    feature_cols = [c for c in df_transformed.columns if c not in exclude_cols]
    
    print(f"Total Features: {len(feature_cols)}")
    print("Features List:")
    for c in feature_cols:
        print(f"- {c}")

if __name__ == "__main__":
    count_features()
