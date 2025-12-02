import pandas as pd
import numpy as np
from experiments.v2_feature_expansion.src.feature_engineering import FeatureEngineer

def test_feature_engineering():
    # Create dummy data
    data = {
        'race_id': ['202301010101', '202301010101', '202301020101', '202301020101'],
        'date': ['2023年1月5日', '2023年1月5日', '2023年2月5日', '2023年2月5日'],
        'horse_id': ['H1', 'H2', 'H1', 'H2'],
        'rank': [1, 2, 2, 1],
        'prize': [100.0, 50.0, 50.0, 100.0],
        'margin': ['1/2', 'ハナ', '大差', '1 1/4'],
        'jockey_id': ['J1', 'J2', 'J1', 'J2'],
        'trainer_id': ['T1', 'T2', 'T1', 'T2'],
        'place': ['Tokyo', 'Tokyo', 'Nakayama', 'Nakayama'],
        'distance': [1600, 1600, 2000, 2000],
        'surface': ['芝', '芝', 'ダ', 'ダ'],
        'weather': ['晴', '晴', '雨', '雨'],
        'condition': ['良', '良', '重', '重'],
        'horse_weight': [480, 500, 482, 498],
        'age': [3, 3, 3, 3],
        'odds': [2.5, 3.0, 4.0, 1.5],
        'passing_order': ['1-1', '2-2', '1-1', '2-2']
    }
    df = pd.DataFrame(data)
    
    fe = FeatureEngineer()
    
    # Test Margin Conversion
    print("Testing Margin Conversion...")
    margins = ['1/2', 'ハナ', '大差', '1 1/4', '同着', 'クビ', 'アタマ']
    for m in margins:
        print(f"Margin '{m}' -> {fe._convert_margin(m)}")
        
    # Test Transform
    print("\nTesting Transform...")
    df_transformed = fe.transform(df)
    
    print("Columns:", df_transformed.columns)
    
    # Check Interval
    print("\nInterval for H1:")
    print(df_transformed[df_transformed['horse_id'] == 'H1'][['date', 'interval']])
    
    # Check Seasonality
    print("\nSeasonality:")
    print(df_transformed[['date', 'month_sin', 'month_cos']].head())
    
    # Check Rolling Stats (might be NaN for first races)
    print("\nRolling Stats (Rank 5 Std):")
    print(df_transformed[['horse_id', 'rank_5_std']])
    
    # Check Margin Val
    print("\nMargin Values:")
    print(df_transformed[['margin', 'margin_val']])

    # Check Jockey Stats
    print("\nJockey Stats:")
    print(df_transformed[['jockey_id', 'jockey_win_rate_100', 'jockey_place_rate_100']])

if __name__ == "__main__":
    test_feature_engineering()
