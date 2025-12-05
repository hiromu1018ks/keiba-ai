import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from experiments.v2_feature_expansion.src.feature_engineering import FeatureEngineer

def main():
    # Load raw data
    results_path = 'data/common/raw_data/results.csv'
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    print(f"Loaded results: {len(df)} records")
    
    # Filter for faster check
    df = df.tail(1000).copy()
    
    # Run Feature Engineering
    fe = FeatureEngineer()
    
    # Fit Transform
    print("Running fit_transform...")
    df_transformed = fe.fit_transform(df)
    
    print("\n--- Feature Check ---")
    cols = df_transformed.columns.tolist()
    print(f"Total features: {len(cols)}")
    
    # Check for new pedigree features
    pedigree_cols = [c for c in cols if 'sire' in c or 'bms' in c or 'breeder' in c or 'owner' in c]
    print(f"\nPedigree Features ({len(pedigree_cols)}):")
    for c in pedigree_cols:
        print(f" - {c}")

    # Check for new connection features
    conn_cols = [c for c in cols if 'jockey_place' in c or 'trainer_place' in c or 'horse_jockey' in c]
    print(f"\nConnection Features ({len(conn_cols)}):")
    for c in conn_cols:
        print(f" - {c}")

    # Check for history features
    hist_cols = [c for c in cols if 'momentum' in c or 'avg_rank' in c or 'position_gain' in c]
    print(f"\nHistory Features ({len(hist_cols)}):")
    for c in hist_cols:
        print(f" - {c}")
        
    # Check if target encoding worked (not all NaN)
    if 'jockey_place_target_enc' in cols:
        sample = df_transformed[['jockey_place', 'jockey_place_target_enc']].head()
        print("\nSample Target Encoding for Jockey@Place:")
        print(sample)
    else:
        print("\njockey_place_target_enc not found!")

if __name__ == "__main__":
    main()
