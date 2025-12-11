import pandas as pd
from src.features.jravan_features import JraVanFeatures

# テスト
df = pd.read_csv('data/common/raw_data/results.csv', nrows=1000)
print('Before merge:')
print(f"  horse_id sample: {df['horse_id'].head(3).tolist()}")

jv = JraVanFeatures()
df_merged = jv.merge_with_scraped_data(df)

print(f'\nAfter merge:')
if 'prev_pci' in df_merged.columns:
    non_null = df_merged['prev_pci'].notna().sum()
    print(f"  prev_pci: {non_null}/{len(df_merged)} ({non_null/len(df_merged)*100:.1f}%)")
else:
    print("  prev_pci NOT FOUND")
