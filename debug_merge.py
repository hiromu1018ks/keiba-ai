"""
全データでマージ0%になる根本原因を徹底調査
5000件では成功、37万件では失敗する理由を特定
"""
import pandas as pd
import numpy as np
from src.features.jravan_features import JraVanFeatures

print("=" * 60)
print("Step 1: 全データの読み込みと前処理")
print("=" * 60)

# 全データ読み込み
df = pd.read_csv('data/common/raw_data/results.csv')
print(f"全データ: {len(df)} rows")

# 日付パース
df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
df['year'] = df['date_dt'].dt.year

print(f"date_dt isna count: {df['date_dt'].isna().sum()}")
print(f"date_dt non-null count: {df['date_dt'].notna().sum()}")

# JRAフィルター
jra_places = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
df['place_code'] = df['race_id'].astype(str).str[4:6]
df = df[df['place_code'].isin(jra_places)].copy()
df = df.sort_values(['date_dt', 'race_id'])

print(f"JRAフィルター後: {len(df)} rows")
print(f"date_dt isna count after filter: {df['date_dt'].isna().sum()}")

print("\n" + "=" * 60)
print("Step 2: JRA-VAN特徴量の準備")
print("=" * 60)

jv = JraVanFeatures()
jv.load_data()
jv_features = jv.extract_features_for_merge()

print(f"JRA-VAN特徴量: {len(jv_features)} rows")
print(f"date_jv sample: {jv_features['date_jv'].head(3).tolist()}")
print(f"date_jv year sample: {jv_features['date_jv'].dt.year.head(3).tolist()}")

print("\n" + "=" * 60)
print("Step 3: マージキー値の詳細比較")
print("=" * 60)

# スクレイピングデータのキー値
df['horse_id_str'] = df['horse_id'].astype(str)
race_id_str = df['race_id'].astype(str)
df['race_no_scraped'] = race_id_str.str[-2:].astype(int)
df['place_code_scraped'] = race_id_str.str[4:6]
df['year_scraped'] = race_id_str.str[:4].astype(int)

print("=== Scraped Data Keys ===")
print(f"horse_id_str dtype: {df['horse_id_str'].dtype}")
print(f"horse_id_str sample: {df['horse_id_str'].head(5).tolist()}")
print(f"date_dt sample: {df['date_dt'].head(5).tolist()}")
print(f"place_code_scraped sample: {df['place_code_scraped'].head(5).tolist()}")
print(f"race_no_scraped sample: {df['race_no_scraped'].head(5).tolist()}")
print(f"year_scraped sample: {df['year_scraped'].head(5).tolist()}")

print("\n=== JRA-VAN Data Keys ===")
jv_features = jv_features.copy()
jv_features['date_jv'] = pd.to_datetime(jv_features['date_jv'])
jv_features['year_jv'] = jv_features['date_jv'].dt.year

print(f"horse_id_jv dtype: {jv_features['horse_id_jv'].dtype}")
print(f"horse_id_jv sample: {jv_features['horse_id_jv'].head(5).tolist()}")
print(f"date_jv sample: {jv_features['date_jv'].head(5).tolist()}")
print(f"place_code_jv sample: {jv_features['place_code_jv'].head(5).tolist()}")
print(f"レース番号 sample: {jv_features['レース番号'].head(5).tolist()}")
print(f"year_jv sample: {jv_features['year_jv'].head(5).tolist()}")

print("\n" + "=" * 60)
print("Step 4: キー値の共通部分を確認")
print("=" * 60)

# 馬IDの共通部分
scraped_horse_ids = set(df['horse_id_str'].unique())
jv_horse_ids = set(jv_features['horse_id_jv'].unique())
common_horse_ids = scraped_horse_ids & jv_horse_ids
print(f"Scraped unique horse_ids: {len(scraped_horse_ids)}")
print(f"JRA-VAN unique horse_ids: {len(jv_horse_ids)}")
print(f"Common horse_ids: {len(common_horse_ids)}")

# 年の共通部分
scraped_years = set(df['year_scraped'].unique())
jv_years = set(jv_features['year_jv'].unique())
common_years = scraped_years & jv_years
print(f"\nScraped unique years: {sorted(scraped_years)}")
print(f"JRA-VAN unique years: {sorted(jv_years)}")
print(f"Common years: {sorted(common_years)}")

# 場所コードの共通部分
scraped_places = set(df['place_code_scraped'].unique())
jv_places = set(jv_features['place_code_jv'].dropna().unique())
print(f"\nScraped unique place_codes: {sorted(scraped_places)}")
print(f"JRA-VAN unique place_codes: {sorted(jv_places)}")

print("\n" + "=" * 60)
print("Step 5: 特定の馬でマッチングテスト")
print("=" * 60)

# 共通の馬IDを1つ取って詳細比較
if common_horse_ids:
    test_horse_id = list(common_horse_ids)[0]
    print(f"Test horse_id: {test_horse_id}")
    
    scraped_horse = df[df['horse_id_str'] == test_horse_id].head(3)
    jv_horse = jv_features[jv_features['horse_id_jv'] == test_horse_id].head(3)
    
    print(f"\nScraped data for this horse:")
    print(scraped_horse[['horse_id_str', 'date_dt', 'place_code_scraped', 'race_no_scraped', 'year_scraped']].to_string())
    
    print(f"\nJRA-VAN data for this horse:")
    print(jv_horse[['horse_id_jv', 'date_jv', 'place_code_jv', 'レース番号', 'year_jv']].to_string())

print("\n" + "=" * 60)
print("Step 6: 実際のマージテスト（限定データ）")
print("=" * 60)

# 共通の馬IDのみでテスト
df_common = df[df['horse_id_str'].isin(common_horse_ids)].head(1000)
print(f"Common horse_ids only (first 1000): {len(df_common)} rows")

# マージ実行
df_test = df_common.merge(
    jv_features,
    left_on=['horse_id_str', 'date_dt', 'place_code_scraped', 'race_no_scraped'],
    right_on=['horse_id_jv', 'date_jv', 'place_code_jv', 'レース番号'],
    how='left',
    suffixes=('', '_jv_dup')
)
merged_count = df_test['prev_pci'].notna().sum()
print(f"Merge result (date-based): {merged_count}/{len(df_test)} ({merged_count/len(df_test)*100:.1f}%)")

# 年ベースでマージ
df_test2 = df_common.merge(
    jv_features,
    left_on=['horse_id_str', 'year_scraped', 'place_code_scraped', 'race_no_scraped'],
    right_on=['horse_id_jv', 'year_jv', 'place_code_jv', 'レース番号'],
    how='left',
    suffixes=('', '_jv_dup')
)
merged_count2 = df_test2['prev_pci'].notna().sum()
print(f"Merge result (year-based): {merged_count2}/{len(df_test2)} ({merged_count2/len(df_test2)*100:.1f}%)")

# 馬IDのみでマージ
df_test3 = df_common.merge(
    jv_features[['horse_id_jv', 'prev_pci']].drop_duplicates('horse_id_jv'),
    left_on='horse_id_str',
    right_on='horse_id_jv',
    how='left'
)
merged_count3 = df_test3['prev_pci'].notna().sum()
print(f"Merge result (horse_id only): {merged_count3}/{len(df_test3)} ({merged_count3/len(df_test3)*100:.1f}%)")

print("\n" + "=" * 60)
print("調査完了")
print("=" * 60)
