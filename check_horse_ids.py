import pandas as pd

# スクレイピングデータ
df = pd.read_csv('data/common/raw_data/results.csv')
df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')

# JRA-VANデータ  
jv = pd.read_csv('data/jra_van/race/2015-2025.csv', encoding='cp932', low_memory=False)
jv['date_jv'] = pd.to_datetime(
    jv['年'].astype(str).str.zfill(2).apply(lambda x: '20' + x if int(x) < 50 else '19' + x) + 
    jv['月'].astype(str).str.zfill(2) + 
    jv['日'].astype(str).str.zfill(2),
    format='%Y%m%d', errors='coerce'
)

# 馬IDの形式確認
df['horse_id_str'] = df['horse_id'].astype(str)
jv['horse_id_jv'] = jv['血統登録番号'].astype(str)

print("=== 馬ID形式の比較 ===")
print(f"Scraped horse_id sample: {df['horse_id_str'].head(10).tolist()}")
print(f"JRA-VAN horse_id sample: {jv['horse_id_jv'].head(10).tolist()}")

# 馬IDの長さ分布
print(f"\nScraped horse_id lengths: {df['horse_id_str'].str.len().value_counts().head()}")
print(f"JRA-VAN horse_id lengths: {jv['horse_id_jv'].str.len().value_counts().head()}")

# ユニーク馬ID
scraped_ids = set(df['horse_id_str'].unique())
jv_ids = set(jv['horse_id_jv'].unique())
common_ids = scraped_ids & jv_ids

print(f"\n=== 馬IDの一致 ===")
print(f"Scraped unique IDs: {len(scraped_ids)}")
print(f"JRA-VAN unique IDs: {len(jv_ids)}")  
print(f"Common IDs: {len(common_ids)}")
print(f"Match rate: {len(common_ids)/len(scraped_ids)*100:.1f}%")

# 共通IDがある場合、同じレースをマッチングできるか確認
if common_ids:
    test_id = list(common_ids)[0]
    print(f"\n=== テストID: {test_id} ===")
    
    s_data = df[df['horse_id_str'] == test_id].head(3)
    j_data = jv[jv['horse_id_jv'] == test_id].head(3)
    
    print("Scraped:")
    print(s_data[['horse_id_str', 'date_dt', 'race_id']].to_string())
    
    print("\nJRA-VAN:")
    print(j_data[['horse_id_jv', 'date_jv', '場所', 'レース番号']].to_string())
