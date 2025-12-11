import pandas as pd

# スクレイピングデータの期間
df = pd.read_csv('data/common/raw_data/results.csv')
df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
print('=== Scraped Data ===')
print(f"期間: {df['date_dt'].min()} ~ {df['date_dt'].max()}")

# JRA-VANデータの期間
jv = pd.read_csv('data/jra_van/race/2015-2025.csv', encoding='cp932')
jv['date_jv'] = pd.to_datetime(
    jv['年'].astype(str).str.zfill(2).apply(lambda x: '20' + x if int(x) < 50 else '19' + x) + 
    jv['月'].astype(str).str.zfill(2) + 
    jv['日'].astype(str).str.zfill(2),
    format='%Y%m%d', errors='coerce'
)
print(f'\n=== JRA-VAN Data ===')
print(f"期間: {jv['date_jv'].min()} ~ {jv['date_jv'].max()}")

# 重複期間
print(f'\n=== 重複期間 ===')
overlap_start = max(df['date_dt'].min(), jv['date_jv'].min())
overlap_end = min(df['date_dt'].max(), jv['date_jv'].max())
print(f"{overlap_start} ~ {overlap_end}")

# 重複するデータ数
df_overlap = df[(df['date_dt'] >= overlap_start) & (df['date_dt'] <= overlap_end)]
jv_overlap = jv[(jv['date_jv'] >= overlap_start) & (jv['date_jv'] <= overlap_end)]
print(f"\nScraped data in overlap period: {len(df_overlap)}/{len(df)} ({len(df_overlap)/len(df)*100:.1f}%)")
print(f"JRA-VAN data in overlap period: {len(jv_overlap)}/{len(jv)} ({len(jv_overlap)/len(jv)*100:.1f}%)")
