# Phase 35: HaronTimeL3/L4 相互排他性検証アプローチ

**Document:** ETL-05 Analysis
**Created:** 2026-05-19
**Status:** Analysis document (pre-ETL verification)

## 1. EveryDB2 スキーマ定義

### SE Table (UMA_RACE) — 馬毎レース情報

| Field | Column | Type | Size | Sentinels | Description |
|-------|--------|------|------|-----------|-------------|
| 58 | HaronTimeL4 | varchar | 3 | 000, 999 | 後4ハロンタイム (99.9秒) |
| 59 | HaronTimeL3 | varchar | 3 | 000, 999 | 後3ハロンタイム (99.9秒) |

- センチネル "000": 初期値 (データなし)
- センチネル "999": 出走取消/競走除外/発走除外/競走中止/タイムオーバー
- 有効値範囲: 33.0 ~ 40.0 秒程度 (0.1秒単位、varchar(3)なので "335" = 33.5秒)
- 障害レース: HaronTimeL3に1F平均タイムを設定、HaronTimeL4は初期値

### RA Table (RACE) — レース詳細

| Field | Column | Type | Size | Sentinels | Description |
|-------|--------|------|------|-----------|-------------|
| 96 | HaronTimeL3 | varchar | 3 | 000, 999 | 後3ハロン (ラップタイム後半3ハロン合計) |
| 97 | HaronTimeL4 | varchar | 3 | 000, 999 | 後4ハロン (ラップタイム後半4ハロン合計) |

- RA table の HaronTime は race-level (先頭馬の値)
- SE table の HaronTime は horse-level (各馬の値)

### データ格納形式

両テーブルとも varchar(3) で、値は "345" のように3桁の文字列。
float64 変換時は divisor=10 ではなく、そのまま float("34.5") ではない。
実際は "345" → 34.5秒 (÷10) の変換が必要だが、ETL Plan 35-01 では
`_TABLE_TYPE_RULES` の `sentinel_float` で sentinels ["000", "999"] をNaN化し、
残りの有効値を `pd.to_numeric` で変換する。

**注意:** EveryDB2 の HaronTimeL3/L4 は varchar(3) で "345" 形式 (÷10なし)。
値 "345" = 34.5秒。Plan 35-01 の sentinel_float ルールでは divisor=1 (デフォルト) で
処理するため、Parquetには345.0として格納される可能性がある。
Phase 36の特徴量計算で ÷10 の正規化を行う。

## 2. 相互排他性の4分類仮説

HaronTimeL3 と HaronTimeL4 は以下の4パターンに分類される:

| 分類 | HaronTimeL3 | HaronTimeL4 | 期待される割合 | 説明 |
|------|-------------|-------------|---------------|------|
| **L3のみ** | 有効値 | NaN | 高い (標準) | 標準的な平地レース。後半3ハロン (最後の600m) のタイム |
| **L4のみ** | NaN | 有効値 | 低い (一部) | 一部競馬場/条件/過去データ。後半4ハロン (最後の800m) |
| **両方あり** | 有効値 | 有効値 | 中程度 | 標準的。距離により両方計測されるケース |
| **両方なし** | NaN | NaN | 中程度 | 出走取消/障害/データなし |

### SE table (馬毎) の分類期待値

- **L3のみ**: 最も多い。現在はL3が標準 (スキーマ説明に「基本的には後3ハロンのみ設定」と記載)
- **L4のみ**: 過去データの一部。スキーマ説明に「過去分のデータは後4ハロンが設定されているものもある」と記載
- **両方あり**: 現在は稀。過去データで両方設定されているケース
- **両方なし**: 出走取消/競走除外/競走中止/タイムオーバー/障害レース

### RA table (レース) の分類期待値

RA table の HaronTime は race-level の集約値。SE table と同様の傾向だが、
レース単位なのでNaN率は異なる可能性がある。

## 3. ETL 実行後の Claude 検証手順 (D-03, D-06)

以下の手順は、Plan 35-01 のコード変更がmainにマージされ、
`run_etl.py --mode full` が実行された後に実施する。

### Step 1: SE table (entries.parquet) の分類集計

```python
import pandas as pd

df = pd.read_parquet("data/raw/entries.parquet")

# HaronTimeL3/L4 の型確認
print("dtype L3:", df["harontimel3"].dtype)
print("dtype L4:", df["harontimel4"].dtype)

# 4分類集計
l3_valid = df["harontimel3"].notna()
l4_valid = df["harontimel4"].notna()

both = (l3_valid & l4_valid).sum()
l3_only = (l3_valid & ~l4_valid).sum()
l4_only = (~l3_valid & l4_valid).sum()
neither = (~l3_valid & ~l4_valid).sum()
total = len(df)

print(f"\nSE Table HaronTime Distribution:")
print(f"  L3 only:  {l3_only:>8d} ({l3_only/total:.1%})")
print(f"  L4 only:  {l4_only:>8d} ({l4_only/total:.1%})")
print(f"  Both:     {both:>8d} ({both/total:.1%})")
print(f"  Neither:  {neither:>8d} ({neither/total:.1%})")
print(f"  Total:    {total:>8d}")

# NaN率
print(f"\nNaN rate L3: {df['harontimel3'].isna().mean():.1%}")
print(f"NaN rate L4: {df['harontimel4'].isna().mean():.1%}")

# 有効値の統計
print(f"\nL3 valid stats:")
print(df["harontimel3"].dropna().describe())
print(f"\nL4 valid stats:")
print(df["harontimel4"].dropna().describe())

# センチナル残存確認 (0.0, 999.0 が存在してはならない)
print(f"\nL3 == 0.0: {(df['harontimel3'] == 0.0).sum()}")
print(f"L3 == 999.0: {(df['harontimel3'] == 999.0).sum()}")
print(f"L4 == 0.0: {(df['harontimel4'] == 0.0).sum()}")
print(f"L4 == 999.0: {(df['harontimel4'] == 999.0).sum()}")
```

### Step 2: RA table (races.parquet) の分類集計

```python
df_races = pd.read_parquet("data/raw/races.parquet")

l3_valid = df_races["harontimel3"].notna()
l4_valid = df_races["harontimel4"].notna()

both = (l3_valid & l4_valid).sum()
l3_only = (l3_valid & ~l4_valid).sum()
l4_only = (~l3_valid & l4_valid).sum()
neither = (~l3_valid & ~l4_valid).sum()
total = len(df_races)

print(f"\nRA Table HaronTime Distribution:")
print(f"  L3 only:  {l3_only:>8d} ({l3_only/total:.1%})")
print(f"  L4 only:  {l4_only:>8d} ({l4_only/total:.1%})")
print(f"  Both:     {both:>8d} ({both/total:.1%})")
print(f"  Neither:  {neither:>8d} ({neither/total:.1%})")
```

### Step 3: 結果をこのファイルに追記

上記スクリプトの出力を以下のセクションに貼り付ける:

```
## 4. 検証結果 (ETL実行後に記入)

*(ETL実行後にここに結果を記載)*
```

## 4. Phase 36 統合ロジックへの引き渡し事項

### harontime_last3f の決定

Phase 36 では HaronTimeL3/L4 を統合して `harontime_last3f` 特徴量を生成する。
このセクションの分析結果に基づいて以下のいずれかを採用する:

| 候補 | 方法 | メリット | デメリット |
|------|------|---------|-----------|
| **A: coalesce(L3, L4)** | L3を優先、L3がNaNならL4 | 簡単、欠損最小化 | L3とL4で単位が異なる可能性 |
| **B: 距離別選択** | 短距離はL3、長距離はL4 | 物理的に意味が合う | 複雑、閾値の決定が必要 |
| **C: L3のみ** | L3のみ使用、L4は無視 | 単純、標準的 | L4のみのデータを捨てる |
| **D: 両方を別特徴量** | L3, L4 それぞれ独立に特徴量化 | 情報損失なし | 欠損率が高い特徴量になる |

### Phase 36 で必要な判断

1. HaronTimeL3/L4 の Parquet 格納形式 (345.0 → 34.5秒変換) をどのタイミングで行うか
2. SE table (馬毎) と RA table (レース) のどちらを特徴量のソースとするか
3. 過去走の HaronTime 平均/分散をどう計算するか (PIT安全性の確保)

---

*Phase: 35-ETL Data Foundation*
*Created: 2026-05-19*
