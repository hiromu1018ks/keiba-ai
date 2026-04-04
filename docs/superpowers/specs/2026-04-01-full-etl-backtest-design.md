# Full ETL & 2024 Backtest Design

**Date**: 2026-04-01
**Status**: Approved

## Problem

2024年のバックテストを実行した際、2024年のレースデータがParquetファイルに存在しなかった。
原因: 前回のETLが `--start`/`--end` パラメータで2023年までしか抽出していなかった。

## Current State

| Parquet File | Date Range | Rows |
|---|---|---|
| `data/raw/races.parquet` | 2014-01-01 ~ 2023-12-31 | 57,948 |
| `data/raw/entries.parquet` | 2014-01-01 ~ 2023-12-31 | 710,241 |
| `data/raw/payouts.parquet` | 2015-01-04 ~ 2023-12-28 | 31,093 |
| `data/odds/odds_tanpuku.parquet` | 2015-01-04 ~ 2023-12-28 | 436,600 |
| `data/odds/odds_wide.parquet` | 2015-01-04 ~ 2023-12-28 | 2,961,832 |
| `data/odds/jodds_tanpuku/` | year=2015 ~ year=2023 | 66,515,211 |

EveryDB2 PostgreSQLには2024年（5,974レース）、2025年（5,142レース）のデータが存在する。

## Design

### Step 1: Full Mode ETL

```bash
PGPASSWORD=<password> python scripts/run_etl.py --mode full --start 20150101 --end 20251231
```

- raced テーブル（n_race, n_uma_race, n_harai, n_odds_tanpuku, n_odds_wide, n_jodds_tanpuku 等）を 2015-01-01 ~ 2025-12-31 で抽出
- jodds_tanpuku は year/month Hiveパーティションで再構築（全パーティション削除後に再作成）
- マスターテーブル（n_uma, n_kisyu_seiseki, n_chokyo_seiseki 等）は日付フィルタなしで全件ダンプ
- 各ファイルは `.parquet.tmp` → rename で原子書き込み
- **注意**: `--start 20150101` により既存の2014年データ（races, entries）はParquetから除外される。学習期間2020-2023 + HorseHistoryFeaturesの5年ルックバック（~2015以降）で2014データは使用されないため問題なし

### Step 2: 2024 Backtest

```bash
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

- 学習期間: 2020-2023（従来通り）
- テスト期間: 2024年
- 特徴量は学習時にParquetから毎回再計算（追加の特徴量生成ステップは不要）

## Expected Results

- ETL後のParquetファイルに2015-2025年の全データが含まれる
- バックテストで2024年のROI、ベット数、最大DD等のメトリクスが取得できる

## Risks

- **ディスク容量**: jodds_tanpuku が ~64MB 増加（問題なし）
- **ETL中断**: 原子書き込みにより既存ファイルは保持される
- **メモリ**: 学習時 jodds_tanpuku はロードされない（2.5GB回避済み）
