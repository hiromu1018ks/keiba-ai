# Full ETL & 2024 Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EveryDB2から2015-2025年の全データをParquetにETLし、2024年のバックテストを実行する

**Architecture:** 既存のETLスクリプト (`run_etl.py --mode full`) を2015-2025年範囲で実行し、続いてバックテストスクリプト (`run_backtest.py`) を学習2020-2023 / テスト2024で実行する。コード変更は不要（パラメータ変更のみ）。

**Tech Stack:** Python, PostgreSQL (EveryDB2), Parquet (pyarrow/pandas), LightGBM

---

### Task 1: Full Mode ETL (2015-2025)

**Files:**
- Execute: `scripts/run_etl.py`
- Read: `data/raw/*.parquet`, `data/odds/jodds_tanpuku/` (出力)
- Config: `config/etl_tables.yaml`, `config/settings.yaml`

- [ ] **Step 1: ETL実行**

```bash
PGPASSWORD=aa8940aa python scripts/run_etl.py --mode full --start 20150101 --end 20251231
```

Expected: ログに各テーブルの行数が出力される。全103テーブルの処理が完了するまで5-10分。
失敗した場合: PostgreSQL接続エラー → `PGPASSWORD` 確認 / テーブル不存在 → `config/etl_tables.yaml` 確認

- [ ] **Step 2: ETL結果の検証**

```bash
python -c "
import pandas as pd, os
files = [
    ('data/raw/races.parquet', 'race_date'),
    ('data/raw/entries.parquet', 'race_date'),
    ('data/raw/payouts.parquet', 'race_date'),
    ('data/odds/odds_tanpuku.parquet', 'race_date'),
    ('data/odds/odds_wide.parquet', 'race_date'),
]
for f, col in files:
    df = pd.read_parquet(f, columns=[col])
    print(f'{f}: {df[col].min()} ~ {df[col].max()}, rows={len(df)}')

# jodds_tanpuku partitions
jodds = 'data/odds/jodds_tanpuku'
years = sorted([d for d in os.listdir(jodds) if d.startswith('year=')])
print(f'\njodds_tanpuku partitions: {years}')
print(f'  年数: {len(years)} (expected: 11 for 2015-2025)')
"
```

Expected:
- 全テーブルのmax dateが2025年末付近
- jodds_tanpukuパーティションが year=2015 ~ year=2025 の11年分

---

### Task 2: 2024 Backtest (学習: 2020-2023)

**Files:**
- Execute: `scripts/run_backtest.py`
- Output: `backtest_result.json` (プロジェクトルート)
- Dependencies: Task 1完了（Parquetデータ必須）

- [ ] **Step 1: バックテスト実行**

```bash
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

Expected: 学習~17分 + テスト~7分で完了。ROI、ベット数、最大DD等のメトリクスがコンソールに出力される。
失敗した場合: データ不在 → Task 1のETLを再実行 / メモリ不足 → jodds_tanpuku参照確認（学習時はロードされないはず）

- [ ] **Step 2: 結果の確認**

```bash
python -c "
import json
with open('backtest_result.json') as f:
    r = json.load(f)
print(f'ROI:        {r[\"total_roi\"]:.1%}')
print(f'Bets:       {r[\"total_bets\"]:,}')
print(f'Stake:      {r[\"total_stake\"]:,} yen')
print(f'Return:     {r[\"total_return\"]:,} yen')
print(f'Max DD:     {r[\"max_drawdown\"]:.1%}')
print(f'Bankroll:   {r[\"final_bankroll\"]:,} yen')
print(f'Train time: {r[\"train_seconds\"]}s')
print(f'Test time:  {r[\"test_seconds\"]}s')
"
```

Expected: ROIが0%以上、ベット数が数千件レベル。

---

### Task 3: MEMORY.md と CLAUDE.md の更新

**Files:**
- Modify: `C:\Users\hirom\.claude\projects\C--Users-hirom-develop-keiba-ai\memory\MEMORY.md`
- Modify: `CLAUDE.md` (パイプラインスクリプトの所要時間表)

- [ ] **Step 1: MEMORY.mdのバックテスト結果を更新**

バックテスト結果のROI、ベット数等をMEMORY.mdの該当セクションに反映。
旧エントリ（ROI 143.3%等）を今回の結果で上書き。

- [ ] **Step 2: CLAUDE.mdのパイプライン所 要時間を確認・更新**

バックテスト結果の `train_seconds` / `test_seconds` を確認し、CLAUDE.mdのスクリプト表の所 要時間が実際と大きく異なる場合は更新。
