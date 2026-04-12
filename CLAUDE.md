# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

競馬AI予測システム v5.5 — 統計的 horse racing prediction system (単勝/複勝/ワイド)。
LightGBM + scikit-learn で ML モデルを構築し、PostgreSQL (EveryDB2/JRA-VAN DataLab) をデータソースとする。
MLflow で実験管理。設計書は `docs/design.md` (v5.5, ~2900行)。

## Development Environment

### Python (mise)

```bash
# Python 3.11 を mise でインストール・アクティベート
mise install
mise activate

# 依存インストール
pip install -e ".[dev]"
```

`mise.toml` で Python 3.11 を固定。`pip install -e ".[dev]"` で ruff, mypy, ipykernel も含む。

### Common Commands

```bash
# テスト実行（DB不要、全テスト mock 使用）
python -m pytest tests/ -v

# 単一テストファイル
python -m pytest tests/test_domain.py -v

# カバレッジ付き
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# リント (ruff)
ruff check src/ tests/

# フォーマット
ruff format --check src/ tests/

# 型チェック (mypy)
mypy src/
```

### Database

PostgreSQL が `localhost:5432/everydb2` で稼働前提。パスワードは環境変数 `PGPASSWORD` で上書き。
設定ファイル: `config/settings.yaml`

## Architecture

### Data Layer (Parquet-based)

```
EveryDB2外部テーブル → PostgreSQL (ETL入力のみ) → Parquetファイル群
                                                      ↓
                         ParquetStore → DataRepository → MLパイプライン
```

### Class Structure

- **`ParquetStore`** (`src/db/parquet_store.py`) — Parquetファイルの読み書き。単一ファイル + 年/月パーティション対応。pyarrow述語プッシュダウン。
- **`DataRepository`** (`src/db/repository.py`) — MLパイプラインの唯一のデータアクセス窓口。日付フィルタ・障害除外・キャッシュ制御。
- **`DatabaseConnection`** (`src/db/connection.py`) — PostgreSQL ETL専用。EveryDB2 → Parquet への書き出し。

### Parquet Files

```
data/raw/races.parquet, entries.parquet, payouts.parquet
data/odds/snapshots.parquet, time_series/ (年/月パーティション), wide.parquet
data/features/horse_features.parquet  (特徴量キャッシュ)
data/predictions/predictions.parquet
data/bets/bets.parquet
```

全テーブルに `race_date` (datetime64) 列を含む。
`race_id` は `_compute_race_id()` でpandas計算（PostgreSQL GENERATED COLUMN不使用）。

### Key Dependencies

- `src/db/parquet_store.py` — pyarrow, pandas
- `src/db/repository.py` — ParquetStore, pandas
- `src/db/connection.py` — SQLAlchemy Core, ParquetStore, pandas

### Consumer Migration

全MLパイプラインコンポーネントは `DataRepository` を使用:
`TrainingPipelineV5`, `BacktestEngine`, `JVLinkFetcher`, `OddsCollector`, `BacktestValidationSuite`

### import path

- `from db.repository import DataRepository` — MLパイプライン用
- `from db.parquet_store import ParquetStore` — 低レベルI/O
- `from db.connection import DatabaseConnection` — ETL専用
- `from domain.types import ...` (pythonpath = `[".", "src"]` 設定済み)

### Planned Phases (design.md 参照)

Phase A (done) → B (features/) → C (models/) → D (betting/) → E (backtest/) → F (automation/, monitoring/)

## Key Design Decisions

- **2段階モデル**: P(hit) × E(odds|hit) でゼロ偏重を排除。Stage1=能力モデル, Stage2=的中時払戻回帰
- **Market Model**: 差分 log_error のみ出力（p_market_pred は Stage2 に入れない）
- **EV補正**: P補正 (binary objective, init_score=logit(p_pred)) × E補正 (weight=1/√p) の2モデルに分解
- **RegimeDetector**: 市場指標 (fav_rate × overround) で 3状態分類 (aggressive/conservative/collapsed)
- **サブモデル**: 芝/ダートの2分割のみ

## Code Style

- Ruff: target py311, line-length=100, rules=E/F/I/N/W
- Mypy: `disallow_untyped_defs = true` (全関数に型アノテーション必須)
- テストは DB不要 (全て mock) — `unittest.mock` を使用
- コミットメッセージ: Conventional Commits (日本語)

## Pipeline Scripts (実行順序)

```bash
# 環境変数（PostgreSQLパスワード）
export PGPASSWORD=<password>

# Step 1: ETL — PostgreSQL (EveryDB2) → Parquet
python scripts/run_etl.py --start 20140101 --end 20231231

# Step 2: 学習 — Parquet → 特徴量生成 → LightGBM Ranker + 補正モデル
python scripts/run_train.py --start 20200101 --end 20231231

# Step 3a: バックテスト (単一年度)
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231

# Step 3b: バックテスト (マルチ年度)
python scripts/run_backtest.py \
  --years 2023 2024 2025 \
  --train-window 4
```

### 各スクリプトの詳細

| スクリプト | 役割 | 入力 | 出力 | 所要時間 |
|-----------|------|------|------|---------|
| `scripts/run_etl.py` | PostgreSQL→Parquet抽出 | EveryDB2外部テーブル | `data/raw/*.parquet`, `data/odds/*.parquet` | ~10分 |
| `scripts/run_train.py` | MLモデル学習 | Parquetファイル群 | MLflowモデル, 特徴量キャッシュ | ~44分 |
| `scripts/run_backtest.py` | 学習+バックテスト (単一年度/マルチ年度) | Parquet + 学習済みモデル | `backtest_result.json` または `data/backtest/multi_year_result.json` | ~57分/年 |

### 直近のバックテスト結果 (2024年テスト)

- 回収率: 89.0% (赤字 — 100円あたり89円回収)
- ベット数: 9,074 / 投資額: 907,400円 / 払戻額: 807,400円
- 学習: 2020-2023 / テスト: 2024
- Parquetデータ範囲: 2015-2025

- PostgreSQL GENERATED列（`distance_band`, `surface` via `track_cd`）はParquet ETLに含まれない → `FeatureEngine._map_basic_features()` でPython再計算
- Phase 1プレースホルダー: `haron_time_zscore_avg` は常にNaN → LightGBMはNaN処理可能だが `PlaceAbilityModel.train()` の `dropna()` で除外済み
- `run_backtest.py` は毎回学習し直す設計（再現性保証）。モデルは `data/models-backtest/` に保存 (本番 `data/models/` は上書きしない)
- `run_multi_year_backtest.py` は廃止。`run_backtest.py --years` に統合済み

## Configuration

`config/settings.yaml` — database, paths, logging, feature_engine, late_money, submodel の設定。
DB接続時は `_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` でプロジェクトルートを解決。
