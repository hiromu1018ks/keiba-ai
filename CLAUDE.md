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

## Pipeline Scripts

### run_etl.py — ETL (PostgreSQL → Parquet)

```bash
python scripts/run_etl.py --mode full --start 20140101 --end 20251231
python scripts/run_etl.py --mode delta
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--mode` | `full`\|`delta` | **必須** | `full`=全量抽出、`delta`=差分マージ |
| `--start` | YYYYMMDD | — | 開始日 (`full`で必須) |
| `--end` | YYYYMMDD | — | 終了日 (`full`で必須) |
| `--tables` | list[str] | — | 対象テーブル (省略時は全テーブル) |

所要時間: ~10分 (full)

### run_train.py — モデル学習

```bash
python scripts/run_train.py --start 20200101 --end 20231231 --ensemble
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--start` | YYYYMMDD | **必須** | 学習開始日 |
| `--end` | YYYYMMDD | **必須** | 学習終了日 |
| `--ensemble` | flag | False | StackedEnsemble有効化 (LGBM+XGB+CB→Ridge) |
| `--experiment` | str | `keiba-v5` | MLflow実験名 |

所要時間: ~17分 / 出力: MLflowモデル + `data/features/horse_features.parquet`

### run_backtest.py — バックテスト

学習→テスト期間の投資シミュレーション。毎回学習し直す設計（再現性保証）。モデルは `data/models-backtest/` に保存。

```bash
# 単一年度
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231 --ensemble

# マルチ年度
python scripts/run_backtest.py --years 2023 2024 2025 --train-window 4 --ensemble

# Optuna最適化済みパラメータで検証
python scripts/run_backtest.py \
  --ensemble --strategy-manifest data/strategy_manifest.json \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--train-start` | YYYYMMDD | — | 学習開始日 (単一年度モード) |
| `--train-end` | YYYYMMDD | — | 学習終了日 (単一年度モード) |
| `--test-start` | YYYYMMDD | — | テスト開始日 (単一年度モード) |
| `--test-end` | YYYYMMDD | — | テスト終了日 (単一年度モード) |
| `--years` | list[int] | — | テスト年度リスト (マルチ年度モード) |
| `--train-window` | int | 4 | 学習年数 (マルチ年度モード) |
| `--ensemble` | flag | False | アンサンブルモデル有効化 |
| `--betting-mode` | `flat`\|`kelly` | `flat` | `flat`=100円固定、`kelly`=Fractional Kelly |
| `--betting-target` | `win`\|`place`\|`wide` | `win` | 投票対象 |
| `--report` | flag | False | HTMLレポート + parquet出力 |
| `--skip-train` | flag | False | 学習スキップ (キャッシュモデル使用、`--ensemble`必須) |
| `--profile` | flag | False | pyinstrumentプロファイリング (`data/profiles/`) |
| `--strategy-manifest` | str | — | Optuna最適化済みmanifest JSON (`--ensemble`必須) |

**モード切替:** `--years`指定→マルチ年度、4つのtrain/test指定→単一年度。いずれか必須。
**出力:** `backtest_result.json`、`data/validation/validation_report.json`、`data/backtest/bt_{year}_*.{csv,parquet}`
所要時間: ~57分/年

### run_wf_validation.py — ウォークフォワード検証

2-fold WF検証で過学習を検出。Feature importance安定性 (Spearman rho) とROI gapを測定。

```bash
python scripts/run_wf_validation.py --ensemble
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--ensemble` | flag | False | アンサンブルモデル有効化 |
| `--betting-target` | `win`\|`place`\|`wide` | `win` | 投票対象 |
| `--profile` | flag | False | pyinstrumentプロファイリング |

**Fold定義 (ハードコード):** Fold0=train 2020-2023/test 2024、Fold1=train 2021-2024/test 2025
**出力:** `data/backtest/wf_validation_result.json`
**判定:** `roi_gap_verdict`, `consistency_verdict`, `stability_verdict`, `overall_verdict`
所要時間: ~2時間/fold (~4時間合計)

### run_strategy_optimization.py — Optuna戦略パラメータ最適化

学習済みモデルに対して16次元戦略パラメータをOptuna TPEで最適化。

```bash
# 単一seed
python scripts/run_strategy_optimization.py \
  --n-trials 100 --models-dir data/models-backtest \
  --output data/strategy_manifest.json

# multi-seed安定性検証
python scripts/run_strategy_optimization.py \
  --n-trials 100 --seeds 42,43,44 \
  --models-dir data/models-backtest \
  --output data/stability/strategy_manifest.json
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--n-trials` | int | 100 | Optuna試行回数 |
| `--seed` | int | 42 | TPESampler乱数シード (単一seed時) |
| `--models-dir` | str | `data/models` | 学習済みモデルディレクトリ |
| `--output` | str | `data/strategy_manifest.json` | 出力manifestパス |
| `--min-bets` | int | 1000 | 1foldあたりの最低ベット数 |
| `--seeds` | str | — | multi-seed安定性検証 (カンマ区切り、例: `42,43,44`) |

**16次元:** fk_aggressive/conservative, ev_aggressive/conservative, edge_aggressive/conservative, dd_threshold_1/2, multiplier_reduced, rolling_window, min_stay_races, target_ev, max_scale, roi_threshold, ev_lower_threshold_turf/dirt
**4fold WF:** fold_start_year=2022, 各foldでRegimeDetector状態リセット
**MedianPruner:** 不良trialを早期切断
所要時間: ~2.5h/trial

### run_tuning.py — ハイパーパラメータチューニング

Optunaで個別モデルのハイパーパラメータを最適化。

```bash
python scripts/run_tuning.py --model win_hit --start 20200101 --end 20231231 --trials 50
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--model` | `win_hit`\|`win_return`\|`place_hit`\|`place_return`\|`ability` | **必須** | 対象モデル |
| `--start` | YYYYMMDD | **必須** | 学習開始日 |
| `--end` | YYYYMMDD | **必須** | 学習終了日 |
| `--trials` | int | 50 | Optuna試行回数 |

**出力:** `data/tuning/{model}_best_params.json`
所要時間: ~30分 (50 trials)

### 備考

- PostgreSQL GENERATED列（`distance_band`, `surface` via `track_cd`）はParquet ETLに含まれない → `FeatureEngine._map_basic_features()` でPython再計算
- Phase 1プレースホルダー: `haron_time_zscore_avg` は常にNaN → LightGBMはNaN処理可能だが `PlaceAbilityModel.train()` の `dropna()` で除外済み
- `run_multi_year_backtest.py` は廃止。`run_backtest.py --years` に統合済み

## Configuration

`config/settings.yaml` — database, paths, logging, feature_engine, late_money, submodel の設定。
DB接続時は `_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` でプロジェクトルートを解決。
