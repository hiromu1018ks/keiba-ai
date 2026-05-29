# keiba-ai

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-2000%2B%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-orange.svg)]()

**JRA競馬AI予測システム** -- 過去のレースデータから機械学習で期待値(EV)を計算し、 statistical edge に基づく投資判断を行うオープンソースシステム。

LightGBM + XGBoost + CatBoost の3モデルスタッキング、2段階予測モデル (P(hit) x E(odds|hit))、Fractional Kellyサイジング、市場レジーム検知を搭載。

---

## 目次

- [特徴](#特徴)
- [システムアーキテクチャ](#システムアーキテクチャ)
- [必要条件](#必要条件)
- [クイックスタート](#クイックスタート)
- [使い方](#使い方)
  - [ETL (データ抽出)](#1-etl-データ抽出)
  - [モデル学習](#2-モデル学習)
  - [バックテスト](#3-バックテスト)
  - [ウォークフォワード検証](#4-ウォークフォワード検証)
  - [ハイパーパラメータチューニング](#5-ハイパーパラメータチューニング)
  - [戦略パラメータ最適化](#6-戦略パラメータ最適化)
  - [ペーパートレード (リアルタイム予測)](#7-ペーパートレード-リアルタイム予測)
  - [診断・分析ツール](#8-診断分析ツール)
- [プロジェクト構成](#プロジェクト構成)
- [設定](#設定)
- [開発ガイド](#開発ガイド)
- [技術スタック](#技術スタック)
- [免責事項](#免責事項)
- [ライセンス](#ライセンス)

---

## 特徴

- **2段階AI予測** -- 「的中確率 P(hit)」と「的中時払戻額 E(odds|hit)」を別モデルで学習。ゼロインフレーション問題を回避し、EV = P x E で期待値を算出
- **3モデルスタッキング** -- LightGBM + XGBoost + CatBoost -> Ridge メタラーナー。Optuna で多様性を強制し、相関ペナルティで過学習を防止
- **市場ブレンドキャリブレーション** -- MarketAwareWinCalibrator: モデル確率とオッズ暗示確率をBenter (1994) logit-blend。51次元セグメント条件付けで人気帯別バイアスを補正
- **レースレベルランキング** -- RaceLevelRanker: Ridge relevance/value scoring で馬の投資スコア (investment_score) を算出し、shadow modeで安全導入
- **100+特徴量** -- 過去成績、血統、騎手・調教師コンテキスト、オッズ動態、市場バイアス、情報非対称性、相対比較、交互作用項等14モジュールから生成
- **Conformal EV区間** -- CQR (Conformalized Quantile Regression) で80%/90%信頼区間を算出し、不確実性を定量化
- **自動資金管理** -- 3段階ドローダウン制御 (NORMAL/REDUCED/STOP) + ヒステリシス + Fractional Kelly サイジング
- **市場レジーム検知** -- 3状態 (AGGRESSIVE/CONSERVATIVE/COLLAPSED) LightGBM多クラス分類でオッズ分布から市場の荒れ具合を検知
- **厳密な検証基盤** -- Walk-forward CV、Shadow Comparison Framework、DeploymentGateEvaluator、OOFHealthValidator、Feature Routing Audit で安全性を保証
- **ペーパートレード** -- レース当日にリアルタイム推論、Slack通知、結果照合、HTMLレポート生成

## システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources                                 │
│                                                                  │
│  EveryDB2 (PostgreSQL)          JV-Link SDK (Windows)            │
│  JRA-VAN DataLab                オッズ時系列 (s_jodds_*)          │
└───────────┬─────────────────────────────────┬───────────────────┘
            │ ETL (run_etl.py)                │
            ▼                                 │
┌───────────────────────┐                     │
│  Parquet Files         │◄────────────────────┘
│  (data/raw, data/odds) │
│  races, entries,       │
│  payouts, odds_ts,     │
│  horse_career, sire    │
└───────────┬───────────┘
            │ DataRepository / ParquetStore
            ▼
┌───────────────────────────────────────────────────────────────────┐
│                      Feature Engineering                          │
│                                                                    │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Horse    │ │ Bloodline │ │ Odds     │ │ Market   │           │
│  │ History  │ │ /Sire/Dam │ │ Dynamics │ │ Bias     │           │
│  └────┬─────┘ └─────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       │             │            │             │                   │
│  ┌────┴─────┐ ┌─────┴─────┐ ┌───┴──────┐ ┌───┴──────┐          │
│  │ Jockey/  │ │ Form      │ │ Race     │ │ Relative │          │
│  │ Trainer  │ │ Cycle     │ │ Level    │ │ /Intra   │          │
│  └────┬─────┘ └─────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       │             │            │             │                   │
│  ┌────┴─────────────┴────────────┴─────────────┴────┐            │
│  │          Interaction / Cross Features             │            │
│  │    Target Encoding (3-fold expanding window)      │            │
│  └───────────────────────┬───────────────────────────┘            │
│                          │                                        │
│              FeatureEngine.build_all()                            │
│              ~100+ features, cache with code-hash                 │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│                      Model Pipeline (TrainingPipelineV5)          │
│                                                                    │
│  ┌──────────────────────────────────────────────┐                 │
│  │ Surface Split: Turf / Dirt (SubModelManager) │                 │
│  └──────────────────┬───────────────────────────┘                 │
│                     │                                              │
│  Stage 1: AbilityModel (LambdaRank) ── softmax ──> p_ability     │
│                     │                                              │
│  Market: MarketModel ──> p_market, log_error_delta               │
│                     │                                              │
│  Stage 2: WinTwoStageModel ──> P(win_hit) x E(win_return)        │
│          PlaceTwoStageModel ──> P(place_hit) x E(place_return)    │
│                     │                                              │
│  EV Correction: EVCorrectionModel (P/E decomposition)            │
│          ConformalEVModel (CQR 80%/90% intervals)                 │
│                     │                                              │
│  Calibrator: MarketAwareWinCalibrator (logit-blend + segments)   │
│  Ranker:    RaceLevelRanker (Ridge relevance + value scoring)     │
│                     │                                              │
│  Safety: RaceQualityScreener, RegimeDetector (3-state)            │
│  Gates:  WinSelectionGate, PlaceSelectionGate, WinProfitSelector  │
│  Policy: WinSelectionPolicy (surface-aware final ranking)         │
│                     │                                              │
│  Stacked Ensemble: LGBM + XGB + CB -> Ridge (optional --ensemble)│
│  All models save to MLflow + local directory                      │
└─────────────────────┬─────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────────────────┐
│                    Betting & Execution                             │
│                                                                    │
│  RacePredictor (shared by BacktestEngine + PaperPredictor)        │
│      │                                                             │
│      ├── RegimeDetector.detect() ──> strategy_params              │
│      ├── EV prediction + correction + conformal intervals         │
│      ├── SelectionGate -> ProfitSelector -> SelectionPolicy       │
│      ├── EVTtailCalibrator (feature family consensus)             │
│      └── LateMoneyFilter (t-3min judgment)                        │
│      │                                                             │
│  BettingOrchestrator                                               │
│      ├── WinStrategy / PlaceStrategy / WideStrategy               │
│      ├── StakeCalculator (Fractional Kelly)                       │
│      ├── DrawdownController (3-tier NORMAL/REDUCED/STOP)          │
│      ├── GateKeeper (edge threshold filter)                       │
│      └── MetaSwitcher (regime-linked params)                      │
│                                                                    │
│  ───> Bet objects with stake, EV, confidence                      │
└───────────────────────────────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
  ┌──────────────┐       ┌──────────────┐
  │ BacktestEngine│      │ Paper Trading │
  │ (simulation)  │      │ (live)        │
  └──────────────┘       └──────────────┘
```

---

## 必要条件

| 要件 | バージョン |
|------|-----------|
| Python | 3.11 (mise推奨) |
| PostgreSQL | EveryDB2 (JRA-VAN DataLab) |
| OS | Windows / Linux / macOS |

**Python依存関係** (pyproject.tomlで管理):

```
pandas>=2.2, numpy>=1.26, scikit-learn>=1.4, lightgbm>=4.3,
xgboost>=2.0, catboost>=1.2, optuna>=3.5, psycopg2-binary>=2.9,
sqlalchemy>=2.0, pyarrow>=14.0, pyyaml>=6.0, mlflow>=2.12,
tqdm>=4.66, jinja2>=3.1
```

**開発用追加依存**: `ruff>=0.4`, `mypy>=1.10`, `ipykernel>=6.29`

---

## クイックスタート

```bash
# 1. リポジトリをクローン
git clone <repository-url>
cd keiba-ai

# 2. Python 3.11 をインストール・アクティベート (mise使用)
mise install
mise activate

# 3. 依存パッケージをインストール
pip install -e ".[dev]"

# 4. テストを実行して動作確認 (DB不要、全テストmock使用)
python -m pytest tests/ -v
```

詳細なセットアップ手順は [Getting Started](docs/guide/04_getting_started.md) を参照してください。

---

## 使い方

PostgreSQL (EveryDB2) が `localhost:5432` で稼働している前提です。

### 1. ETL (データ抽出)

EveryDB2 (PostgreSQL) からParquetファイルへのETL。103テーブル対応。

```bash
# 環境変数の設定
export PGPASSWORD=<your_password>

# 全量抽出 (初回のみ、約10分)
python scripts/run_etl.py --mode full --start 20140101 --end 20251231

# 差分マージ (2回目以降、高速)
python scripts/run_etl.py --mode delta
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--mode` | `full`\|`delta` | **必須** | `full`=全量抽出、`delta`=差分マージ |
| `--start` | YYYYMMDD | -- | 開始日 (`full`で必須) |
| `--end` | YYYYMMDD | -- | 終了日 (`full`で必須) |
| `--tables` | list[str] | -- | 対象テーブル (省略時は全テーブル) |

### 2. モデル学習

特徴量生成 + 全モデル学習。4年学習ウィンドウ推奨。

```bash
# 基本学習 (約17分)
python scripts/run_train.py --start 20220101 --end 20251231

# アンサンブル有効 (LGBM+XGB+CB->Ridge)
python scripts/run_train.py --start 20220101 --end 20251231 --ensemble

# MLflow実験名指定
python scripts/run_train.py --start 20220101 --end 20251231 --experiment keiba-v5
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--start` | YYYYMMDD | **必須** | 学習開始日 |
| `--end` | YYYYMMDD | **必須** | 学習終了日 |
| `--ensemble` | flag | False | StackedEnsemble有効化 |
| `--experiment` | str | `keiba-v5` | MLflow実験名 |

### 3. バックテスト

学習→テスト期間の投資シミュレーション。毎回学習し直す設計で再現性を保証。

```bash
# 単一年度
python scripts/run_backtest.py \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231 --ensemble

# マルチ年度 (推奨)
python scripts/run_backtest.py --years 2023 2024 2025 --train-window 4 --ensemble

# Optuna最適化済みパラメータで検証
python scripts/run_backtest.py \
  --ensemble --strategy-manifest data/strategy_manifest.json \
  --train-start 20200101 --train-end 20231231 \
  --test-start 20240101 --test-end 20241231

# Kelly基準 + HTMLレポート
python scripts/run_backtest.py \
  --years 2024 2025 --train-window 4 --ensemble \
  --betting-mode kelly --report
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--train-start` | YYYYMMDD | -- | 学習開始日 (単一年度) |
| `--train-end` | YYYYMMDD | -- | 学習終了日 (単一年度) |
| `--test-start` | YYYYMMDD | -- | テスト開始日 (単一年度) |
| `--test-end` | YYYYMMDD | -- | テスト終了日 (単一年度) |
| `--years` | list[int] | -- | テスト年度リスト (マルチ年度) |
| `--train-window` | int | 4 | 学習年数 (マルチ年度) |
| `--ensemble` | flag | False | アンサンブル有効化 |
| `--betting-mode` | `flat`\|`kelly` | `flat` | 100円固定 or Fractional Kelly |
| `--betting-target` | `win`\|`place`\|`wide` | `win` | 投票対象 |
| `--report` | flag | False | HTMLレポート出力 |
| `--skip-train` | flag | False | 学習スキップ (キャッシュ使用) |
| `--profile` | flag | False | pyinstrumentプロファイリング |
| `--strategy-manifest` | str | -- | Optuna最適化済みmanifest JSON |
| `--calibration-bt` | flag | False | OddsBandFilterキャリブレーションBT |

所要時間: 約41分/年 (manifestなし), 約57分/年 (manifestあり)

### 4. ウォークフォワード検証

2-fold WF検証で過学習を検出。Feature importance安定性 (Spearman rho) とROI gapを測定。

```bash
python scripts/run_wf_validation.py --ensemble
```

| 引数 | 型 | デフォルト | 説明 |
|------|-----|-----------|------|
| `--ensemble` | flag | False | アンサンブル有効化 |
| `--betting-target` | `win`\|`place`\|`wide` | `win` | 投票対象 |
| `--profile` | flag | False | プロファイリング |

Fold定義: Fold0 = train 2020-2023 / test 2024, Fold1 = train 2021-2024 / test 2025

### 5. ハイパーパラメータチューニング

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

所要時間: 約30分 (50 trials)

### 6. 戦略パラメータ最適化

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
| `--seed` | int | 42 | TPESampler乱数シード |
| `--models-dir` | str | `data/models` | 学習済みモデルディレクトリ |
| `--output` | str | `data/strategy_manifest.json` | 出力manifestパス |
| `--seeds` | str | -- | multi-seed安定性検証 (例: `42,43,44`) |

16次元: Kelly fraction, EV/edge閾値, DD閾値, rolling window等。4fold WF + MedianPruner。

### 7. ペーパートレード (リアルタイム予測)

学習済みモデルで実際のレース当日にリアルタイム推論。

```bash
export PGPASSWORD=<your_password>
export SLACK_WEBHOOK_URL=<your_slack_webhook_url>

# Setup -- 当日のレース一覧を確認
python scripts/run_paper_trading.py --mode setup --date 2026-04-12

# Predict -- 発走5分前に実行 (約25秒)
python scripts/run_paper_trading.py --mode predict --date 2026-04-12 --ensemble

# Reconcile -- レース結果を取得して勝敗を確定
python scripts/run_paper_trading.py --mode reconcile --date 2026-04-12

# Dry-run -- 過去データでシミュレーション
python scripts/run_paper_trading.py --mode dry-run --date 2024-07-13
```

| モード | 内容 | タイミング |
|--------|------|-----------|
| `setup` | 当日のレース一覧・出走馬を取得 | レース前 |
| `predict` | 特徴量生成 -> AI推論 -> ベット判定 -> Slack通知 | 発走5分前 |
| `reconcile` | レース結果取得 -> 勝敗計算 -> HTMLレポート生成 | レース終了後 |
| `dry-run` | 過去データで一括シミュレーション | いつでも |

### 8. 診断・分析ツール

本番運用には不要だが、モデル開発・分析に有用なツール群。

#### Shadow Comparison (ベースライン vs シャドウモデル比較)

```bash
python scripts/run_shadow_comparison.py \
  --baseline-root data/models-backtest \
  --shadow-root data/models-backtest \
  --folds 2024 2025 --report
```

#### Shadow Diagnosis (3段階プログレッシブ診断)

```bash
python scripts/run_shadow_diagnosis.py \
  --input-dir data/backtest/shadow --report
```

#### Feature Routing Audit (特徴量リーク検査)

```bash
python scripts/run_feature_routing_audit.py --output-dir data/audit
```

#### 特徴量重要度分析

```bash
python scripts/analyze_feature_importance.py --all-models --tier-report
```

#### IC評価 (Information Coefficient)

```bash
python scripts/run_ic_eval.py data/oof/oof_predictions.parquet --mlflow
```

#### Gain-per-Depth診断

```bash
python scripts/run_gpd.py --ensemble
```

#### その他の分析ツール

| スクリプト | 内容 |
|-----------|------|
| `scripts/analyze_odds_movement.py` | オッズ変動分析 (Steamer/Stable/Drifter) |
| `scripts/analyze_high_odds.py` | 高オッズ的中パターン分析 (Cohen's d + TreeSHAP) |
| `scripts/analyze_loss_segments.py` | バックテスト損失12次元セグメント分析 |
| `scripts/analysis_distribution_shift.py` | 学習/BT特徴量分布シフト検出 |
| `scripts/compare_bt_pt_features.py` | バックテスト vs ペーパートレード特徴量比較 |
| `scripts/prune_noise_features.py` | Tier 1ノイズ特徴量プルーニング |
| `scripts/freeze_feature_manifest.py` | 全モデルFEATURE_COLS SHA256凍結 |
| `scripts/precompute_career_stats.py` | Point-in-Time馬キャリア統計事前計算 |
| `scripts/precompute_sire_stats.py` | Point-in-Time種牡馬統計事前計算 |
| `scripts/diagnose_phase36_diff.py` | ベースライン vs カレントBT差分診断 |
| `scripts/scrape_everydb2_manual.py` | EveryDB2データフォーマット定義スクレイピング |

---

## プロジェクト構成

```
keiba-ai/
├── src/                          # メインソースコード
│   ├── domain/                   # ドメイン型・データクラス
│   │   ├── types.py              # Enum: Surface, BetType, RegimeState, POST_RACE_COLS
│   │   └── models.py             # Dataclass: Race, Entry, Bet, TrainedModelsV5 等
│   │
│   ├── db/                       # データアクセス層
│   │   ├── parquet_store.py      # Parquet読み書き (pyarrow述語プッシュダウン)
│   │   ├── repository.py         # MLパイプラインデータアクセス窓口
│   │   ├── readers.py            # 各種データロードヘルパー
│   │   ├── connection.py         # PostgreSQL接続 (ETL専用)
│   │   ├── etl.py                # ETLエンジン (EveryDB2 -> Parquet)
│   │   ├── everydb2_queries.py   # EveryDB2直接クエリ
│   │   ├── odds_extractor.py     # 発走前オッズスナップショット抽出
│   │   ├── model_loader.py       # MLflow/ローカル -> TrainedModelsV5 復元
│   │   └── schema.py             # PostgreSQL DDL定義
│   │
│   ├── features/                 # 特徴量生成 (14モジュール, 100+列)
│   │   ├── feature_engine.py     # 特徴量オーケストレータ (キャッシュ付き)
│   │   ├── horse_history_features.py   # 過去成績
│   │   ├── bloodline_features.py       # 血統 (種牡馬/BMS)
│   │   ├── sire_features.py            # 種牡馬産駒統計
│   │   ├── dam_pedigree_features.py     # 母系統
│   │   ├── odds_dynamics_features.py   # オッズ変動 (drop_rate, velocity)
│   │   ├── market_bias_features.py     # 市場バイアス (entropy, overround)
│   │   ├── intra_race_features.py      # レース内相対特徴量
│   │   ├── info_asymmetry_features.py  # 情報非対称性
│   │   ├── race_difficulty_model.py    # レース難易度スコア
│   │   ├── race_level_features.py      # レースレベル集約特徴量
│   │   ├── market_cross_features.py    # 市場クロス整合性 (Harville)
│   │   ├── form_cycle_features.py      # フォームサイクル
│   │   ├── pace_aptitude_features.py   # ペース適性
│   │   ├── course_features.py          # コース適性
│   │   ├── jockey_context_features.py  # 騎手コンテキスト
│   │   ├── trainer_context_features.py # 調教師コンテキスト
│   │   ├── jockey_trainer_combo.py     # 騎手-調教師コンビ
│   │   ├── odds_deviation_features.py  # オッズ乖離
│   │   ├── high_odds_features.py       # 高オッズ特化
│   │   ├── interaction_features.py     # 交互作用項
│   │   ├── relative_features.py        # 相対比較特徴量
│   │   ├── record_features.py          # コースレコード
│   │   ├── mining_features.py          # n_mining予測
│   │   ├── target_encoding.py          # ターゲットエンコーディング
│   │   ├── horse_career_stats.py       # キャリア統計計算
│   │   ├── leakage_validators.py       # リーク検証
│   │   └── win_feature_analysis.py     # 特徴量分析ユーティリティ
│   │
│   ├── models/                   # 予測モデル群 (28ファイル)
│   │   ├── stage1_ability_model.py      # Stage1 能力モデル (LambdaRank)
│   │   ├── market_model.py             # 市場確率予測
│   │   ├── two_stage_return_model.py   # Stage2 Win/Place (PxE)
│   │   ├── ev_correction_model.py      # EV補正 (P/E decomposition)
│   │   ├── conformal_ev_model.py       # CQR Conformal EV区間
│   │   ├── regime_detector.py          # 市場レジーム検知 (3状態)
│   │   ├── race_quality_screener.py    # レース品質スクリーニング
│   │   ├── place_ability_model.py      # 複勝的中確率
│   │   ├── stacked_ensemble.py         # 3モデルスタッキング
│   │   ├── benter_combination.py       # Benter (1994) logitブレンド
│   │   ├── market_aware_win_calibrator.py  # 市場ブレンドキャリブレータ
│   │   ├── race_level_ranker.py        # レースレベルランキング
│   │   ├── win_selection_gate.py       # 単勝選択ゲート
│   │   ├── win_profit_selector.py      # 利益指向候補セレクタ
│   │   ├── win_selection_policy.py     # 最終単勝選択ポリシー
│   │   ├── place_selection_gate.py     # 複勝選択ゲート
│   │   ├── wide_two_stage_model.py     # ワイド2段階モデル
│   │   ├── wide_pair_builder.py        # ワイドペアビルダー
│   │   ├── walk_forward_cv.py          # Walk-forward CV
│   │   ├── submodel_manager.py         # 芝/ダート分割管理
│   │   ├── reproducibility.py          # 再現性設定 (seed固定)
│   │   ├── ic_evaluator.py             # IC評価フレームワーク
│   │   ├── drift_diagnostics.py        # 分布ドリフト診断
│   │   ├── ev_diagnostics.py           # EV推定精度診断
│   │   └── gpd_diagnostics.py          # Gain-per-Depth診断
│   │
│   ├── betting/                  # 投票戦略・資金管理 (13ファイル)
│   │   ├── orchestrator.py       # 投票オーケストレータ
│   │   ├── stake_calculator.py   # Fractional Kelly
│   │   ├── drawdown_controller.py # 3段階DD制御
│   │   ├── win_strategy.py       # 単勝戦略
│   │   ├── place_strategy.py     # 複勝戦略
│   │   ├── wide_strategy.py      # ワイド戦略
│   │   ├── gate_keeper.py        # エッジ閾値フィルタ
│   │   ├── meta_switcher.py      # レジーム連動パラメータ
│   │   ├── late_money_filter.py  # t-3min判定
│   │   ├── odds_band_filter.py   # オッズ帯別ROIフィルタ
│   │   ├── ev_tail_calibration.py # EVテールキャリブレーション
│   │   └── default_strategy.py   # デフォルト戦略ビルダー
│   │
│   ├── backtest/                 # バックテスト・検証 (12ファイル)
│   │   ├── engine.py             # BacktestEngine (投資シミュレーション)
│   │   ├── race_predictor.py     # 共通推論パイプライン
│   │   ├── validation_suite.py   # バックテスト検証スイート
│   │   ├── parameter_freeze_protocol.py # パラメータ凍結プロトコル
│   │   ├── diagnostic_logger.py  # レース診断ログ
│   │   ├── report.py             # HTMLレポート生成
│   │   ├── validation_report.py  # 検証結果JSON
│   │   ├── shadow_comparison.py  # Shadow Comparison Framework
│   │   ├── shadow_diagnosis.py   # 3段階プログレッシブ診断
│   │   ├── shadow_report.py      # Shadow HTMLレポート
│   │   └── deployment_gates.py   # デプロイメントゲート評価
│   │
│   ├── pipelines/                # パイプライン
│   │   └── training_pipeline.py  # 学習パイプライン v5.4
│   │
│   ├── tuning/                   # ハイパーパラメータ最適化
│   │   ├── optuna_tuner.py       # モデルHPチューナー
│   │   └── strategy_optimizer.py # 16次元戦略パラメータ最適化
│   │
│   ├── investment/               # 投資特徴量フレーム
│   │   ├── schema_registry.py   # 94仕様/9カテゴリスキーマ定義
│   │   ├── feature_frame.py     # dual-mode特徴量ビルダー
│   │   ├── leakage.py           # リーク検証
│   │   ├── cache.py             # Parquet + sidecar manifest
│   │   └── manifest.py          # SHA256スキーマハッシュ
│   │
│   ├── validation/               # 検証基盤
│   │   ├── oof_health_validator.py  # OOF健全性 fail-fast検証
│   │   └── artifact_profiles.py     # アーティファクトプロファイル
│   │
│   ├── audit/                    # 監査基盤
│   │   └── feature_routing_registry.py  # 特徴量ルーティング監査レジストリ
│   │
│   ├── ingestion/                # データ取得
│   │   ├── jvlink_fetcher.py     # JV-Linkデータ取得
│   │   └── odds_collector.py     # オッズ時系列収集
│   │
│   ├── paper_trading/            # ペーパートレード (6ファイル)
│   │   ├── config.py             # 設定管理
│   │   ├── predictor.py          # 予測ロジック
│   │   ├── watcher.py            # レース監視
│   │   ├── reconciler.py         # 結果照合
│   │   └── report.py             # HTMLレポート
│   │
│   ├── automation/               # 自動化
│   │   ├── scheduler.py          # レース日スケジューラ
│   │   ├── safety_guard.py       # バンクロール安全ガード
│   │   └── pat_voter.py          # JRA-IPAT自動投票インターフェース
│   │
│   ├── monitoring/               # 監視
│   │   ├── model_monitor.py      # モデル性能モニタリング
│   │   ├── notifier.py           # Slack通知
│   │   └── auto_retrain_trigger.py # 自動再学習トリガー
│   │
│   └── utils/                    # ユーティリティ
│       ├── timing.py             # 実行時間計測
│       ├── profiling.py          # pyinstrumentプロファイリング
│       └── wf_splits.py          # Walk-forward分割ユーティリティ
│
├── scripts/                      # CLIスクリプト (24ファイル)
├── tests/                        # テスト (143ファイル, 2000+テスト)
├── config/                       # 設定ファイル
│   ├── settings.yaml             # メイン設定
│   ├── backtest_config.yaml      # バックテスト設定
│   └── etl_tables.yaml           # ETLテーブルマッピング (103テーブル)
├── data/                         # データディレクトリ (gitignore)
│   ├── raw/                      # 生データ Parquet
│   ├── odds/                     # オッズ Parquet
│   ├── features/                 # 特徴量キャッシュ
│   ├── models/                   # 学習済みモデル
│   ├── backtest/                 # バックテスト結果
│   └── paper_trading/            # ペーパートレード結果
├── docs/                         # ドキュメント
│   ├── guide/                    # 入門ガイド
│   ├── concepts/                 # 概念説明
│   └── reference/                # リファレンス
└── pyproject.toml                # プロジェクト設定
```

---

## 設定

### config/settings.yaml

```yaml
database:
  host: "localhost"
  port: 5432
  dbname: "everydb2"
  user: "postgres"
  password: ""               # 環境変数 PGPASSWORD で上書き

feature_engine:
  exclude_steeple: true      # 障害レース (TrackCD 51-59) を除外

late_money:
  cancel_threshold: 0.25     # オッズ25%以上急落 -> キャンセル
  add_rise_threshold: 0.30   # オッズ30%以上急騰 -> 追加候補
  cancel_time_minutes: 3     # t-3min で判定

submodel:
  surfaces: ["turf", "dirt"]
  distance_bands:            # 距離帯定義 (芝/ダート別)
    turf:
      sprint: [0, 1400]
      mile: [1401, 1700]
      intermediate: [1701, 2100]
      long: [2101, 9999]

betting_strategy:
  default_fractional_kelly: 0.5
  kelly_fraction_cap: 0.25
  target_ev: 1.10
  max_scale: 2.0
  regime_fractions:
    aggressive: 0.50
    conservative: 0.25
    collapsed: 0.00
```

### 環境変数

| 変数 | 説明 |
|------|------|
| `PGPASSWORD` | PostgreSQLパスワード (settings.yamlのpasswordより優先) |
| `SLACK_WEBHOOK_URL` | ペーパートレードSlack通知用Webhook URL |

---

## 開発ガイド

### テスト

全テストは`unittest.mock`を使用し、データベース不要で実行可能。

```bash
# 全テスト実行
python -m pytest tests/ -v

# 単一テストファイル
python -m pytest tests/test_domain.py -v

# カバレッジ付き
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### リント・フォーマット

```bash
# リント (Ruff)
ruff check src/ tests/

# フォーマットチェック
ruff format --check src/ tests/

# フォーマット適用
ruff format src/ tests/
```

### 型チェック

```bash
# Mypy (strict mode: 全関数に型アノテーション必須)
mypy src/
```

### コーディング規約

- **Ruff**: target py311, line-length=100, rules=E/F/I/N/W
- **Mypy**: `disallow_untyped_defs = true` (全関数に型アノテーション必須)
- **コミットメッセージ**: Conventional Commits (日本語)
- **インポートパス**: `pythonpath = [".", "src"]` 設定済み

```python
# 主要インポートパス
from db.repository import DataRepository
from db.parquet_store import ParquetStore
from db.connection import DatabaseConnection           # ETL専用
from domain.types import Surface, BetType, RegimeState
from domain.models import TrainedModelsV5, Race, Entry
from models.market_aware_win_calibrator import MarketAwareWinCalibrator
from backtest.race_predictor import RacePredictor
from validation.oof_health_validator import OOFHealthValidator
from audit.feature_routing_registry import run_feature_audit
```

---

## マイルストーン履歴

| Milestone | Phases | ROI | Key Deliverable | Status |
|-----------|--------|-----|----------------|--------|
| v1.0 | 1-4 | -- | Win Model: SHAP分析 + Benter補正 + Selection Gate + WF検証 | ✅ Shipped |
| v1.1 | 5-7 | -- | ROI Advanced: EMA特徴量 + オッズ偏差 + 3-model stacking | ✅ Shipped |
| v1.2 | 8-10 | -- | Win Backtest: 精算修正 + ベット履歴 + パイプライン最適化 | ✅ Shipped |
| v1.3 | 11-13 | 91.6% | Betting Strategy: EV_lower + OddsBandFilter + DD制御 + Optuna 16-dim | ✅ Shipped |
| v1.4 | 14-18 | 83.1% | Ensemble Filter: WinSelectionGate再学習 + 動的EV_lower + ドリフト診断 | ✅ Shipped |
| v1.5 | 19-22 | 84.4% | Model Accuracy: Isotonic EV補正 + 高オッズ18特徴量 + CQR区間 | ✅ Shipped |
| v1.6 | 23-28 | 85.7% | Feature Overhaul: POST_RACE漏洩排除 + 22新特徴量 + Target Encoding | ✅ Shipped |
| v1.7 | 29-34 | **97.8%** | Market-Independent: レースレベル6特徴量 + Harville cross + IC/GPD | ✅ Shipped |
| v1.8 | 35-36.1.1 | -- | Turf Precision: haron/lap特徴量 + MarketModel修正 | ✅ Shipped |
| v2.0 | 37-38 | 87.8% | Investment Pipeline: OOFHealthValidator + InvestmentFeatureFrame | ✅ Shipped |
| v2.1 | 39-42 | TBD | MAWC + Ranker (shadow mode) + Shadow Comparison + Deployment Gates | ✅ Shipped |
| v2.2 | 43-46 | **進行中** | ROI Recovery Analysis: 診断 → ビセクション → 構造的修正 → 品質ゲート | 🔄 In Progress |

> **ROI推移:** v1.7で97.8%（最高）→ v2.0で87.8%に回帰（Phase 36の強特徴量がMarketModelに副作用）→ v2.2で回復を目指す

---

## 週末予想ワークフロー (推奨運用)

```
月例再学習判定
  │
  +-- YES --> run_train.py (4年ウィンドウ、約17分)
  │
  +-- NO ----+
             │
      delta ETL (直近データをParquetに反映)
             │
      setup --> レース一覧確認
             │
      predict --> 発走5分前にベット生成 (約25秒/レース)
             │
      レース終了後 --> reconcile --> 結果記録・HTMLレポート更新
```

**学習期間設計**: 予想対象年の前年から4年遡る。

| 予想年 | 学習期間 |
|--------|---------|
| 2026 | 2022-01-01 ~ 2025-12-31 |
| 2027 | 2023-01-01 ~ 2026-12-31 |

月1回の再学習で十分。LightGBMは4年分のデータでロバストな性能を発揮する。

---

## ドキュメントマップ

### 入門編

- [競馬の基礎知識](docs/guide/01_keiba_basics.md) -- 競馬のルールとデータの見方
- [AI予測の基礎](docs/guide/02_ai_prediction_basics.md) -- AIはどうやって予測しているのか
- [システム全体像](docs/guide/03_system_overview.md) -- このシステムがやっていることの全体像
- [はじめ方](docs/guide/04_getting_started.md) -- 環境構築から最初の予測まで
- [ワークフロー](docs/guide/05_workflow.md) -- 週末予想の具体的な手順

### 中級編

- [データパイプライン](docs/concepts/01_data_pipeline.md) -- データ収集から特徴量生成まで
- [予測モデル](docs/concepts/02_prediction_models.md) -- 2段階モデルの仕組み
- [高度なモデル手法](docs/concepts/03_advanced_models.md) -- EV補正・レジーム検知・市場モデル
- [投票戦略](docs/concepts/04_betting_strategy.md) -- 資金管理とDDコントローラー
- [バックテストと検証](docs/concepts/05_backtest_validation.md) -- ウォークフォワード検証の設計

### 上級編

- [アーキテクチャ](docs/reference/01_architecture.md) -- 全体設計と設計判断の理由
- [コード構成](docs/reference/02_code_structure.md) -- ディレクトリ構造と主要モジュール
- [設定ファイル](docs/reference/03_configuration.md) -- settings.yamlの全項目解説
- [コントリビューション](docs/reference/04_contributing.md) -- 開発参加の手引き

---

## 技術スタック

| カテゴリ | 技術 |
|----------|------|
| 言語 | Python 3.11 |
| 機械学習 | LightGBM 4.3+, XGBoost 2.0+, CatBoost 1.2+ |
| メタラーナー | Ridge regression (sklearn) |
| キャリブレーション | IsotonicRegression, LogisticRegression (sklearn) |
| 確率区間 | CQR Conformal Quantile Regression (LightGBM quantile) |
| 最適化 | Optuna 3.5+ (TPE sampler, MedianPruner) |
| データ処理 | pandas 2.2+, numpy 1.26+, pyarrow 14.0+ |
| データベース | PostgreSQL (EveryDB2 / JRA-VAN DataLab), SQLAlchemy |
| 実験管理 | MLflow 2.12+ |
| レポート | Jinja2 (HTML), matplotlib (charts) |
| 品質ツール | Ruff (lint/format), Mypy (strict型チェック), pytest |
| 統計検定 | scipy (KS test, Spearman, Wasserstein), sklearn metrics |

---

## 免責事項

本システムは**学習・研究目的**で公開されているオープンソースソフトウェアです。バックテストの良好な結果は将来の成績を保証するものではありません。競馬は不確実性の高いギャンブルであり、本システムを使用して生じた損失について、開発者は一切の責任を負いません。実際の投票に使用するかどうかは自己責任でお願いします。ギャンブルには依存リスクがあり、法的に制限されている地域もあります。健全な範囲で楽しみましょう。

## ライセンス

MIT License
