# keiba-ai — Win Model Improvement

## What This Is

競馬AI予測システム v5.5 — 統計的 horse racing prediction system (単勝/複勝/ワイド)。
LightGBM + XGBoost + CatBoost 3モデルスタッキング、Optuna個別HP最適化、Conformal EV区間による信頼性評価、9新特徴量(EMA時系列・ペースフィグア・オッズ変動・オッズ乖離)を搭載。
MLflow で実験管理。PostgreSQL (EveryDB2/JRA-VAN DataLab) をデータソースとする。

## Core Value

単勝モデルのバックテストROIを100%超えにすること。全ての改善はこの指標で判断する。

## Requirements

### Validated

- ✓ ETL pipeline: PostgreSQL (EveryDB2) → Parquet 抽出 — existing
- ✓ Feature engine: 14モジュール・100+列の特徴量生成 — existing
- ✓ 2-stage model: Win/Place/Wide 各P(hit)×E(odds|hit) — existing
- ✓ Surface submodel: 芝/ダート独立学習 — existing
- ✓ Backtest engine: 歴史シミュレーション + バンクロール追跡 — existing
- ✓ OOF prediction: 学習リーク防止 — existing
- ✓ Betting system: Strategy/Orchestrator/StakeCalculator/DDController — existing
- ✓ MLflow experiment tracking — existing
- ✓ Paper trading: ライブ推論パイプライン — existing
- ✓ 既存特徴量の単勝に対する有効性を分析・特定 — v1.0
- ✓ 単勝特化の特徴量を設計・実装(レース展開、勝ち癖等) — v1.0
- ✓ モデル構造の最適化(キャリブレーション、アンサンブル、EV補正) — v1.0
- ✓ 単勝ベッティング戦略の最適化(Kelly基準、レジーム適応) — v1.0
- ✓ 単勝モデルの品質検証(時系列交差検証、過学習チェック) — v1.0
- ✓ EMA重み付け時系列特徴量(TSER-01~03)とペースフィグア(PACE-01~02) — v1.1
- ✓ オッズ変動特徴量(acceleration + direction_consistency) — v1.1
- ✓ オッズ乖離EV特徴量(deviation_rank/zscore) + Conformal EV区間 — v1.1
- ✓ 3モデルスタッキング多様性強制(Optuna + early stopping + feature subset) — v1.1

### Active

(None — next milestone to be defined)

### Out of Scope

| Feature | Reason |
|---------|--------|
| 複勝/ワイドモデルの変更 | 単勝に集中するため |
| 新データ源の導入 | 既存EveryDB2データで十分 |
| 実馬券購入機能 | ペーパートレードまで |
| Web UI | CLIベースで十分 |
| リアルタイムオッズ収集の改善 | 既存機能をそのまま使用 |
| LSTM/Transformer時系列モデリング | 過去5-15走では過学習リスク高 |
| 複雑メタラーナー(GBM/NN) | 特徴量3個ではRidgeが最適 |
| sklearn StackingClassifier | ネイティブブースティングAPIとPIT安全フォールドに非対応 |

## Current Milestone: TBD (Next milestone to be planned)

**Goal:** TBA

## Context

### 現状の課題

- v1.1で3本柱(特徴量9追加・オッズ乖離EV・3モデルスタッキング)を実装完了
- バックテストROI検証が未実施(PostgreSQL環境必要)
- 学習時間がOptunaチューニングにより推定2-3倍に増加

### 技術背景

- 3モデルGBMスタッキング: LightGBM + XGBoost + CatBoost (Optuna個別HP最適化)
- Conformal EV区間: 80%/90% 2レベルalpha信頼区間 + confidence_score
- Parquetベースのデータパイプライン(PostgreSQLはETL専用)
- RegimeDetector: 3状態(aggressive/conservative/collapsed)
- WinStrategy: Conformal信頼性スコア付きベッティング

### 検討すべき改善方向

1. **バックテストROI検証**: 実際のバックテストでROI>100%を確認 — 最優先
2. **ベッティング戦略精緻化**: Kelly基準・RegimeDetector精緻化 — v1.2候補
3. **Stage1 Rankerスタッキング**: 3Ranker→メタRanker構成 — 複雑度高
4. **Late money特徴量**: オッズスナップショット粒度検証が必要

## Constraints

- **Tech stack**: Python 3.11, LightGBM, XGBoost, CatBoost, Optuna, pandas, pyarrow
- **Data**: EveryDB2 (2015-2025) — 新データ源は追加しない
- **Testing**: 全テスト mock使用(DB不要) — 1,113テスト
- **Code style**: Ruff (py311, line-length=100), mypy (strict)
- **No external services**: ローカル実行のみ、クラウドサービス不使用

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 単勝に集中 | 複勝で100%超え困難、単勝は中穴でエッジが出やすい | — Pending (backtest未実行) |
| 既存データ活用 | EveryDB2のデータ量(2015-2025)で十分な学習データがある | — Pending |
| 2段階モデル維持 | P×E分解は理論的に正当、改善は精度面で行う | — Pending |
| 特徴量分析から開始 | モデル改善の前に、どの特徴量が効いているか把握が必要 | ✓ Good (v1.0) |
| 3モデルスタッキング | 単一モデルより精度向上が期待できる | ✓ Good (v1.1) |
| EMA halflife=3 | 金融時系列解析の標準値。3走前の重みは直近の50% | ✓ Good (v1.1) |
| ペース3サブ特徴量分解 | LightGBMが非線形組み合わせを自動学習 | ✓ Good (v1.1) |
| Optuna探索空間分離 | 各モデルに異なる木複雑度で多様性強制 | ✓ Good (v1.1) |
| confidenceをpair_scoresのみ | 組合せ爆発(4次元目)を回避 | ✓ Good (v1.1) |
| ベッティング戦略は後回し | モデル精度を先に最大化し、戦略は精度が十分になってから調整 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-04 after v1.1 milestone completion*
