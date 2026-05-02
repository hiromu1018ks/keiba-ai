# keiba-ai — Win Model Improvement

## What This Is

競馬AI予測システムの単勝モデル改善プロジェクト。既存のLightGBM 2段階モデル(P(hit) × E(odds|hit))をベースに、単勝ベッティングのバックテストROIを100%超えに引き上げる。Python 3.11 + LightGBM + scikit-learn、PostgreSQL(EveryDB2)データソース、Parquetベースのデータパイプライン。

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
- ✓ 既存特徴量の単勝に対する有効性を分析・特定 — Phase 1
- ✓ 単勝特化の特徴量を設計・実装(レース展開、勝ち癖等) — Phase 1
- ✓ モデル構造の最適化(キャリブレーション、アンサンブル、EV補正) — Phase 2
- ✓ 単勝ベッティング戦略の最適化(Kelly基準、レジーム適応) — Phase 3
- ✓ 単勝モデルの品質検証(時系列交差検証、過学習チェック) — Phase 4

### Active

- [ ] 単勝モデルのバックテストROI > 100%を達成 (WF検証スクリプト実行待ち)

### Out of Scope

- 複勝/ワイドモデルの変更 — 単勝に集中するため
- 新データ源の導入 — 既存EveryDB2データで十分と判断
- 実馬券購入機能 — ペーパートレードまで
- Web UI — CLIベースで十分
- リアルタイムオッズ収集の改善 — 既存機能をそのまま使用

## Context

### 現状の課題

- 複勝中心に改善を進めてきたがROI 89%で赤字、100%の壁を超えられない
- 2024年テストデータ: ベット数9,074 / 投資額907,400円 / 払戻額807,400円
- 学習期間: 2020-2023 / テスト期間: 2024

### 技術背景

- Parquetベースのデータパイプライン(PostgreSQLはETL専用)
- LightGBM Ranker(能力推定) + Binary(hit/EV補正)
- RegimeDetector: 3状態(aggressive/conservative/collapsed)
- BenterCombination: scipy最適化による確率統合
- WinStrategy: 既存の単勝ベッティングロジック

### 検討すべき改善方向

1. **特徴量分析**: 既存100+列のうち単勝に効く特徴量を特定。不要特徴量の除去も検討
2. **キャリブレーション**: IsotonicRegressionの精度向上、過剰補正の防止
3. **アンサンブル**: 既存XGBoost/CatBoost(Level-1)の統合強化
4. **Benter最適化**: 確率統合の重み最適化
5. **ベッティング**: Kelly基準による最適賭け金、マーケットレジーム判定の精緻化

## Constraints

- **Tech stack**: Python 3.11, LightGBM, scikit-learn, pandas, pyarrow — 既存技術スタックを維持
- **Data**: EveryDB2 (2015-2025) — 新データ源は追加しない
- **Testing**: 全テスト mock使用(DB不要) — 既存テスト方針を維持
- **Code style**: Ruff (py311, line-length=100), mypy (strict) — 既存規約を維持
- **No external services**: ローカル実行のみ、クラウドサービス不使用

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 単勝に集中 | 複勝で100%超え困難、単勝は中穴でエッジが出やすい | — Pending |
| 既存データ活用 | EveryDB2のデータ量(2015-2025)で十分な学習データがある | — Pending |
| 2段階モデル維持 | P×E分解は理論的に正当、改善は精度面で行う | — Pending |
| 特徴量分析から開始 | モデル改善の前に、どの特徴量が効いているか把握が必要 | — Pending |

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
*Last updated: 2026-05-02 after initialization*
