# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Win Model

**Shipped:** 2026-05-03
**Phases:** 4 | **Plans:** 7 | **Sessions:** ~3

### What Was Built
- SHAP/gain特徴量重要度分析モジュール + 単勝特化6新特徴量(odds-to-ability比、クラス落リバウンド等)
- WinBenterGate(基本確率×市場確率ブレンド) + Beta/Isotonicキャリブレーション比較
- WinSelectionGate + Conformal信頼性推定 + JRA控除率25%考慮edge_threshold調整
- WFValidationResult + 過学習検出(ROI gap/一貫性/Spearman安定性) + run_wf_validation.py CLI

### What Worked
- TDD RED/GREEN gateが品質を担保(各タスクがtest→featの2コミット)
- 自動deviation修正ルール(Rule 1-3)が実装中のバグを即座検出
- feature-first build orderが安全な基盤を提供

### What Was Inefficient
- テストモックパスの不一致が何度か発生(ローカルimport対策)
- v1.0終了時にHuman UATを実施できず、v1.1に繰越

### Patterns Established
- EMA手動重み付け: weights = (1-decay)**np.arange(n), reversed, normalized
- テストモックはメソッド内ローカルimportに直接パッチ(models.module.Class)

### Key Lessons
1. テストモックパスは実際のimport方法(ローカルvsトップレベル)に合わせて設計する
2. LightGBM feature_namesはDatasetコンストラクタで設定(lgb.trainには渡せない)
3. Conformal予測はalpha方向に注意(高alpha = 狭い区間)

### Cost Observations
- Model mix: ~60% opus, ~30% sonnet, ~10% haiku
- Sessions: ~3
- Notable: Plan 07-01 (ensemble)が26分と最長、Optunaモックテストが時間消費

---

## Milestone: v1.1 — ROI Advanced Model

**Shipped:** 2026-05-03
**Phases:** 3 | **Plans:** 5 | **Sessions:** ~2

### What Was Built
- 9新特徴量: EMA重み付けハロンタイム・クラス調整フォーメトリック・z-score改善トラジェクトリ・ペースフィグア3サブ・オッズacceleration/consistency
- Odds Deviation EV: deviation_rank/zscore + Conformal EV区間(80%/90%) + conformal_confidence_score
- 3モデルスタッキング多様性強制: Optuna探索空間分離 + early stopping + feature subset最適化
- 多様性検証: OOF予測相関 + importance Spearman順位相関の二重検証

### What Worked
- Optuna探索空間分離が各モデルに明確に異なる特性を付与(LGB浅い木/XGB中深さ/CAT深い木)
- predict_lower_boundをthin wrapperにリファクタリングで二重保守を防止(Pitfall 2)
- deviation featuresのrace-relative設計(race内rank+zscore)がノイズに強い
- Phase 5→6→7の直列依存がデータフローを明確にした

### What Was Inefficient
- Optunaモックテストの実行時間が長い(n_trials=3でも2-3分/テストスイート)
- Phase 6でコードレビュー後の修正に5コミット(WR-01~05)が必要だった
- テストDataFrameにrace_idがないケースを見落とし、後でrace_idガードを追加

### Patterns Established
- Post-model feature computation: compute_odds_deviation_features()はAbilityModel後に呼び出す
- 3点2次微分: vel_late(t-30→t-10) - vel_early(t-60→t-30) でsteam move検出
- Optuna _suggest_*_paramsパターン: 各モデルに独立した探索空間定義

### Key Lessons
1. テストDataFrameは本番DataFrameと同じ列を持つことを保証する(race_idガード等)
2. alpha scaling方向は単体テストで必ず検証(test_narrower_alpha_narrower_interval)
3. PlaceTwoStageModel.RETURN_FEATURE_COLSのparity invariantを常に維持する
4. Optunaテストはn_trials=3で高速化、本番はn_trials=30

### Cost Observations
- Model mix: ~50% opus, ~35% sonnet, ~15% haiku
- Sessions: ~2
- Notable: Phase 6が19分(4 auto-fixed deviations含む)、Phase 7が26分

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | ~3 | 4 | GSD workflow導入、TDD gate確立 |
| v1.1 | ~2 | 3 | Optuna導入、post-model features確立 |

### Cumulative Quality

| Milestone | Tests | Coverage | LOC (src/) |
|-----------|-------|----------|------------|
| v1.0 | ~800 | mock-based | ~19,000 |
| v1.1 | 1,113 | mock-based | ~20,773 |

### Top Lessons (Verified Across Milestones)

1. TDD RED/GREEN gateは品質の最強の味方 — 全12プランで一貫して適用
2. テストモックパスの不一致が両マイルストーンで発生 — ローカルimportに直接パッチするパターンを確立
3. 自動deviation修正ルール(Rule 1-3)が2マイルストーン合計10件のバグを検出・修正
