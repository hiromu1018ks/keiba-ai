# Milestones

## v1.0 Win Model — 2026-05-03

**Status:** ✅ Shipped
**Phases:** 4 | **Plans:** 7 | **LOC:** 48,528

### Key Accomplishments

1. SHAP/gain特徴量重要度分析 + 6新特徴量(odds-to-ability比、クラス落リバウンド等)追加
2. WinBenterGate実装(基本確率×市場確率ブレンド) + Beta/Isotonicキャリブレーション比較
3. WinSelectionGate + Conformal信頼性推定 + JRA控除率25%考慮edge_threshold調整
4. WFValidationResult + 過学習検出(ROI gap/一貫性/Spearman安定性の3基準) + run_wf_validation.py CLI

### Known Deferred Items

- WF検証スクリプトの実際の実行(PostgreSQL環境必要、~4時間)
- Human UAT 2項目(04-HUMAN-UAT.md)

### Archive

- `.planning/milestones/v1.0-ROADMAP.md`
- `.planning/milestones/v1.0-REQUIREMENTS.md`

## v1.1 ROI Advanced Model — 2026-05-03

**Status:** ✅ Shipped
**Phases:** 3 (5-7) | **Plans:** 5 | **LOC:** ~20,773

### Key Accomplishments

1. 9新特徴量追加: EMA重み付けハロンタイム・クラス調整フォーメトリック・z-score改善トラジェクトリ・ペースフィグア3サブ特徴量・オッズ変動2次微分/方向一貫性
2. Odds Deviation EV: deviation_rank/zscore + Conformal EV区間(80%/90% 2レベル) + conformal_confidence_score統合
3. 3モデルスタッキング多様性強制: Optuna探索空間分離 + early stopping + feature_fraction最適化
4. 多様性検証: OOF予測相関 + feature importance Spearman順位相関の二重検証

### Known Deferred Items

- WF検証スクリプトの実際の実行(PostgreSQL環境必要、~4時間)
- Human UAT 3項目(01-HUMAN-UAT, 04-HUMAN-UAT, 07-UAT)
- バックテストROI検証(run_backtest.py実行、PostgreSQL環境必要)

### Archive

- `.planning/milestones/v1.1-ROADMAP.md`
- `.planning/milestones/v1.1-REQUIREMENTS.md`

## v1.3 Betting Strategy Optimization — 2026-05-05

**Status:** ✅ Shipped
**Phases:** 3 (11-13) | **Plans:** 7 | **LOC:** ~18,820

### Key Accomplishments

1. OddsBandFilter + EV_lower フィルター — 赤字オッズバンド除外とEV下限フィルターでベット品質を向上
2. COLLAPSED regime skip — 不安定市場でのベット完全スキップ + 除外カウンター統計
3. Kelly基準レジーム別サイジング — AGGRESSIVE/CONSERVATIVE/COLLAPSED別fractional_kelly注入
4. DD再設計 (DD%のみ3段階制御) — ROI依存を排除しヒステリシス付き状態機械に再設計
5. Optuna TPE 14次元最適化 — 全戦略パラメータの同時最適化 + Walk-forward 2fold評価
6. ParameterFreezeProtocol — JSON manifest + SHA256改ざん検知でルックアヘッドバイアス防止

### Known Deferred Items

- バックテストROI検証(run_backtest.py実行、PostgreSQL環境必要)
- Look-ahead bias risk in parameter optimization — walk-forward validation required
- Regime detector oscillation risk — hysteresis counter may need adjustment
- PostgreSQL環境が必要な検証が複数残存(WF検証、バックテスト)

### Archive

- `.planning/milestones/v1.3-ROADMAP.md`
- `.planning/milestones/v1.3-REQUIREMENTS.md`
