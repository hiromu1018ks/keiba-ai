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

## v1.4 Ensemble Filter Recalibration — 2026-05-07

**Status:** ✅ Shipped
**Phases:** 5 (14-18) | **Plans:** 10 | **LOC:** ~19,300

### Key Accomplishments

1. WinSelectionGate ensemble OOF再学習 + KS/Wassersteinドリフト診断 + use_ensemble伝播バグ修正
2. EV_lower固定1.0→OOF 25th percentile動的化で過剰除外3,594件を解消 + EV診断(ECE/Brier/Reliability)
3. ルックアヘッドバイアス修正 + アンサンブルベースtraining_bet_history生成 + OddsBandFilter再キャリブレーション
4. 16次元Optuna最適化 + 4fold化 + multi-seed(42/43/44)安定性検証 + 不安定次元自動固定
5. PFP SHA256改ざん検知二重検証 + 自動検証レポート生成(ROI判定 + 5項目原因分析)

### Known Deferred Items

- Human UAT 5項目 (全てPostgreSQL依存): ROI確認、EV filter除外確認、Optuna実行確認、multi-seed確認、レポート目視確認
- VAL-01 partial: code complete, actual ROI>100% + 100+ bets requires PostgreSQL
- 2 cross-phase warnings (MLflow fallback path, engine early returns — both non-impactful)
- Minor: 23 ruff lint warnings in test_strategy_optimizer.py (N806, E501)

### Archive

- `.planning/milestones/v1.4-ROADMAP.md`
- `.planning/milestones/v1.4-REQUIREMENTS.md`
- `.planning/milestones/v1.4-MILESTONE-AUDIT.md`

## v1.5 Model Accuracy Improvement — 2026-05-10

**Status:** ✅ Shipped
**Phases:** 5 (19, 19.1, 20, 21, 22) | **Plans:** 13 | **LOC:** ~24,970

### Key Accomplishments

1. Isotonic EVキャリブレーション + オッズバンド別補正層 — EV過大評価2.42倍を是正
2. バックテスト高速化5段階 — キャリブレーションBT条件付きスキップ、Categorical包括適用、特徴量キャッシュ強化
3. 高オッズ的中18新特徴量 — クラストラジェクトリ、フォーム改善率、環境変化適性
4. CQR Conformal EV予測区間 — 80%/90%信頼区間 + 動的EV_lowerフィルタリング
5. 統合バックテスト — ROI v1.4:83.1% → v1.5:84.4% (+1.3pp改善、目標95%は未達)

### Known Deferred Items

- ROI 95%目標未達 (84.4%、次マイルストーンで改善要)
- CQR過学習修正済み(f3a4c10)だが、CQR設計自体の見直しが必要
- WF検証未実行 (過学習の有無未確認)
- 高オッズ帯(20+)でのベット機会なし
- debug/data-leak-phase-20-22.md (status: diagnosed、修正済み)

### Archive

- `.planning/milestones/v1.5-ROADMAP.md`
