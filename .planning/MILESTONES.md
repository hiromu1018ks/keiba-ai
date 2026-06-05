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

## v1.6 Feature Engineering Overhaul — 2026-05-17

**Status:** Shipped
**Phases:** 6 (23-28) | **Plans:** 14 | **LOC:** ~23,215

### Key Accomplishments

1. POST_RACE情報漏洩完全排除 + CQR whitelist化 + 3層CI漏洩検出テスト (Phase 23)
2. 100+特徴量Tier分類基盤 + コードハッシュキャッシュ無効化 + ノイズプルーニングパイプライン (Phase 24)
3. EveryDB2未活用データから22新特徴量抽出: mining 4 + 血統 4 + BMS 2 + record 1 + 相対比較 7 + 騎手/調教師/コンビ 12 (Phase 25-26)
4. ドメイン知識交互作用項12個(カテゴリ積3+数値積6) + OOF安全ターゲットエンコーディング3特徴量 (Phase 27)
5. マルチ年度BT (ROI 85.7%, +1.3pp改善) + 12モデルSHA256特徴量凍結manifest (Phase 28)

### Known Deferred Items

- ROI 100%目標未達 (85.7%、特徴量アプローチの限界)
- WF検証スクリプトの実際の実行(PostgreSQL環境必要、~4時間)
- Human UAT 5項目 (全てPostgreSQL依存)
- test_training_pipeline.py 3件既知失敗(RecordFeatures.compute mock問題)
- n_taisyogata_mining/n_sale/n_banusiテーブルからの特徴量抽出(未検証)

### Archive

- `.planning/milestones/v1.6-ROADMAP.md`
- `.planning/milestones/v1.6-REQUIREMENTS.md`

## v1.7 Market-Independent Edge Discovery — 2026-05-19

**Status:** ✅ Shipped
**Phases:** 6 (29-34) | **Plans:** 15 | **LOC:** ~24,100

### Key Accomplishments

1. ETL Pipeline拡張 — 三連複/馬連/三連単オッズParquet抽出 + DataRepository DI + カバレッジ検証
2. 11新特徴量を全12モデルに登録 — 6 rl_* (entropy/dispersion/gap等) + 5 MCF (Harville理論オッズ)
3. IC評価フレームワーク構築 — 4定式化IC (B差分/C直交/E Incremental/Per-race) + OOF save hook + CLI
4. Gain-per-Depth診断システム — 179特徴量カテゴリマップ + MDR/FAD メトリクス + 可視化CLI
5. BT 2024 ROI 85.7% → 97.8% (+12.1pp改善) — ダート107.4%、Aggressive 116.7%、高オッズ(10+) 179.9%
6. POST_RACE漏洩安全性検証 — 13テスト全通過 + Manifest v1.7凍結

### Known Deferred Items

- Phase 30/34 VERIFICATION.md不足 (11 requirements PARTIAL, implementation complete)
- ROI 100%目標未達 (97.8%、あと2.2pp)
- Turf conservative regime unprofitable
- training_pipeline _build_race_level_features() rl_*列処理未追加
- WF検証未実行 (~4時間、PostgreSQL環境必要)
- Human UAT 5項目 (PostgreSQL依存)

Known deferred items at close: 11 (see STATE.md Deferred Items)

### Archive

- `.planning/milestones/v1.7-ROADMAP.md`
- `.planning/milestones/v1.7-REQUIREMENTS.md`
- `.planning/milestones/v1.7-MILESTONE-AUDIT.md`

## v1.8 Turf Precision Calibration — 2026-05-20

**Status:** ✅ Shipped
**Phases:** 4 (35-36.1.1) | **Plans:** 10 | **LOC:** ~25,800

### Key Accomplishments

1. ETL Data Foundation — HaronTime/LapTime/Jyuni float64変換 + POST_RACE 41列化 + sentinel NaN化 + 3層CI漏洩検出
2. Feature Computation — TRF 3特徴量 + INT 3交互作用 + HLF Haron/Lap 7特徴量を12モデル全登録
3. HaronTime L4/LapTime Redesign — クロスレベル派生特徴量 (closing_speed_ratio, haron_race_gap, pace_adj_finish) + backtest engine hist_features修正
4. MarketModel & RaceQuality配線修正 — Phase36強特徴量の3モデル除外 + race aggregate追加 + EV Tail Calibration + v1.7差分診断

### Known Deferred Items

- BT ROI 87.8% (v1.7 was 97.8%) — Phase 36.1.1 fixes not BT-validated
- WF検証未実行 (~4時間)
- test_training_pipeline.py 3件既知失敗

Known deferred items at close: 8

### Archive

- `.planning/milestones/v2.0-ROADMAP.md` (combined v1.8+v2.0 archive)
- `.planning/milestones/v2.0-REQUIREMENTS.md`

## v2.0 Investment Pipeline Restructuring — 2026-05-27

**Status:** ✅ Shipped
**Phases:** 2 (37-38) | **Plans:** 5 | **LOC:** ~44,582 (src)

### Key Accomplishments

1. OOF Health Infrastructure — OOFHealthValidator基盤 (fail-fast validation + SHA256 manifest + anomaly detection) + ev_oof_fold fold assignment配線
2. InvestmentFeatureFrame — 94 specs / 9 categories スキーマレジストリ (InvestmentFeatureSpec frozen dataclass) + dual-mode builder (train/infer) + leakage guard + Parquet cache + sidecar manifest

### Known Deferred Items

- BT ROI 87.8% at v2.0 close, target 100%+
- HLF/TRF/INT features implemented but not BT-validated for IC improvement
- CAL-01~03 calibration layers deferred to Phase 39+
- VAL-02~06 validation metrics deferred to Phase 39+

Known deferred items at close: 12 (see STATE.md Deferred Items)

### Archive

- `.planning/milestones/v2.0-ROADMAP.md`
- `.planning/milestones/v2.0-REQUIREMENTS.md`

## v2.2 ROI Recovery Analysis — 2026-06-02

**Status:** Closed — not_deployable
**Phases:** 4 (43-46) | **Plans:** 8

### Key Accomplishments

1. Shadow Diagnosis — 2024/2025 fixed-foldでbaseline vs shadowのBrier/logloss/ECE、選定差分、APR乖離を分析
2. ROI Bisect — MAWC/Ranker/OBF/Selectionの帰属分析により、MAWCキャリブレーションを主要原因として特定
3. Structural Fix — 36-dim conservative MAWC variantを実装し、既存モデルを上書きせず別variantとして保存
4. Quality Gate Verification — Phase 46 runtimeでOOFHealthValidator/FeatureRoutingAuditはPASS、DeploymentGateEvaluatorはFAIL

### Final Runtime Verdict

| Item | Result |
|------|--------|
| Quality Gate | FAIL |
| Deployment | not_deployable |
| Baseline test ROI | -8.0% |
| Conservative MAWC test ROI | -11.3% |
| Decision | Do not replace baseline 51-dim MAWC |

### Known Deferred Items

- Conservative MAWCの全交互作用削除は過剰だったため、v2.3+で選択的interaction維持を検討
- DeploymentGateEvaluatorのoverall metric 0.0集計はtech debtとして見直し候補
- Ranker/OBF/selection thresholdはv2.2では変更せず、必要なら別マイルストーンで扱う

### Archive

- `.planning/milestones/v2.2-ROADMAP.md`
- `.planning/milestones/v2.2-REQUIREMENTS.md`
- `.planning/milestones/v2.2-MILESTONE-SUMMARY.md`

## v2.3 Track Condition Feature Integration — 2026-06-05

**Status:** ✅ Shipped
**Phases:** 4 (47-50) | **Plans:** 7

### Key Accomplishments

1. Track Condition ETL Pipeline — 含水率189K行・クッション値133K行CSV→23,259レースParquet変換 + DataRepository統合
2. 23 Track Condition Features — T1/T2(8) + T3(4) + T4(11) 全ティア実装、NaN-safe surface-aware設計 + 外科的ルーティング(6登録/4除外)
3. PIT-safe Horse Aptitude — 14列馬個体適性precompute (expanding window + shift(1)) + condition classification
4. Safety CI Validation — Feature Routing Audit PASS, POST_RACE 3層CI PASS, Surface-aware NaN 17/23 PASS, IC評価フレームワーク
5. Post-hoc EV Optimization — Raw ROI 87.3% → --min-win-ev 1.40 で 124.4% ROI (505 bets, +¥12,340) を発見

### Milestone Stats

| Metric | Value |
|--------|-------|
| Commits | 48 |
| Files changed | 62 |
| Lines added | +9,749 |
| Timeline | 2 days (2026-06-04 → 2026-06-05) |
| Tests | 2,503 |

### Known Deferred Items

- IC評価レポート生成(OOF予測が必要、別途run_train.py実行)
- 4 RACE_CONDITION特徴量の100% NaN修正(track_month_stats利用可能性問題)
- sire_x_cushion_bandの51.63% NaN改善(種牡馬×クッション交差データ不足)
- WF検証スクリプトの実際の実行(~4時間、PostgreSQL環境必要)
- Human UAT 5項目 (PostgreSQL依存)
- test_training_pipeline.py 3件既知失敗

Known deferred items at close: 6 (see STATE.md Deferred Items)

### Archive

- `.planning/milestones/v2.3-ROADMAP.md`
- `.planning/milestones/v2.3-REQUIREMENTS.md`
