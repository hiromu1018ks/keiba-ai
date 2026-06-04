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

## Milestone: v1.3 — Betting Strategy Optimization

**Shipped:** 2026-05-05
**Phases:** 3 | **Plans:** 7 | **Sessions:** ~2

### What Was Built
- OddsBandFilter (ROI<100%バンド除外) + EV_lower >= 1.0 フィルター + COLLAPSED regime skip
- StakeCalculator コンストラクタ注入 + apply_ev_scaling() + レジーム別Kelly分数注入
- DD再設計: ROI依存を排除したDD%のみ3段階制御 + DDConfig dataclass + ヒステリシス
- RegimeDetector override_params外部化 + MetaSwitcher値乖離解消
- ParameterFreezeProtocol: JSON manifest + SHA256改ざん検知
- StrategyOptimizer: Optuna TPE 14次元最適化 + 軽量Walk-forward 2fold + CLI

### What Worked
- Phase 11→12→13の直列依存がデータフロー(フィルター→サイジング→チューニング)を明確にした
- DDConfig dataclass化がDD制御のテスタビリティを劇的に向上(全パラメータ外部注入)
- ヒステリシス付き状態機械がDD制御の安定性を担保(発振防止)
- SHA256 manifestがルックアヘッドバイアス防止を機械的に保証

### What Was Inefficient
- REQUIREMENTS.mdのチェック更新が実装と同期していなかった(SUMMARY.mdでは完了だがREQUIREMENTS.md未更新)
- Optuna 4.x互換性問題(TrialState importパス変更)にテスト実行時まで気づかなかった
- テスト実行中に3つのバグを発見(Optuna import, DD閾値逆転, mock patch paths)

### Patterns Established
- コンストラクタ注入パターン: fractional_kelly/DDConfig/override_params 全て__init__で外部注入可能
- フィルター適用順序: COLLAPSED skip (race-level) → EV filter (candidate-level) → OddsBandFilter (candidate-level)
- JSON manifest + SHA256: sort_keys=True + indent=2 でdeterministic保証
- Lazy import pattern: _run_single_backtest内でimport(model_loader依存分離)

### Key Lessons
1. Optuna 4.xでは `from optuna.trial import TrialState` を使う(`optuna.TrialState`は4.xで非公開)
2. 独立範囲のサンプラーが閾値逆転を生成する可能性あり → _build_strategy_config内で補正が必要
3. テストモックパスはlazy importの実際のソースモジュールにパッチする(モジュール属性に非依存)
4. DD制御は的中率環境に合わせて設計する(10%的中率ではROIはノイズすぎる)

### Cost Observations
- Model mix: ~40% opus, ~40% sonnet, ~20% haiku
- Sessions: ~2
- Notable: Phase 13が最長(26分、3 plans)、Phase 12が最短(5分、2 plans)

---

## Milestone: v1.4 — Ensemble Filter Recalibration

**Shipped:** 2026-05-07
**Phases:** 5 | **Plans:** 10 | **Sessions:** ~3

### What Was Built
- WinSelectionGate ensemble OOF再学習 + KS/Wassersteinドリフト診断 + use_ensemble伝播バグ修正
- EV_lower固定1.0→OOF 25th percentile動的化 + EV診断(ECE/Brier/Reliability/時系列ドリフト)
- ルックアヘッドバイアス修正 + アンサンブルベースtraining_bet_history生成 + OddsBandFilter再キャリブレーション
- 16次元Optuna最適化 + 4fold化 + multi-seed(42/43/44)安定性検証 + 不安定次元自動固定
- PFP SHA256改ざん検知二重検証 + 自動検証レポート生成(ROI判定 + 5項目原因分析)

### What Worked
- Phase 14→15→16→17→18の直列依存がデータフロー(gate→filter→band→optimize→validate)を明確にした
- ドリフト診断の数値化(KS/Wasserstein)がフィルター再キャリブレーションの必要性を定量的に裏付けた
- multi-seed安定性検証が不安定な次元を自動検出し、過学習耐性を客観的に評価できた
- 検証レポートの原因分析5項目が「なぜROI<100%か」のデバッグを構造化した

### What Was Inefficient
- 全5フェーズのHuman UATがPostgreSQL環境依存で実行できず、コード完成と実際の検証が分離された
- Optunaテストのモックパス不一致が再発(ローカルimportパターン)
- Phase 17の16次元拡張でテストのモック更新が多岐にわたり、修正コストが高かった

### Patterns Established
- OOF分布ベースの動的閾値: 固定値ではなく学習データ分布から閾値を導出するパターン
- ルックアヘッドバイアス防止: training_bet_history生成は常にデフォルトパラメータを使用
- PFP二重検証: freeze(freeze時) + verify(使用時) で全return pathで整合性を保証
- 検証レポート構造化: ROI判定 + 原因分析(odds band/regime/EV/bet count/surface)の5項目テンプレート

### Key Lessons
1. フィルター閾値はモデル出力分布に適合させる必要がある(単一モデルとアンサンブルで分布が異なる)
2. Optuna次元追加時は既存テストのモックを全て更新する必要がある(サジェスト関数の引数が増える)
3. ルックアヘッドバイアスは最適化ループ内でのパラメータ使用に潜みやすい — 明示的分離が必要
4. 改ざん検知はfreezeとverifyを分離することで、全コードパスで整合性を検証できる

### Cost Observations
- Model mix: ~35% opus, ~40% sonnet, ~25% haiku
- Sessions: ~3
- Notable: Phase 17が最長(4fold + 16次元最適化)、Phase 14が最短(診断モジュール)

---

## Milestone: v1.6 — Feature Engineering Overhaul

**Shipped:** 2026-05-17
**Phases:** 6 (23-28) | **Plans:** 14 | **Sessions:** ~5

### What Was Built
- POST_RACE情報漏洩完全排除 — whitelist FEATURE_COLS + 3層CI検出テスト
- 100+特徴量Tier分類 + ノイズ特徴量プルーニング + コードハッシュキャッシュ無効化
- EveryDB2未活用テーブルから22新特徴量(mining/血統/BMS/record/相対比較/騎手/調教師/コンビ)
- 12ドメイン知識交互作用項(カテゴリ積3+数値積6+既存3) + OOF安全3-fold TE 3特徴量
- マルチ年度BT (ROI 85.7%, +1.3pp) + 12モデルSHA256特徴量凍結manifest

### What Worked
- Phase 23 Safety Gateが最優先だった — 漏洩排除なしに新特徴量追加は信頼性ゼロ
- _train_submodel() Group A-G パターンが新特徴量モジュール統合を標準化
- TDD RED/GREEN gateがPhase 26-27の22+15新特徴量追加で品質を維持
- groupby("race_id")相対特徴量がレース内比較の自然な設計を提供
- SHA256凍結manifestがFEATURE_COLS変更をトレーサブルに

### What Was Inefficient
- Phase 25 Quick Win 12特徴量がROIに全く寄与しなかった(LightGBM gain=0)
- BacktestEngine horse-level特徴量mergeキー不整合で4連続fixコミットが必要だった
- 推論パスに6特徴量欠落(stage2 relative + target encoding) — training/predictionの二重管理が原因
- 22新特徴量+12交互作用+3TE追加でROI+1.3ppのみ — 投入対効果が悪い
- test_win_feature_analysis.pyのoriginal_allハードコードリストが毎フェーズ手動更新必要

### Patterns Established
- POST_RACE whitelist: blacklistではなくFEATURE_COLS whitelistで安全保証
- _train_submodel() Group integration: 新特徴量モジュールはGroup X blockで統合
- wide-to-long pivot: n_mining 18頭wide format → long変換パターン
- OOF safe TE: 3-fold expanding window + Beta smoothing + cold start global mean
- Feature freeze manifest: 12モデルFEATURE_COLS JSON + SHA256で凍結

### Key Lessons
1. 特徴量の量より質 — 37新特徴量でROI+1.3ppは限界。次はアプローチ自体を見直す必要
2. training/predictionパスの二重管理はバグの温床 — 推論パス欠落は気づきにくい
3. LightGBMは不要特徴量を自動的にgain=0にするが、特徴量数増加は学習時間を延ばす
4. PIT監査(POST_RACE分類)は新特徴量追加の前提条件 — 安全性確認なしに進めるとリーク混入
5. テストのハードコードリスト(original_all)はFEATURE_COLS変更のボトルネック — 動的取得を検討

### Cost Observations
- Model mix: ~50% opus, ~35% sonnet, ~15% haiku
- Sessions: ~5
- Notable: Phase 28-02 マルチ年度BTが5時間(実行待ち含む)、Phase 27が3プラン27分で最効率的

---

## Milestone: v1.8 — Turf Precision Calibration

**Shipped:** 2026-05-20
**Phases:** 4 (35-36.1.1) | **Plans:** 10 | **Sessions:** ~3

### What Was Built
- ETL Data Foundation — HaronTime/LapTime/Jyuni float64変換 + POST_RACE 41列化 + sentinel NaN化
- Feature Computation — TRF 3特徴量 + INT 3交互作用 + HLF Haron/Lap 7特徴量 + 12モデル全登録
- HaronTime L4/LapTime Redesign — クロスレベル派生3特徴量 (closing_speed_ratio, haron_race_gap, pace_adj_finish)
- MarketModel & RaceQuality配線修正 — 27特徴量/モデルの外科的除外 + race aggregate追加 + EV Tail Calibration

### What Worked
- Phase 35のETL基盤が一括構築され、Phase 36の特徴量計算がスムーズに進行
- HaronTime sentinelルールの宣言型dict設計が複数センチネルパターンを統一的に処理
- Phase 36.1.1の原因診断(4項目)がルーティング修正を的確にガイド
- v1.7 vs v2.0 差分診断スクリプトがROI低下の原因分解を定量的に示した

### What Was Inefficient
- Phase 36強特徴量の一律登録がMarketModel/RaceQualityを崩壊 — Phase 36.1.1で修正に4プラン消費
- HaronTimeL4データソース誤り(entries vs races)がPhase 36.1で発覚 — 再設計が必要だった
- BT ROI 97.8%→87.8%の低下をPhase 36.1.1完了後に検証できず(BT再実行未完了)

### Patterns Established
- sentinelルール宣言型dict: columns/sentinels/divisor keysでETL変換ルールを一元管理
- 外科的特徴量除外: 全モデル一律登録ではなく、モデル役割に応じたFEATURE_COLS管理
- 差分診断スクリプト: v[N-1] vs v[N] のmerge-based classificationでROI変動原因を分解

### Key Lessons
1. 強特徴量の一律登録は市場モデル等の特殊役割モデルを崩壊させる — モデル役割別ルーティングが必須
2. POST_RACE情報のETLはsentinelパターンが複数(000/999/00)あるため宣言型ルールで統一する
3. 特徴量追加後は必ずBT再実行でROI検証する — コード修正完了≠ROI改善確認

### Cost Observations
- Model mix: ~45% opus, ~40% sonnet, ~15% haiku
- Sessions: ~3
- Notable: Phase 36.1.1が4プラン中2プランでworktree使用、Phase 35は10分で完了

---

## Milestone: v2.0 — Investment Pipeline Restructuring

**Shipped:** 2026-05-27
**Phases:** 2 (37-38) | **Plans:** 5 | **Sessions:** ~3

### What Was Built
- OOF Health Infrastructure — OOFHealthValidator (fail-fast + anomaly detection + SHA256 manifest) + ev_oof_fold配線
- InvestmentFeatureFrame — 94 specs / 9 categories schema registry + dual-mode builder (train/infer) + leakage guard + Parquet cache

### What Worked
- OOFHealthValidatorのfrozen dataclass + profile-driven validationが型安全性と拡張性を両立
- InvestmentFeatureSpec frozen dataclassが94 specsのメタデータをコンパイル時に検証
- dual-mode builder (train=infer同一スキーマ) が学習/推論パスの整合性を機械的に保証
- Parquet cache + sidecar manifestが決定性出力を保証(同一入力→同一出力)
- コードレビュー(CR-01/02, WR-01~04)が命名・docstring・edge case品質を向上

### What Was Inefficient
- Phase 37 worktreeのmergeでテストが失敗 — mock OOFHealthValidatorのパス調整が必要だった
- Phase 38 plan 02のderived featuresが20個に膨張 — 94 specsの内20がderivedで計算コスト増
- スキーマレジストリ94 specsの定義が手作業中心 — 自動生成ツールがあれば効率化可能

### Patterns Established
- frozen dataclass schema registry: InvestmentFeatureSpecで全特徴量のメタデータを型安全に管理
- dual-mode builder: train mode (OOF-safe) / infer mode (production) で同一出力スキーマ
- fail-fast at save point: OOF保存時に健全性検査、異常は下流に伝播前に検出
- sidecar manifest: Parquetファイル横にJSON manifestで決定性・トレーサビリティ保証

### Key Lessons
1. frozen dataclass + dict-based FEATURE_SPEPSが大規模スキーマ管理に有効 — 追加・変更が安全
2. train/infer同一スキーマはテスト可能なアサーションで機械的に検証すべき
3. OOF検証は保存時(producer-side)が最適 — consumer-sideでは既に汚染データが使われる
4. コードレビューの命名指摘(CR-01/02)はSPEC段階で防げる — 仕様レビューの重要性

### Cost Observations
- Model mix: ~40% opus, ~45% sonnet, ~15% haiku
- Sessions: ~3
- Notable: Phase 38 plan 02がderived features 20個で最長、Phase 37 plan 01が29テストでTDD効率的

---

## Milestone: v2.2 — ROI Recovery Analysis

**Shipped:** 2026-06-02 (closed — not_deployable)
**Phases:** 4 (43-46) | **Plans:** 8 | **Sessions:** ~4

### What Was Built
- Shadow Diagnosis — 3ステップ段階的除外診断エンジン + 5セグメント別APR/ECE乖離分析
- ROI Bisect — ComponentAttributionエンジン(逐次帰属+条件付きSHAP) + HistoricalBisect(v1.7→v2.0補助比較)
- Structural Fix — MawcConservativeRetrainer(36-dim, C grid [0.003-0.03], 100倍強正則化) + favorite band guard
- Quality Gate Verification — QualityGateOrchestrator(2-stage flow + 3-label framework) + RUNBOOK + 手動再現手順

### What Worked
- 診断→ビセクション→修正→検証の段階的アプローチが原因特定に効果的だった
- ComponentAttributionの係数分析でMAWCのbeta_market=0.90支配(logit_market coef=0.39)を特定
- Shadow Comparison Frameworkが品質劣化の定量的比較基盤として有効
- conservative MAWCを既存モデル上書きではなく別variant保存にした判断が安全だった
- QualityGateOrchestratorの3-label framework(Quality Gate/ROI Trend/Deployment)が判定を明確化

### What Was Inefficient
- Conservative MAWCの全交互作用削除(15個)が過剰だった — 36-dimでは正則化が強すぎてtest汎化しなかった
- Phase 46 runtime Stage 2で6/18 conditions FAIL — 学習時ROI改善(288-317%)がtest結果(-11.3%)と逆行
- Historical bisectの信頼度がLOW — Phase 35-36間のartifactが不完全で推定に留まった
- DeploymentGateEvaluatorのoverall metric 0.0集計バグがtech debtとして残存

### Patterns Established
- not_deployable マイルストーンの記録パターン — 品質ゲートFAILでも診断・分析成果物は残す
- 3-label deployment判定: Quality Gate(PASS/FAIL) × ROI Trend × Deployment verdict
- conservative variant pattern — 既存モデルを上書きせず別ディレクトリに保存

### Key Lessons
1. MAWCの市場過重ブレンド(beta_market=0.90)が確率過度圧縮を引き起こす — 交互作用項の削除は逆効果だった
2. 学習時ROI改善とtest汎化は別物 — training ROI 288-317% vs test ROI -11.3%の乖離が示す通り
3. 特徴量削減アプローチより、正則化強度の調整が本質 — 51-dim→36-dimよりC値の適正範囲探求が先
4. ROI回復は「何を直すか」より「どこが壊れているか」の特定が8割 — Shadow/Bisectの診断価値が高い
5. not_deployableでもマイルストーンとして閉じることで、分析成果物のトレーサビリティを保持できる

### Cost Observations
- Model mix: ~40% opus, ~40% sonnet, ~20% haiku
- Sessions: ~4
- Notable: Phase 46がruntime verification主体でセッション長、Phase 44 bisectが1222行のComponentAttribution実装で最大

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | ~3 | 4 | GSD workflow導入、TDD gate確立 |
| v1.1 | ~2 | 3 | Optuna導入、post-model features確立 |
| v1.2 | ~1 | 3 | バックテストパイプライン高速化 |
| v1.3 | ~2 | 3 | ベット戦略最適化、コンストラクタ注入パターン確立 |
| v1.4 | ~3 | 5 | フィルター再キャリブレーション、動的閾値、PFP二重検証 |
| v1.5 | ~3 | 5 | EVキャリブレーション、高オッズ特徴量、CQR Conformal区間 |
| v1.6 | ~5 | 6 | 特徴量オーバーホール、漏洩排除、EveryDB2新特徴量、TE導入 |
| v1.7 | ~2 | 6 | 市場独立性獲得、IC評価フレームワーク、GPD診断 |
| v1.8 | ~3 | 4 | 上がりタイムETL、芝相対特徴量、MarketModel配線修正 |
| v2.0 | ~3 | 2 | OOF検査基盤、投資特徴量フレーム、schema registry |
| v2.1 | ~2 | 4 | Shadow Comparison、DeploymentGate、Feature Routing Audit |
| v2.2 | ~4 | 4 | ROI Recovery Analysis(not_deployable)、3-label判定framework |

### Cumulative Quality

| Milestone | Tests | Coverage | LOC (src/) |
|-----------|-------|----------|------------|
| v1.0 | ~800 | mock-based | ~19,000 |
| v1.1 | 1,113 | mock-based | ~20,773 |
| v1.3 | 1,200+ | mock-based | ~18,820 |
| v1.4 | 1,327 | mock-based | ~19,300 |
| v1.5 | 1,392+ | mock-based | ~24,970 |
| v1.6 | 1,527 | mock-based | ~23,215 |
| v1.7 | 1,540+ | mock-based | ~24,100 |
| v2.0 | 2,056 | mock-based | ~44,582 |
| v2.1 | 2,231 | mock-based | ~46,400 |
| v2.2 | 2,343 | mock-based | ~48,200 |

### Top Lessons (Verified Across Milestones)

1. TDD RED/GREEN gateは品質の最強の味方 — 全84プラン(v1.0-v2.0)で一貫して適用
2. テストモックパスの不一致が複数マイルストーンで発生 — ローカルimportに直接パッチするパターンを確立
3. 自動deviation修正ルール(Rule 1-3)が10マイルストーン合計25+件のバグを検出・修正
4. コンストラクタ注入パターンがテスタビリティを劇的に向上 — Optuna最適化の前提となる
5. フィルター閾値はモデル出力分布に適合させる必要がある — v1.4で実証
6. 特徴量の量より質 — 37新特徴量でROI+1.3ppは限界、アプローチ自体の見直しが必要 — v1.6で実証
7. 強特徴量の一律登録は特殊役割モデルを崩壊させる — モデル役割別ルーティングが必須 — v1.8で実証
8. frozen dataclass schema registryが大規模特徴量管理に有効 — v2.0で確立
9. Shadow-first deploymentが確率品質劣化を安全に検出 — 新パイプラインはshadow modeで品質ゲート通過まで不活性 — v2.1で確立
10. 学習時ROI改善とtest汎化は別物 — training 288% vs test -11.3%の乖離。正則化強度調整が特徴量削減より本質 — v2.2で実証
