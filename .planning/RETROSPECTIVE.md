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

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | ~3 | 4 | GSD workflow導入、TDD gate確立 |
| v1.1 | ~2 | 3 | Optuna導入、post-model features確立 |
| v1.2 | ~1 | 3 | バックテストパイプライン高速化 |
| v1.3 | ~2 | 3 | ベット戦略最適化、コンストラクタ注入パターン確立 |
| v1.4 | ~3 | 5 | フィルター再キャリブレーション、動的閾値、PFP二重検証 |

### Cumulative Quality

| Milestone | Tests | Coverage | LOC (src/) |
|-----------|-------|----------|------------|
| v1.0 | ~800 | mock-based | ~19,000 |
| v1.1 | 1,113 | mock-based | ~20,773 |
| v1.3 | 1,200+ | mock-based | ~18,820 |
| v1.4 | 1,327 | mock-based | ~19,300 |

### Top Lessons (Verified Across Milestones)

1. TDD RED/GREEN gateは品質の最強の味方 — 全38プランで一貫して適用
2. テストモックパスの不一致が複数マイルストーンで発生 — ローカルimportに直接パッチするパターンを確立
3. 自動deviation修正ルール(Rule 1-3)が5マイルストーン合計14+件のバグを検出・修正
4. コンストラクタ注入パターンがテスタビリティを劇的に向上 — Optuna最適化の前提となる
5. フィルター閾値はモデル出力分布に適合させる必要がある — v1.4で実証
