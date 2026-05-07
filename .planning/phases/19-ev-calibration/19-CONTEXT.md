# Phase 19: EV推定キャリブレーション - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>
## Phase Boundary

既存EVCorrectionModelのP×E補正後にIsotonic Regressionを適用し、OOF予測ベースでEVを直接キャリブレーションして、全セグメントのEV過大評価倍率を1.0±0.2に収束させる。

**In scope:**
- EVC-01: OOF予測ベースのIsotonic EVキャリブレーション。ev_win_corrected → Isotonic → ev_win_calibrated
- EVC-02: オッズバンド別EV補正層。Isotonic適用後の残差をオッズバンド別に補正
- EVC-03: EVCorrectionModel統合。correct_ev()内でIsotonic + オッズバンド補正を適用
- OOF EV予測生成のTrainingPipelineV5組み込み
- IsotonicモデルのSubmodelSetへの追加とModelLoader対応
- サーフェス別(芝/ダート)独立Isotonicモデルの学習・保存
- mockベースの自動テスト

**Out of scope:**
- 新しい特徴量の追加 (Phase 20)
- Conformal EV予測区間 (Phase 21)
- バックテストROI検証 (Phase 22)
- 複勝/ワイドモデルのIsotonic対応
- LightGBMハイパーパラメータの再最適化
- Isotonic以外のキャリブレーション手法(Platt scaling等)

</domain>

<decisions>
## Implementation Decisions

### Isotonic適用方式 (EVC-01)
- **D-01:** 既存P×E補正(ev_win_corrected)の後にIsotonicを適用する。二重補正構成。P×E補正が非線形パターンを捉え、Isotonicが残存する体系的バイアスを非パラメトリックに吸収する。Isotonic単体置き換えではなく上乗せ。
- **D-02:** Isotonicの適用単位はサーフェス別(芝/ダート)独立モデル。芝とダートはオッズ分布・的中率が異なるため別キャリブレーションが自然。オッズバンド別の独立Isotonicはサンプル不足リスクがあるため採用しない。
- **D-03:** Isotonicの学習ターゲットは EV→actual_return 直接。X=ev_win_corrected(OOF)、y=actual_return(OOF)。ECE改善に最も直接的でSuccess Criteriaに正確に対応。
- **D-04:** Isotonicの境界処理は y_min=0, out_of_bounds='clip'。EVが負になるのを防ぎ、学習範囲外の予測は最近傍の学習値にクリップ。sklearn標準の安全な設定。

### OOF生成・保存設計 (EVC-01)
- **D-05:** OOF EV予測はTrainingPipelineV5の_train_submodel()内で生成する。学習パイプラインの既存フロー内で一貫して処理。独立スクリプトやフラグ制御は行わない。
- **D-06:** IsotonicモデルはSubmodelSetの新しいフィールドとして保存する。既存のモデル保存/読み込みパターンと統一。ModelLoaderが自動的に読み込み、PFP改ざん検知の対象にも含まれる。
- **D-07:** バックテスト時は学習済みIsotonicをロードして適用するのみ。テスト期間中の再学習はデータリークになるため行わない。バックテストの学習フェーズ(pipeline.run())でIsotonicは毎回再学習される。

### パイプライン統合 (EVC-03)
- **D-08:** IsotonicキャリブレーションはEVCorrectionModel.correct_ev()の最後で適用する。ev_win_corrected → Isotonic → ev_win_calibrated。呼び出し側(RacePredictor、BacktestEngine)の変更を最小化。EVC-03要件に合致。
- **D-09:** OOF EV予測生成のために_train_submodel()内でK-foldループを追加する。AbilityModel OOF → WinTwoStage predict → EVCorrection correct_ev → Isotonic fitのチェーン。

### オッズバンド別補正 (EVC-02)
- **D-10:** オッズバンド別補正はIsotonic適用後の残差に対して適用する。補正順序: P×E補正 → Isotonic → オッズバンド別補正。オッズバンド境界はPhase 16 D-08の固定値 `[1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+]` を維持。

### Claude's Discretion
- _train_submodel()内でのOOF EV生成の具体的なK-fold実装(分割数、時系列ソート)
- Isotonic適用後のオッズバンド別補正の具体的な手法(バンド別スケーリング、回帰、等)
- SubmodelSetへのIsotonicフィールドの命名規則と保存形式(.joblib)
- ModelLoaderの読み込み拡張の実装詳細
- OOF EV予測のメモリ管理(大量データの処理)
- テストのfixtureデータとモック構成
- correct_ev()へのIsotonic適用の具体的な実装(初期化判定、未学習時のフォールバック)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### EVCorrectionModel（主変更対象 — EVC-01/EVC-03）
- `src/models/ev_correction_model.py` — EVCorrectionModel実装。P補正×E補正のcorrect_ev()。Isotonic適用ポイント
- `src/models/ev_correction_model.py:293-330` — correct_ev()メソッド。ev_win_corrected生成後にIsotonic適用を追加

### TrainingPipelineV5（OOF生成ポイント — EVC-01/D-05/D-09）
- `src/pipelines/training_pipeline.py` — _train_submodel()内でOOF EV生成を追加
- `src/pipelines/training_pipeline.py:282` — AbilityModel→WinTwoStage→EVCorrection学習チェーン

### StackedEnsemble（OOF生成パターン参照）
- `src/models/stacked_ensemble.py:68-97` — K-fold OOF予測生成の既存パターン。_train_submodel()のOOF実装の参考

### SubmodelSet（Isotonic保存先 — EVC-03/D-06）
- `src/domain/models.py:229` — SubmodelSet dataclass。Isotonicフィールドの追加ポイント
- `src/db/model_loader.py:365-570` — ModelLoader._load_from_local()。Isotonicモデル読み込みの追加ポイント
- `src/pipelines/training_pipeline.py:934-1045` — _save_models_local()。Isotonicモデル保存の追加ポイント

### RacePredictor（推論パイプライン — EVC-03）
- `src/backtest/race_predictor.py:420-480` — get_win_candidates()。ev_win_correctedを使用する箇所
- `src/backtest/race_predictor.py:89` — predict()メソッド

### EV診断（品質評価 — EVC-01 Success Criteria）
- `src/models/ev_diagnostics.py` — compute_ev_diagnostics()。ECE/Brier/Reliability評価
- `src/models/ev_diagnostics.py:22-24` — 診断対象列定義(EV_PRED_COLUMN等)。ev_win_calibrated対応

### バックテストエンジン
- `src/backtest/engine.py:414-429` — run()メソッド。strategy_params受け取り
- `src/backtest/engine.py:472-475` — POST_RACE除外。Isotonic適用後もこの制約を維持

### オッズバンド関連（EVC-02）
- `src/betting/odds_band_filter.py` — OddsBandFilter。バンド境界 `[1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+]` の定義
- `src/models/ev_diagnostics.py:160-282` — compute_ev_diagnostics()。オッズバンド別EV過大評価分析

### 前フェーズのCONTEXT（必読 — 決定の連続性）
- `.planning/phases/18-validation-freeze/18-CONTEXT.md` — Phase 18決定(PFP検証、manifest注入)
- `.planning/phases/17-optuna-optimization/17-CONTEXT.md` — Phase 17決定(16次元Optuna、4fold、multi-seed)
- `.planning/phases/16-odds-band-rebuild/16-CONTEXT.md` — Phase 16決定(ルックアヘッド修正、固定バンド境界)
- `.planning/phases/15-ev-filter-enhancement/15-CONTEXT.md` — Phase 15決定(EV_lower動的閾値、EV診断)

### 要件定義
- `.planning/REQUIREMENTS.md` — EVC-01, EVC-02, EVC-03の要件定義
- `.planning/ROADMAP.md` — Phase 19 Success Criteria

### ドメイン型
- `src/domain/types.py:29-34` — RegimeState enum

### 設定ファイル
- `config/settings.yaml` — database, paths, feature_engine設定

### 既存テストパターン
- `tests/test_backtest_engine.py` — BacktestEngine既存テスト(1198行)。mockベース
- `tests/test_ev_correction_model.py` — EVCorrectionModel既存テスト。mockベース

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **EVCorrectionModel** (`src/models/ev_correction_model.py`): P補正×E補正のcorrect_ev()が完備。Isotonic適用ポイントとして最後にev_win_calibrated生成を追加するのみ
- **StackedEnsemble K-fold OOF** (`src/models/stacked_ensemble.py:68-97`): K-fold OOF予測生成の既存パターン。_train_submodel()のOOF EV生成の設計テンプレート
- **SubmodelSet** (`src/domain/models.py:229`): Isotonicフィールドの追加先。既存パターンでModelLoaderが自動読み込み
- **EV診断** (`src/models/ev_diagnostics.py`): ECE/Brier/Reliabilityの計算が既に実装済み。ev_win_calibratedの品質評価にそのまま使用可能
- **sklearn IsotonicRegression**: 既に依存関係に含まれる(scikit-learn >=1.4)。追加インストール不要

### Established Patterns
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。Phase 19テストもこのパターンに従う
- **コンストラクタ注入パターン**: パラメータはコンストラクタ引数で注入。Phase 12-13で確立
- **時系列分割**: train/valid分割は必ずrace_dateでソート後、前80%/後20%に分割。look-ahead bias防止
- **SubmodelSet保存/読み込み**: 新しいモデルフィールドはSubmodelSetに追加→ModelLoader._load_from_local()と_save_models_local()の両方を更新
- **パイプライン統合パターン**: use_ensemble=True時の自動診断・キャリブレーション実行。Phase 14-16で確立

### Integration Points
- **ev_correction_model.py:328-330** — correct_ev()のev_win_corrected生成直後。Isotonic適用を追加(D-08)
- **training_pipeline.py:282** — _train_submodel()の学習チェーン。OOF EV生成のK-foldループを追加(D-05, D-09)
- **domain/models.py:229** — SubmodelSet dataclass。Isotonicフィールド追加(D-06)
- **db/model_loader.py:365-570** — _load_from_local()。Isotonic読み込み追加
- **models/ev_diagnostics.py:22-24** — EV_PRED_COLUMN定義。ev_win_calibratedへの対応検討

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- Isotonic Regressionの非パラメトリック性質は過学習リスクが低く、EV過大評価の体系的バイアス除去に最適
- サーフェス別Isotonicはオッズバンド別よりサンプル効率が良く、芝/ダートの分布差を確実に吸収
- OOF EV予測のチェーン(AbilityModel OOF → WinTwoStage → EVCorrection → Isotonic)は時系列順を維持する必要がある
- バックテストの学習フェーズでIsotonicも毎回再学習されるため、テスト期間でのIsotonicは常にOOS評価

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 19-EV推定キャリブレーション*
*Context gathered: 2026-05-07*
