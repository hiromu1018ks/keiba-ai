# Phase 21: Conformal EV予測区間 - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

既存RobustConfidenceEstimatorをCQR（Conformal Quantized Regression）に置き換え、EV推定の不確実性を分布フリーに定量化する。信頼区間下界(EV_lower_90) < 1.0のベットを除外し、EV_excluded=0の問題を解消する。

**In scope:**
- CONF-01: CQRベースConformal Prediction EV区間実装。LightGBM quantile regression + CP補正
- CONF-02: EV信頼区間下界に基づく動的フィルタリング（EV_lower_90 < 1.0で除外）
- CONF-03: ConformalEVModelのパイプライン統合・診断レポート更新
- バリデーション分割ベースのCQR学習（K-fold不要）
- サーフェス別(芝/ダート)独立CQRモデル
- 既存EV診断へのCQRカバレッジ指標追加
- mockベースの自動テスト

**Out of scope:**
- 新しい特徴量の追加 (Phase 20完了済み)
- バックテストROI検証 (Phase 22)
- 複勝/ワイドモデルのCQR対応
- K-fold OOFベースのキャリブレーション（実行時間3時間+の問題あり）
- モデル再学習・ハイパーパラメータ再最適化

</domain>

<decisions>
## Implementation Decisions

### CQR手法設計 (CONF-01)
- **D-01:** 既存RobustConfidenceEstimatorを完全に置き換え、新規ConformalEVModelクラスを作成する。CQR（Conformal Quantized Regression）を採用。絶対値残差ベースの既存手法は異分散性を扱えず、EV_excluded=0の根本原因。
- **D-02:** LightGBM quantile regressionでα/2と1-α/2の2つの分位点モデルをサーフェス別(芝/ダート)に学習する。CQRの標準アプローチ。α=0.1（90%区間）の場合、quantile=0.05と0.95の2モデル。サーフェス別独立モデルで計4つのLightGBMモデル。
- **D-03:** CQRのターゲットはEV直接（ev_win_calibrated）。Phase 19のIsotonic適用後のEVを入力とする。CQR理論（Romano et al., 2019）でも予測対象そのものの区間推定が標準的。
- **D-04:** 2-alpha構成（80%+90%）を採用。90%区間でフィルタリング、80%区間でconfidence_score計算。既存のconformal_confidence_score出力との互換性を維持。

### 動的フィルタリング (CONF-02)
- **D-05:** フィルタリング閾値はEV_lower_90 < 1.0の固定閾値。JRA控除率25%後の損益分岐点。経済的解釈が明確。OOF分布でカバレッジ率を検証するが閾値自体は固定。
- **D-06:** ConformalEVModelはRobustConfidenceEstimatorと同じ出力列（EV_lower_win_corrected, EV_upper_win_corrected, conformal_confidence_score）を生成する。race_predictor.pyの既存フィルターロジックは変更不要。

### キャリブレーションデータ源
- **D-07:** TrainingPipelineV5の既存バリデーション分割（race_date後方20%）の予測結果からCQRを学習する。K-fold OOFは使わない（Phase 19でrun_backtest.pyが3時間+になった問題を回避）。追加推論1回のみで実行時間への影響は最小（数秒〜数十秒）。
- **D-08:** バリデーション分割での推論チェーン: AbilityModel → WinTwoStage → EVCorrection → Isotonic → ConformalEV。この推論結果とactual_returnからCQRの非適合スコアを計算し、CP補正量子を求める。

### パイプライン統合 (CONF-03)
- **D-09:** CQR学習を_train_submodel()の学習チェーン末尾に追加する。依存順序: Isotonic → ConformalEV。既存フローに統合。
- **D-10:** ConformalEVModelはSubmodelSetに追加する（Phase 19パターン踏襲）。ModelLoaderが自動読み込み。PFP改ざん検知対象。.lgb形式で保存。
- **D-11:** 既存EV診断（ev_diagnostics.py）を拡張してCQRカバレッジ率・区間幅の指標を追加する。Success Criteria（90%カバレッジ率）の自動検証。

### Claude's Discretion
- ConformalEVModelクラスの具体的なAPI設計（calibrate/predict_intervalメソッドのシグネチャ）
- LightGBM quantile regressionのハイパーパラメータ（既存モデルと同じかCQR用に調整）
- CQR非適合スコアの計算詳細（conformity score = max(q_low - y, y - q_high)の標準実装）
- サーフェス別CQRモデルのSubmodelSetフィールド命名規則
- バリデーション分割でのactual_return計算方法（win payout map参照）
- テストのfixtureデータとモック構成
- RobustConfidenceEstimatorの削除に伴う既存テストの移行

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### RobustConfidenceEstimator（置き換え対象 — CONF-01）
- `src/models/robust_confidence_estimator.py` — 既存CP実装。完全置き換え対象。出力列名とAPI互換性の参照元
- `tests/test_robust_confidence_estimator.py` — 既存テスト。移行参考

### RacePredictor（フィルター統合ポイント — CONF-02）
- `src/backtest/race_predictor.py:165-171` — predict_interval()呼び出し箇所。ConformalEVModelに差し替え
- `src/backtest/race_predictor.py:432-482` — get_win_candidates()。EV_lower_win_corrected < thresholdフィルター。変更不要（出力列名互換）

### BacktestEngine（除外統計 — CONF-02）
- `src/backtest/engine.py:77-81` — BacktestResultデータクラス。n_ev_excludedカウンター定義
- `src/backtest/engine.py:900-905` — n_ev_excluded集計。candidates.attrs["n_ev_excluded"]伝播

### TrainingPipelineV5（学習統合ポイント — CONF-03）
- `src/pipelines/training_pipeline.py:282` — _train_submodel()学習チェーン。Isotonic → ConformalEV追加箇所
- `src/pipelines/training_pipeline.py:934-1045` — _save_models_local()。ConformalEVモデル保存追加
- `src/db/model_loader.py:365-570` — _load_from_local()。ConformalEVモデル読み込み追加

### SubmodelSet（モデル保存先 — CONF-03/D-10）
- `src/domain/models.py:229` — SubmodelSet dataclass。ConformalEVフィールド追加ポイント

### EV診断（品質評価 — CONF-03/D-11）
- `src/models/ev_diagnostics.py` — compute_ev_diagnostics()。CQRカバレッジ指標追加先
- `src/models/ev_diagnostics.py:22-24` — 診断対象列定義

### 前フェーズのCONTEXT（必読 — 決定の連続性）
- `.planning/phases/19-ev-calibration/19-CONTEXT.md` — Phase 19決定（Isotonic、OOF生成、SubmodelSet統合）
- `.planning/phases/20-high-odds-pattern-features/20-CONTEXT.md` — Phase 20決定（高オッズ特徴量）

### 要件定義
- `.planning/REQUIREMENTS.md` — CONF-01, CONF-02, CONF-03の要件定義
- `.planning/ROADMAP.md` — Phase 21 Success Criteria

### ドメイン型
- `src/domain/types.py` — Surface, BetType等の型定義
- `src/domain/models.py` — SubmodelSet等のデータクラス

### 設定ファイル
- `config/settings.yaml` — database, paths設定

### 既存テストパターン
- `tests/test_backtest_engine.py` — BacktestEngine既存テスト
- `tests/test_ev_correction_model.py` — EVCorrectionModel既存テスト
- `tests/test_race_predictor.py` — RacePredictor既存テスト（conformal関連テスト含む）

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **RobustConfidenceEstimator** (`src/models/robust_confidence_estimator.py`): 出力列名（EV_lower_win_corrected等）とAPI（calibrate/predict_interval）の互換性参照元。完全置き換え後は削除
- **SubmodelSet** (`src/domain/models.py:229`): ConformalEVフィールド追加先。Phase 19のIsotonic統合パターンを踏襲
- **EV診断** (`src/models/ev_diagnostics.py`): ECE/Brier/Reliability計算が既存。CQRカバレッジ指標を拡張追加
- **TrainingPipelineV5** (`src/pipelines/training_pipeline.py`): バリデーション分割予測が既に生成されている。CQR学習の入力として再利用
- **LightGBM**: 既に依存関係に含まれる。quantile objective='quantile'で分位点回帰が可能。追加インストール不要

### Established Patterns
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用
- **コンストラクタ注入**: パラメータはコンストラクタ引数で注入
- **SubmodelSet保存/読み込み**: 新フィールド追加→ModelLoader両方更新（Phase 19パターン）
- **サーフェス別独立モデル**: 芝/ダートで別モデル。SubmodelSetのdict[str, SubmodelSet]構造
- **バリデーション分割**: race_date降順で前80%学習/後20%検証。時系列ソート必須
- **PFP（ParameterFreezeProtocol）**: JSON manifest + SHA256。ConformalEVも対象

### Integration Points
- **training_pipeline.py:_train_submodel()末尾** — Isotonic後にConformalEV学習追加
- **training_pipeline.py:_save_models_local()** — ConformalEV .lgb保存追加
- **db/model_loader.py:_load_from_local()** — ConformalEV読み込み追加
- **domain/models.py:SubmodelSet** — conformal_ev_modelフィールド追加
- **backtest/race_predictor.py:165-171** — predict_interval()呼び出しをConformalEVModelに差し替え
- **models/ev_diagnostics.py** — CQRカバレッジ指標追加

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- Phase 19のK-fold OOF生成はrun_backtest.pyの実行時間を3時間+にしたため、K-foldを使わない効率的なアプローチが必須
- バリデーション分割活用なら追加推論1回のみでCQR学習可能。実行時間への影響は数秒〜数十秒
- CQRの非適合スコアは標準的なconformity score = max(q_low - y, y - q_high)を使用
- LightGBMのquantile objectiveはalpha パラメータで分位点を指定。既存のbinary/regression設定とは別
- CQRのカバレッジ率検証はSuccess Criteria（90%以上）に直接対応。OOF検証で確認

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 21-Conformal EV予測区間*
*Context gathered: 2026-05-09*
