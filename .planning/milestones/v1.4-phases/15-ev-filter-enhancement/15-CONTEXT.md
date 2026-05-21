# Phase 15: EV Filter Enhancement - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning

<domain>
## Phase Boundary

EV_lower閾値がアンサンブルOOF分布に基づく動的閾値に置き換わり、過剰除外が解消されるとともにEV推定精度が可視化されている状態になる。

**In scope:**
- EVF-01: EV_lower閾値を固定1.0からアンサンブルOOF分布の分位点に基づく動的閾値に変更する
- EVF-02: OOF EV推定値と実際の払戻額を比較し、EV推定精度を評価する診断機能を追加する
- RobustConfidenceEstimatorのアンサンブル残差での再キャリブレーション (EVF-01の前提作業)
- EV診断モジュールの新規作成 (ECE + Brier score + Reliability diagram + 時系列ドリフト追跡)

**Out of scope:**
- OddsBandFilterの再キャリブレーション (Phase 16)
- Optuna最適化の実行 (Phase 17)
- 複勝/ワイドモデルの変更
- 新しいモデルや特徴量の追加
- RegimeDetectorの調整
- WinSelectionGateの再学習 (Phase 14完了済み)

</domain>

<decisions>
## Implementation Decisions

### 動的閾値の計算方式 (EVF-01)
- **D-01:** 複合方式を採用 — Percentile方式(25th percentile of positive-edge OOF winners)を初期値とし、Phase 17 Optunaの14次元探索に閾値を15次元目として追加して最適化する。手動閾値チューニングのバイアスを排除し、データ駆動の最適化を行う。
- **D-02:** 閾値はSurface別(芝/ダート)で各OOF winners分布から独立に計算する。Optunaの探索次元は16次元(14 + サーフェス別2閾値)。Phase 14 D-03のサーフェス別分析パターンを踏襲。
- **D-03:** EV_lowerがNaNの場合、サーフェス別のデフォルト閾値にフォールバックする。Conformal未学習レースでも一貫した判定動作を確保。Phase 11 D-03のfillna(1.0)から変更。

### EV推定精度診断 (EVF-02)
- **D-04:** 深度診断(学術的)を実装: EV予測vs実際払戻の相関/RMSE + Reliability diagram(ECE: Expected Calibration Error) + Brier score分解 + 時系列ドリフト追跡。Phase 14のJSON+コンソール多粒度パターンを拡張。
- **D-05:** パイプライン統合 — run_backtest.py --ensemble実行時に自動でEV診断が実行される。独立スクリプトは作成しない。Phase 14のdrift_diagnostics.pyパターンに準拠。

### Conformal再キャリブレーション (EVF-01前提)
- **D-06:** RobustConfidenceEstimatorをアンサンブルOOF残差で再calibrateする。現在の単一LightGBM残差キャリブレーションでは、アンサンブルの実際の誤差分布に対して過大な区間幅を出力し、EV_lowerが不当に低く算出される。アンサンブル残差で再キャリブレーションすることで区間幅が適正化され、動的閾値の効果を最大化する。~20行のpipeline data routing変更で対応可能。

### Claude's Discretion
- EV診断モジュールの具体的なJSONスキーマ設計
- Percentile計算の実装詳細(どのOOFサブセットを正のエッジ勝利馬とするか)
- Brier score分解の実装方法(reliability/uncertainty/resolution)
- Reliability diagramのビン数と表示形式
- 時系列ドリフト追跡の粒度(年度別/四半期別)
- サーフェス別フォールバック閾値の具体的な計算方法
- テスト戦略(モックベース、既存パターン踏襲)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### EV_lowerフィルター実装 (主変更対象)
- `src/backtest/race_predictor.py` lines 420-480 — get_win_candidates()。EV_lower >= 1.0ハードフィルター(D-01)の追加ポイント。動的閾値に置き換える
- `src/backtest/race_predictor.py` lines 434-451 — 現在のEV_lowerフィルター。fillna(1.0)フォールバック(D-03)をサーフェス別デフォルトに変更

### Conformal信頼区間 (EV_lowerのソース — 再キャリブレーション対象)
- `src/models/robust_confidence_estimator.py` lines 96-234 — predict_interval()。EV_lower_win_correctedの計算。CP quantile + rolling quantile
- `src/models/robust_confidence_estimator.py` lines 96-127 — calibrate()または非calibrate時のフォールバック処理。use_ensemble時のcalibrationフロー

### パイプライン統合 (診断実行ポイント)
- `src/pipelines/training_pipeline.py` lines 283-813 — _train_submodel()。use_ensemble=True時のアンサンブルOOF生成→Conformal学習→診断の全経路
- `src/pipelines/training_pipeline.py` lines 792-812 — Phase 14ドリフト診断統合箇所。同じパターンでEV診断を追加
- `scripts/run_backtest.py` lines 86, 455 — --ensembleフラグ定義とpipeline.run()呼び出し

### Phase 14ドリフト診断 (パターン参照)
- `src/models/drift_diagnostics.py` — compute_drift_diagnostics()。JSON + コンソール出力パターン。EV診断モジュールの設計テンプレート

### バックテストエンジン (除外統計ログ)
- `src/backtest/engine.py` lines 603-820 — run() レースループ。除外件数カウンタのインクリメントパターン

### レポート
- `src/backtest/report.py` lines 17-99 — BacktestReportGenerator。除外統計表示
- `src/backtest/report.py` lines 155-248 — _compute_condition_stats()。オッズバンド分析(1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+)

### ドメイン型
- `src/domain/types.py` lines 29-34 — RegimeState enum (AGGRESSIVE, CONSERVATIVE, COLLAPSED)

### 既存テストパターン
- `tests/test_drift_diagnostics.py` — Phase 14ドリフト診断テスト(8テスト)。mockベース。EV診断テストの参照実装
- `tests/test_win_selection_gate.py` — WinSelectionGateModelテスト

### 研究
- `.planning/research/SUMMARY.md` — Phase 15研究サマリ。Percentile方式推奨、~30行+~10行の実装見積

### 要件
- `.planning/REQUIREMENTS.md` — EVF-01, EVF-02の要件定義

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **RobustConfidenceEstimator.predict_interval()** (`robust_confidence_estimator.py:96-234`): モデル非依存。win_df/place_dfを受け取ってEV_lower/EV_upperを計算。calibrate()メソッドで残差分布からCP quantileを学習。アンサンブル残差で再calibrate可能
- **drift_diagnostics.py** (`models/drift_diagnostics.py`): Phase 14で作成。compute_drift_diagnostics() + console_summary() + DRIFT_COLUMNS。JSON出力 + コンソールサマリのパターン。EV診断モジュールのベースとして再利用
- **get_win_candidates()** (`race_predictor.py:420-480`): 既存の候補選択ロジック。EV_lower >= 1.0固定閾値を動的閾値に置き換えるだけで対応
- **_compute_condition_stats()** (`report.py:155-248`): pd.cut() でオッズバン化 → groupby → ROI計算。EV診断のオッズバンド別分析に同じパターンを適用

### Established Patterns
- **パイプライン統合診断パターン**: use_ensemble=True時に_training_submodel()内で診断モジュール呼び出し → JSON + コンソール出力。Phase 14で確立
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。Phase 15テストもこのパターンに従う
- **コンストラクタ注入パターン**: パラメータはコンストラクタ引数で注入。Phase 12 D-10で確立
- **Surface別分析パターン**: df[df["surface"]=="芝"] で分割 → 各分布独立分析。Phase 14 D-03で確立

### Integration Points
- **race_predictor.py:434-440** — EV_lower固定閾値 → 動的閾値に変更
- **robust_confidence_estimator.py** — calibrate()呼び出し箇所でアンサンブル残差を使用するよう変更
- **training_pipeline.py:792-812** — Phase 14ドリフト診断と同じ位置にEV診断を追加
- **新規ファイル**: `src/models/ev_diagnostics.py` (EV推定精度診断モジュール)

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- Conformal再キャリブレーション + 動的閾値の二段階構成で3,594件除外を根本解決する
- EV診断は学術的精度評価(ECE, Brier score分解, Reliability diagram)を含む深度診断
- Phase 14パターンの踏襲により、実装効率が高い(インフラ再利用)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 15-EV Filter Enhancement*
*Context gathered: 2026-05-06*
