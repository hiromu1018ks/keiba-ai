# Phase 3: Selection Gate, Confidence & Betting - Context

**Gathered:** 2026-05-02
**Status:** Ready for planning

<domain>
## Phase Boundary

学習済みゲートで低信頼レースを除外し、JRA控除率25%を考慮した最適ベッティング戦略を統合する。

**In scope (from ROADMAP.md):**
- SELC-01: PlaceSelectionGateパターンを踏襲したWinSelectionGate実装（学習済みバイナリフィルター）
- SELC-02: Conformal predictionに基づく信頼性推定で低信頼度レースを除外
- BETT-01: JRA控除率25%を考慮したエッジ閾値設定・調整

**Out of scope:**
- Place/Wide SelectionGateの変更
- Walk-forward検証（Phase 4）
- Kelly基準の根本的見直し（BETT-02 fractional Kelly）
- レジーム適応型ベッティングの単勝パラメータ（BETT-03）
- アンサンブル手法（ENSB-01, ENSB-02）

</domain>

<decisions>
## Implementation Decisions

### WinSelectionGate設計
- **D-01:** PlaceSelectionGateの完全踏襲。OOF walk-forward score tables + smoothed scoring + add-second reranker + soft_pass_maskを全て再現。新クラスWinSelectionGateModelとして実装
- **D-02:** 入力変数はPlaceSelectionGateと同一構造（prob/edge/oddsの3次元binning）。Win特化入力の追加は行わない
- **D-03:** オッズソースはtanoddslow（最終単勝オッズ）。PlaceSelectionGateがfukuoddslowを使うのと同じパターン
- **D-04:** add-second rerankerを実装。ゲートがOOFデータから2頭目の有効性を学習するデータ駆動アプローチ
- **D-05:** 列名はwin_selection_prob, win_selection_edge, tanoddslow。PlaceSelectionGateのplace_selection_prob/place_selection_edge/fukuoddslowに対応

### Conformal信頼性推定
- **D-06:** 既存RobustConfidenceEstimatorを拡張。Win/Place両対応の既存コードを活用。CP quantileの精度向上（race-condition-dependent calibration）を行う
- **D-07:** EV下限値（EV_lower_win_corrected）をWinSelectionGateの入力edgeとして使用。3次元binning構造は維持
- **D-08:** 低信頼レースは閾値で完全除外（SELC-02要件）。WinSelectionGateのmin_prob/min_edge/max_oddsで足切り。賭け金調整ではなく除外

### JRA控除率とエッジ閾値
- **D-09:** エッジ計算はp差分方式。edge = p_model - p_market（p_market = 1/tanoddslow）。p_marketは既に控除率込みなので、p_modelとの差分が真のエッジを表す
- **D-10:** RegimeDetectorのedge_threshold設定にJRA控除率を反映。レジーム別に控除率を考慮した閾値を設定（現在の0.04/0.05/0.08から引き上げ）
- **D-11:** Kelly賭け金計算は既存の簡易Kelly(edge/(odds-1), cap=25%)を維持。WinSelectionGateがベット可否を判定し、Kellyは賭け金計算のみ担当

### Placeパターンからの逸脱
- **D-12:** WinSelectionGateModel新クラスを作成。PlaceSelectionGateModelとは独立。SubmodelSetにwin_selection_gateフィールドを追加
- **D-13:** 学習→保存→読み込みの3点更新パターンに従い、training_pipeline.py / model_loader.py / domain/models.py を更新
- **D-14:** RacePredictor.predict()でWin予測後のBenter適用後(line 124あたり)にWinSelectionGateを適用

### Claude's Discretion
- RegimeDetectorの具体的なedge_threshold値（控除率考慮後の最適値）
- RobustConfidenceEstimatorのrace-condition-dependent calibrationの詳細実装
- WinSelectionGateのsmoothed scoreのprior_weight等ハイパーパラメータ
- add-second rerankerの閾値グリッドの範囲・粒度

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### WinSelectionGate関連コード（踏襲パターン）
- `src/models/place_selection_gate.py` — PlaceSelectionGateModel完全実装(1044行)。WinSelectionGateModelの設計テンプレート。score tables, smoothed scoring, add-second reranker, soft_pass_maskの全実装
- `src/backtest/race_predictor.py` — 予測パイプライン。lines 192-202がPlaceSelectionGate適用部。WinSelectionGateの適用ポイント

### Conformal信頼性推定関連コード
- `src/models/robust_confidence_estimator.py` — 既存RobustConfidenceEstimator。Win CP quantile + Rolling Quantileのmin(Rule4)。拡張対象

### ベッティング・エッジ関連コード
- `src/betting/win_strategy.py` — WinStrategy。簡易Kelly計算(edge/(odds-1), cap=25%)
- `src/betting/gate_keeper.py` — GateKeeper。edge >= 0.03のデフォルト閾値。JRA控除率反映で引き上げが必要
- `src/betting/orchestrator.py` — BettingOrchestrator。RegimeDetectorからedge_threshold取得(line 155)
- `src/models/regime_detector.py` — RegimeDetector。3状態のedge_threshold設定(lines 183/211/224)

### ドメインモデル
- `src/domain/models.py` — SubmodelSet dataclass(line 229)。win_selection_gateフィールド追加対象
- `src/domain/models.py` — Bet dataclass(line 150)。edgeフィールド(line 159)

### 学習・保存・読み込み
- `src/pipelines/training_pipeline.py` — 学習パイプライン。PlaceSelectionGate学習を参考にWinSelectionGate学習を追加
- `src/db/model_loader.py` — モデル読み込み。win_selection_gateの読み込み追加が必要

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **PlaceSelectionGateModel** (`src/models/place_selection_gate.py`): 全1044行の完全実装。OOF walk-forward score tables, smoothed scoring, add-second reranker, soft_pass_maskをそのままWin向けに複製・修正可能
- **RobustConfidenceEstimator** (`src/models/robust_confidence_estimator.py`): Win/Place両対応の既存CP実装。calibrate()とpredict_lower_bound()のWinパスを拡張
- **ensure_place_selection_columns()** パターン: 列存在確認→フォールバック生成のパターン。Win版(ensure_win_selection_columns)が必要
- **build_place_selection_ev()** パターン: lower_ev → corrected_ev → direct_evのフォールバック連鎖。Win版が必要

### Established Patterns
- **SubmodelSet拡張パターン**: Optionalフィールド追加（`win_selection_gate: WinSelectionGateModel | None = None`）の既存パターンに従う
- **学習→保存→読み込みの3点更新**: 新モデル追加時は training_pipeline.py / model_loader.py / domain/models.py の3ファイル更新が必須
- **RacePredictor適用パターン**: predict()内でsubmodelからcomponentを取得→適用→結果をDataFrameに書き戻す
- **GateKeeper + RegimeDetector連動**: OrchestratorがRegimeDetectorからedge_thresholdを取得→GateKeeperに渡す

### Integration Points
- **TrainingPipelineV5._train_submodel()**: WinSelectionGate学習の挿入ポイント。PlaceSelectionGate学習の後にWin版を追加
- **RacePredictor.predict()**: Win Benter適用後(line 124)にWinSelectionGateを適用
- **RegimeDetector.get_strategy_params()**: edge_thresholdの値をJRA控除率考慮に更新
- **BacktestEngine.run()**: バックテスト時のtanoddslow列アクセス。既にDataFrame内に存在
- **SubmodelSet**: win_selection_gateフィールドの追加。domain/models.py:229のdataclass

</code_context>

<specifics>
## Specific Ideas

- ユーザーは「ベストプラクティスを追求」という方針。品質優先で実装する
- PlaceSelectionGateの完全踏襲により、複雑なスコアリング機構の再設計リスクを回避する
- JRA控除率25%はedge計算のp差分方式(p_model - p_market)で暗黙的に考慮される

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 3-Selection Gate, Confidence & Betting*
*Context gathered: 2026-05-02*
