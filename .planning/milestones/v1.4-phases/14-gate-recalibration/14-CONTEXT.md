# Phase 14: Gate Recalibration - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning

<domain>
## Phase Boundary

WinSelectionGateがアンサンブルOOF予測分布に適合し、単一モデル/アンサンブル間の確率分布ドリフトが定量的に診断可能になり、`use_ensemble`フラグがModelLoader→RacePredictor→BacktestEngine→WinSelectionGate全経路で正しく伝播されていることがテストで検証されている状態になる。

**In scope:**
- WinSelectionGateのアンサンブルOOF再学習検証 (GATE-01)
- ks_2samp/wasserstein_distance分布ドリフト診断機能の追加 (GATE-02)
- use_ensembleフラグ伝播経路のテストによる検証 (GATE-03)

**Out of scope:**
- EV_lower閾値の動的化 (Phase 15)
- OddsBandFilterの再キャリブレーション (Phase 16)
- Optuna最適化 (Phase 17)
- 新しいモデルや特徴量の追加
- RegimeDetectorの調整

</domain>

<decisions>
## Implementation Decisions

### 分布ドリフト診断の設計
- **D-01:** 診断機能をバックテストパイプラインに統合する。`run_backtest.py --ensemble`実行時に自動で分布ドリフト診断が実行される。独立スクリプトは作成しない。
- **D-02:** 診断結果はJSONファイル + コンソールサマリで出力する。JSONは`data/backtest/`に保存し、コンソールにはKS統計量/p-value/Wasserstein距離の要約を表示する。
- **D-03:** 分布比較の粒度は最大限に: (1) 主要確率・EV列(p_win_final, edge_win, EV_lower_win_corrected, win_selection_prob等)の全データ比較、(2) サーフェス別(芝/ダート)の分割比較、(3) 年度別時系列でのドリフト推移追跡。全てks_2sampとwasserstein_distanceの両方を使用。
- **D-04:** ドリフト検出時の対応: KS p-value < 0.05 または Wasserstein距離が閾値超過の場合にWARNINGログで再学習を推奨。バックテストは継続するが、JSON結果に`drift_detected: true`フラグと推奨アクションを含める。

### use_ensemble伝播テスト戦略
- **D-05:** フラグ経路のモックベーステストを採用する。値レベルのアサーションではなく、各コンポーネントにuse_ensemble=Trueが正しく渡ることをモックで検証する。
- **D-06:** 統合テスト1つでModelLoader→TrainingPipeline→RacePredictor→WinSelectionGateの全体経路を検証する。コンポーネント別個別テストではなく、1つのテストクラスでend-to-endのフラグ伝播を確認する。
- **D-07:** use_ensemble=Trueの経路のみテストする。False(デフォルト)の経路は既存テストがカバーしている前提。

### ゲート再学習検証方法
- **D-08:** ゲート再学習検証は二段構え: (1) ユニットテストでfixtureデータを使い、単一モデルOOFとアンサンブルOOFで学習したゲートのprob_edges/edge_edges/odds_edgesが異なることを確定的に検証、(2) パイプラインのランタイムで、use_ensemble=True時に学習後のゲートedgesがデフォルト(未学習)のedgesと異なることをassertionで確認。

### Claude's Discretion
- 診断機能の具体的な閾値(Wasserstein距離のwarn/error閾値)は研究者・プランナーがデータから決定してよい
- JSON出力のスキーマ設計はプランナーに委ねる
- テストのfixtureデータの内容はプランナーに委ねる

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### ゲートモデル
- `src/models/win_selection_gate.py` — WinSelectionGateModel実装。train/score/save/loadメソッド。quantile bin edgesとscore tableの構造
- `src/models/win_selection_gate.py:804-878` — train()メソッド。OOF DataFrameを受け取ってprob_edges/edge_edges/odds_edgesとscore tablesを構築

### パイプライン統合
- `src/pipelines/training_pipeline.py:283-813` — _train_submodel()。use_ensembleフラグ受け取り→StackedEnsemble生成→OOF予測→WinSelectionGate.train()の全経路
- `src/pipelines/training_pipeline.py:784-792` — WinSelectionGate学習箇所。df_oofからwsg_train_dfを構築してgate.train()を呼ぶ
- `src/backtest/race_predictor.py:130-139` — RacePredictorでのWinSelectionGate使用箇所

### モデルローダー
- `src/db/model_loader.py:37-47` — ModelLoader.load()。use_ensemble_override引数
- `src/db/model_loader.py:446-472` — _load_hit_model()。use_ensemble=Trueで.joblib(StackedEnsemble)をロード
- `src/db/model_loader.py:474-570` — load_from_dir()。meta.jsonのuse_ensembleとjoblib整合性チェック

### バックテストエントリ
- `scripts/run_backtest.py:86` — --ensembleフラグ定義
- `scripts/run_backtest.py:455` — pipeline.run(use_ensemble=args.ensemble)呼び出し

### 既存テストパターン
- `tests/test_win_selection_gate.py` — WinSelectionGateModelの既存テスト。mockベース

### 研究
- `.planning/research/SUMMARY.md` — Phase 14の研究サマリ。ゲート再学習は~20行のデータルーティング変更

### 要件
- `.planning/REQUIREMENTS.md` — GATE-01, GATE-02, GATE-03の要件定義

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `WinSelectionGateModel.train()`: モデル非依存。DataFrameを受け取ってquantile bin edgesとscore tablesを構築。アンサンブルOOFもそのまま処理可能
- `_quantile_edges()`: 確率・エッジ・オッズのquantile区間計算。既存ユーティリティ
- `_build_score_tables()`: combo_scores/pair_scores/single_scoresの構築。Bayesian smoothing付き
- `scipy.stats.ks_2samp`, `scipy.stats.wasserstein_distance`: 既にインストール済み(via scikit-learn transitive)

### Established Patterns
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。Phase 14テストもこのパターンに従う
- **TrainingPipelineV5._train_submodel()**: use_ensemble=True時にStackedEnsembleを生成し、OOFを通じて下流コンポーネントに伝播
- **バックテストパイプライン**: run_backtest.py --ensemble → TrainingPipelineV5.run(use_ensemble=True) → 全モデル再学習

### Integration Points
- `TrainingPipelineV5._train_submodel()` 内のゲート学習セクション(784-792行目)に診断機能を統合
- `run_backtest.py` の`--ensemble`パスに診断出力ロジックを追加
- テストファイル: `tests/test_ensemble_gate_propagation.py` (新規)

</code_context>

<specifics>
## Specific Ideas

- 診断のベストプラクティスを追求: 単なる比較ではなく、サーフェス別・年度別の多角的なドリフト分析を実装する
- 分布ドリフト診断はバックテスト結果と同じ場所(`data/backtest/`)にJSON保存
- ユーザーは「ベストプラクティス追求」を一貫して選択しており、実装難易度よりも品質・網羅性を優先する意向がある

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-Gate Recalibration*
*Context gathered: 2026-05-06*
