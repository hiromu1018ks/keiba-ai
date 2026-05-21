# Phase 16: Odds Band Rebuild - Context

**Gathered:** 2026-05-06
**Status:** Ready for planning

<domain>
## Phase Boundary

strategy_optimizer.pyのルックアヘッドバイアスが修正され、アンサンブルモデルで生成されたtraining_bet_historyに基づいてOddsBandFilterが正しく再キャリブレーションされている状態になる。

**In scope:**
- ODDS-01: アンサンブルモデルでtraining_bet_historyを再生成し、OddsBandFilter.calibrate()でバンド別ROIを再計算する
- ODDS-02: strategy_optimizer.pyのルックアヘッドバイアスを修正し、training_bet_history生成にデフォルトパラメータを使用する
- BacktestEngine.run()内でのtraining_bet_history自動生成（パイプライン統合）
- ルックアヘッドバイアス修正のテスト検証

**Out of scope:**
- Optuna最適化の実行 (Phase 17)
- バリデーション & パラメータ凍結 (Phase 18)
- 複勝/ワイドモデルの変更
- 新しいモデルや特徴量の追加
- OddsBandFilterのバンド境界変更 (固定境界を維持)

</domain>

<decisions>
## Implementation Decisions

### ルックアヘッドバイアス修正 (ODDS-02)
- **D-01:** training_bet_history生成に使用するデフォルトパラメータのソースはRegimeDetector.get_strategy_params()のハードコード既定値とする。これが「最適化前」の値のSingle Source of Truthであり、Phase 13 D-15で外部化済みのため最も一貫性がある。
- **D-02:** デフォルトパラメータは16次元全てを適用する。Kelly分数・EV閾値・DD制御・EVスケーリング・OddsBandFilter ROI閾値の全て。ベット生成自体がこれら全パラメータの影響を受けるため、部分適用では不十分なルックアヘッド防止になる。
- **D-03:** strategy_optimizer.pyに_build_default_config()メソッドを追加し、RegimeDetector既定値からstrategy_config dictを構築する。既存の_build_strategy_config()と並存。懸念事項の分離が明確でテストも容易。
- **D-04:** _run_single_backtest()のステップ3（training-phase backtest）を_build_default_config()の出力を使用するように変更。ステップ4-5（test-phase backtest）は引き続きOptuna提案のstrategy_configを使用。

### パイプライン統合 (ODDS-01)
- **D-05:** run_backtest.py --ensemble実行時にBacktestEngine.run()内で自動的にtraining_bet_historyを生成し、OddsBandFilterをキャリブレーションする。Phase 14-15のパイプライン統合パターンに準拠。
- **D-06:** training_bet_historyの生成はBacktestEngine.run()内で完結させる。test_startの前にtrain_start～train_endのバックテストを実行し、bet_historyを生成。engine.py内部で完結するためスクリプト側の変更は最小限。
- **D-07:** training_bet_history生成時はデフォルトパラメータを使用（strategy_optimizer.pyと同じルール）。BacktestEngineにuse_ensemble=Trueが渡っている場合、モデル情報のtrain_start/train_endからトレーニング期間を特定して自動実行。

### バンド境界の取り扱い
- **D-08:** オッズバンド境界は現状の固定値 `[1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+]` を維持する。ROI計算のみアンサンブルデータで更新。境界変更はOptuna探索空間を増大させ適合リスクが高まるため、Phase 17 Optunaのroi_threshold (1次元)で間接的に調整する。

### テスト検証
- **D-09:** ルックアヘッドバイアス修正は二段階検証: (1) モックでtraining backtestがデフォルトパラメータで実行されたことを確認 + trainingとtestで異なるstrategy_configが使用されたことを検証、(2) デフォルトパラメータで生成されたtraining_bet_historyがOptuna最適化後と異なるOddsBandFilter除外バンドを生成することを確認。

### Claude's Discretion
- _build_default_config()の具体的な実装（どのRegimeDetectorメソッドから値を取得するか）
- BacktestEngine.run()内でのtraining_bet_history自動生成の具体的なロジック
- テストのfixtureデータの内容
- デフォルトパラメータでDDConfigを構築する際の具体的な値

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### OddsBandFilter（主変更対象 — ODDS-01）
- `src/betting/odds_band_filter.py` — OddsBandFilter実装。calibrate()がbet_historyからバンド別ROI計算・除外バンド決定。filter()が候補DataFrameを除外
- `src/betting/odds_band_filter.py:38-78` — calibrate()メソッド。bet_history itemsのキー要件: "odds", "result", "stake"

### strategy_optimizer.py（ルックアヘッド修正対象 — ODDS-02）
- `src/tuning/strategy_optimizer.py:118-192` — _run_single_backtest()。ステップ3(150-168行目)がtraining_bet_history生成箇所。strategy_configをデフォルトに変更
- `src/tuning/strategy_optimizer.py:83-116` — _build_strategy_config()。Optuna params → BacktestEngine injection dict変換
- `src/tuning/strategy_optimizer.py:51-81` — _suggest_params()。16次元探索空間定義

### BacktestEngine（パイプライン統合ポイント — ODDS-01）
- `src/backtest/engine.py:414-429` — run()メソッド。training_bet_historyパラメータ受け取り
- `src/backtest/engine.py:661-663` — OddsBandFilter.calibrate()呼び出し箇所
- `src/backtest/engine.py:382-385` — OddsBandFilterインスタンス生成。roi_thresholdはstrategy_paramsから取得
- `src/backtest/engine.py:798-805` — OddsBandFilter.filter()による候補除外

### RegimeDetector（デフォルトパラメータソース — ODDS-02）
- `src/models/regime_detector.py:185-243` — get_strategy_params()。25+ハードコードパラメータ。Phase 13 D-15で外部化済み
- `src/models/regime_detector.py:133-176` — detect()。3状態分類

### ドメイン型
- `src/domain/types.py:29-34` — RegimeState enum (AGGRESSIVE, CONSERVATIVE, COLLAPSED)

### 設定ファイル
- `config/settings.yaml` lines 39-47 — betting_strategy section。デフォルト値定義

### 既存テストパターン
- `tests/test_odds_band_filter.py` — OddsBandFilter既存テスト。mockベース
- `tests/test_strategy_optimizer.py` — StrategyOptimizer既存テスト。mockベース

### 前フェーズのCONTEXT（必読 — 決定の連続性）
- `.planning/phases/15-ev-filter-enhancement/15-CONTEXT.md` — Phase 15決定（EV_lower動的閾値、Conformal再キャリブレーション、パイプライン統合パターン）
- `.planning/phases/14-gate-recalibration/14-CONTEXT.md` — Phase 14決定（ドリフト診断パイプライン統合パターン、use_ensemble伝播）

### 要件定義
- `.planning/REQUIREMENTS.md` — ODDS-01, ODDS-02の要件定義
- `.planning/ROADMAP.md` — Phase 16 Success Criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **OddsBandFilter** (`src/betting/odds_band_filter.py`): calibrate()はモデル非依存。bet_history(list[dict])を受け取ってバンド別ROI計算。アンサンブル由来のbet_historyもそのまま処理可能
- **BacktestEngine.run()** (`src/backtest/engine.py:414-429`): training_bet_historyパラメータを既に受け取る。OddsBandFilter.calibrate()へのデータ渡しも実装済み(661-663行目)
- **RegimeDetector.get_strategy_params()** (`src/models/regime_detector.py:185-243`): デフォルトパラメータのソース。Phase 13 D-15で主要パラメータをコンストラクタ注入に外部化済み
- **_build_strategy_config()** (`src/tuning/strategy_optimizer.py:83-116`): Optuna params → BacktestEngine injection dictの変換ロジック。_build_default_config()の設計テンプレート

### Established Patterns
- **パイプライン統合診断パターン**: use_ensemble=True時に自動で診断・キャリブレーション実行。Phase 14-15で確立
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。Phase 16テストもこのパターンに従う
- **コンストラクタ注入パターン**: パラメータはコンストラクタ引数で注入。Phase 12 D-10、Phase 13 D-17で確立
- **DDConfig dataclass** (`src/betting/drawdown_controller.py`): _build_default_config()内でもDDConfigインスタンスを構築する必要あり

### Integration Points
- **strategy_optimizer.py:150-168** — training_bet_history生成箇所。strategy_config → _build_default_config()の出力に変更
- **backtest/engine.py:661-663** — OddsBandFilter.calibrate()呼び出し。training_bet_history自動生成ロジックを追加
- **src/tuning/strategy_optimizer.py** — _build_default_config()メソッド新規追加
- **strategy_optimizer.pyの新規メソッド**: `_build_default_config()` — RegimeDetector既定値からstrategy_config dictを構築

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- ルックアヘッドバイアス修正は16次元全てのデフォルトパラメータ適用が最も厳密
- パイプライン統合はPhase 14-15の確立パターンを踏襲することで実装効率が高い
- バンド境界は固定維持でPhase 17 Optunaのroi_thresholdで間接調整

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 16-Odds Band Rebuild*
*Context gathered: 2026-05-06*
