# Phase 12: Stake Sizing Enhancement - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

レジーム状態に応じたKelly分数とEV比例乗算器により、高確信ベットに重点配分された賭け金が算出される。

**In scope (from ROADMAP.md):**
- SIZE-01: レジーム状態別にKelly分数が異なり、AGGRESSIVE > CONSERVATIVE > COLLAPSED(=0)の順で賭け金が計算される
- SIZE-02: 高EVベットの賭け金にEV比例乗算器(min(ev/target_ev, max_scale))が適用され、同一レジーム内でEVが高いほど賭け金が大きくなる
- 成功基準3: フィルター+サイジング変更後のバックテストROIがベースライン(89.0%)を上回る

**Out of scope:**
- DD制御パラメータの再調整 (Phase 13: Risk Calibration & Parameter Optimization)
- パラメータ凍結・Optuna最適化 (Phase 13)
- 複勝/ワイドモデルの変更
- BettingOrchestrator/WinStrategy/PlaceStrategy (ライブパス) の変更
- 新データ源の導入
- モデル再学習

**Plans:** TBD

</domain>

<decisions>
## Implementation Decisions

### レジーム別Kelly分数 (SIZE-01)
- **D-01:** fractional_kelly をレジーム別に設定: AGGRESSIVE=0.50 (half-Kelly), CONSERVATIVE=0.25 (quarter-Kelly), COLLAPSED=0.00 (no bet)。金融ベッティングの標準プラクティスに準拠
- **D-02:** KELLY_FRACTION_CAP=0.25 は固定。実効cap = 0.25 × fractional_kelly で自然にレジーム別調整（AGGRESSIVE: 0.125, CONSERVATIVE: 0.0625）
- **D-03:** MIN_STAKE=100円, MAX_STAKE=10,000円は現在値を維持。JRA運用制約として固定
- **D-04:** RACE_EXPOSURE_CAP=0.02 (2%) は全レジーム共通で固定。破滅リスク防止のセーフティネットとしてレジームに依存しない

### EV比例乗算器 (SIZE-02)
- **D-05:** target_ev=1.10, max_scale=2.0 で固定（レジーム非依存）。AGGRESSIVEのev_threshold=1.10と同一値で一貫性あり
- **D-06:** 公式: `scale = min(ev / target_ev, max_scale)`、`stake = kelly_stake * scale`。EV < target_ev の場合は scale < 1.0 となり縮小効果（低EVベットの賭け金抑制）
- **D-07:** EV乗算器は StakeCalculator に `apply_ev_scaling()` メソッドとして追加。単一責務でテスト容易

### サイジングパイプライン適用順序
- **D-08:** Kelly → EV乗算 → DD の順序。DDを最終リスクゲートとすることで、EV拡大がDD制御をバイパスしない。リスク管理の標準パターン
- **D-09:** パイプライン全体: `kelly_stake = calc_stake(edge, odds, bankroll)` → `ev_scaled = apply_ev_scaling(kelly_stake, ev)` → `final_stake = dd_ctrl.adjust_stake(ev_scaled, bankroll)`

### パラメータ注入方法 (Phase 13 Optuna前提)
- **D-10:** コンストラクタ注入パターン。StakeCalculator.__init__() で fractional_kelly, target_ev, max_scale を引数受取。デフォルト値は settings.yaml の betting_strategy section から読み込み
- **D-11:** RegimeDetector.get_strategy_params() と MetaSwitcher._default_params() に fractional_kelly を追加。ev_threshold, edge_threshold と同じパターンでregime params dictに含める
- **D-12:** Phase 13 Optuna最適化ではコンストラクタ引数で直接注入。設定ファイル経由ではなくプログラマティックに制御

### スコープ（Phase 11決定の踏襲）
- **D-13:** 変更対象はバックテストパスのみ: StakeCalculator + RacePredictor + RegimeDetector/MetaSwitcher。BettingOrchestrator, WinStrategy, PlaceStrategy（ライブパス）は変更しない。Phase 11の決定「BettingOrchestrator非対象」を踏襲

### Claude's Discretion
- StakeCalculator.calc_stake() の具体的なリファクタリング（ハードコード定数 → インスタンス変数化）
- apply_ev_scaling() のシグネチャと返り値の型
- settings.yaml の betting_strategy section のスキーマ設計
- RegimeDetector.get_strategy_params() への fractional_kelly 追加方法
- テスト戦略（StakeCalculator単体テスト + RacePredictor統合テスト）
- EV値の取得元（RacePredictor.select_bets() の DataFrameカラム）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### StakeCalculator（主変更対象）
- `src/betting/stake_calculator.py` lines 25-30 — ハードコード定数群（FRACTIONAL_KELLY等）。インスタンス変数化の対象
- `src/betting/stake_calculator.py` lines 32-77 — calc_stake()。Kelly計算。fractional_kellyをインスタンス変数から取得するよう変更
- `src/betting/stake_calculator.py` lines 79-122 — check_race_exposure()。2% cap。変更なし（共通固定）

### RegimeDetector / MetaSwitcher（パラメータ追加）
- `src/models/regime_detector.py` lines 185-240 — get_strategy_params()。fractional_kellyの追加ポイント
- `src/betting/meta_switcher.py` — _default_params()。同上

### RacePredictor（サイジングパイプライン統合）
- `src/backtest/race_predictor.py` lines 596-753 — select_bets()。Kelly→EV乗算→DDパイプラインの挿入ポイント
- `src/backtest/race_predictor.py` lines 648-680 — win bet stake計算。EV乗算の追加ポイント
- `src/backtest/race_predictor.py` lines 740-770 — place bet stake計算（参照のみ、変更なし）

### BacktestEngine（regime params利用）
- `src/backtest/engine.py` lines 706-710 — regime検出 + strategy_params取得。fractional_kellyをStakeCalculatorに渡すポイント

### ドメイン型
- `src/domain/types.py` lines 21-26 — DrawdownState enum (NORMAL, REDUCED, RECOVERING)
- `src/domain/types.py` lines 29-34 — RegimeState enum (AGGRESSIVE, CONSERVATIVE, COLLAPSED)

### 設定ファイル
- `config/settings.yaml` — betting_strategy section の追加ポイント

### 要件定義
- `.planning/REQUIREMENTS.md` — SIZE-01, SIZE-02
- `.planning/ROADMAP.md` — Phase 12 Success Criteria
- `.planning/STATE.md` — 決定済みアーキテクチャ

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **StakeCalculator** (`src/betting/stake_calculator.py`): 既存のKelly計算ロジック。calc_stake() の基本構造（edge/odds/bankroll → stake）はそのまま活用。ハードコード定数 → コンストラクタ引数にリファクタリング
- **RegimeDetector.get_strategy_params()** (`src/models/regime_detector.py`): 既にev_threshold, edge_threshold等をレジーム別に返している。fractional_kellyの追加は同じパターン
- **RacePredictor.select_bets()** (`src/backtest/race_predictor.py`): バックテストパスでのステーク計算。Kelly→DDの既存フローにEV乗算を挿入
- **DrawdownController** (`src/betting/drawdown_controller.py`): adjust_stake() メソッド。最終リスクゲートとしてそのまま利用（変更なし）

### Established Patterns
- **regime params活用パターン**: get_strategy_params() → params["ev_threshold"] 等を候補選択で使用。fractional_kellyも同じパターン
- **コンストラクタ注入パターン**: 現在のStakeCalculatorは引数なしコンストラクタだが、他のモデルクラス（RegimeDetector等）はコンストラクタでパラメータを受け取る
- **除外統計ログパターン**: engine.py のレースループでカウンタをインクリメント → run() 終了時にINFO ログ出力

### Integration Points
- **stake_calculator.py:25-30** — ハードコード定数 → コンストラクタ引数化
- **stake_calculator.py** — apply_ev_scaling() メソッド追加（新規）
- **regime_detector.py:185-240** — get_strategy_params() に fractional_kelly 追加
- **meta_switcher.py** — _default_params() に fractional_kelly 追加
- **race_predictor.py:648-680** — select_bets() のwin stake計算にEV乗算追加
- **engine.py:706-710** — regime_paramsからfractional_kellyをStakeCalculatorに渡す
- **config/settings.yaml** — betting_strategy section 追加（デフォルト値定義）

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- 全技術的判断にベストプラクティスを適用。ユーザーは専門的詳細より結果（ROI改善）を重視
- Kelly → EV乗算 → DDの順序は、DDを最終ゲートとするリスク管理の標準パターン
- コンストラクタ注入は Phase 13 Optuna最適化の前提。ハードコード定数では最適化不可
- バックテストパスのみ変更（ライブパスはPhase 11決定を踏襲）

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-Stake Sizing Enhancement*
*Context gathered: 2026-05-05*
