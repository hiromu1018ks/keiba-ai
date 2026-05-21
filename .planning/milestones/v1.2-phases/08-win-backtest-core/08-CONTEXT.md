# Phase 8: Win Backtest Core - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

ユーザーが単勝モードのバックテストを実行し、正しい単勝ROI・的中率・バンクロール推移を得られるようにする。

**In scope (from ROADMAP.md):**
- WIN-01: build_win_payout_map()で単勝払戻しデータ(paytansyoumaban1/paytansyopay1)からpayout_mapを構築
- WIN-02: final_odds_mapがtanoddslow(単勝オッズ)を使用し、単勝ベットの正しい決済を行う
- WIN-03: get_win_candidates()がwin_selection_ev/edge/prob列で候補をフィルタリング
- WIN-04: BacktestEngineにbetting_targetパラメータを追加し、単勝/複勝モードを切り替え(デフォルト=WIN)
- WIN-05: Conformal信頼性スコアを単勝ベット判定に組み込み、高信頼度ベットのみを生成

**Out of scope:**
- 単勝レポート・分析 (Phase 9: Win Reporting)
- パフォーマンス最適化 (Phase 10: Pipeline Performance)
- 複勝/ワイドモデルの変更
- ベッティング戦略の変更(Kelly基準・RegimeDetector精緻化はv1.3以降)
- 新データ源の導入

**Plans:** 2 plans
- 08-01: Win payout map + final odds map + betting_target dispatch (WIN-01, WIN-02, WIN-04)
- 08-02: Win candidate selection + Conformal confidence integration + WF validation (WIN-03, WIN-05)

</domain>

<decisions>
## Implementation Decisions

### 単勝決済精度 (WIN-01, WIN-02)
- **D-01:** 決済は実際の払戻金(`paytansyopay1/100`)を使用。JRA公式払戻金で最も正確
- **D-02:** `build_win_payout_map()`を新規追加。`(race_id, umaban) → paytansyopay1/100` の辞書構築
- **D-03:** `final_win_odds_map`は`tanoddslow/100`を使用。`entries.parquet`の`tanoddslow`列を参照
- **D-04:** `paytansyopay1`欠損時のフォールバック: `tanoddslow/100`を使用し、WARNINGログを出力
- **D-05:** `EveryDB2Queries.get_payouts()` SQLに`paytansyoumaban1`, `paytansyopay1`を追加。将来のETLで正しく取得できるようにする

### 候補選択基準 (WIN-03, WIN-05)
- **D-06:** 基本フィルタ: `win_selection_edge > 0` AND `tanoddslow >= 1.0`。バランス型の選択
- **D-07:** ランキング方式: `win_gate_score`降順でソートし、上位候補を選択
- **D-08:** `win_gate_pass`はフィルタに使用せずログ表示のみ。ゲート未学習時でも候補選択が機能する
- **D-09:** 1レースあたり最大2頭の候補。現在のplace選択と同じ上限
- **D-10:** `get_win_candidates()`をRacePredictorに新規追加。`get_place_candidates()`と対称的な設計

### betting-target設計 (WIN-04)
- **D-11:** `--betting-target`は排他型モード: `win|place|wide`のいずれか1つのみ選択
- **D-12:** デフォルト値は`win`。v1.2は「単勝」マイルストーンであり、目的に合致
- **D-13:** ディスパッチはRacePredictor経由。`get_win_candidates()`/`get_place_candidates()`をBacktestEngineがbetting_targetに応じて呼び分け。ベストプラクティス追求
- **D-14:** BacktestEngine.__init__()に`betting_target: str = "win"`パラメータを追加

### WF検証の単勝化 (WIN-05)
- **D-15:** `run_wf_validation.py`は最小修正。`--betting-target`引数(default=win)を追加し、BacktestEngineに渡すのみ
- **D-16:** フォールド定義は変更なし。既存の2フォールド(Fold 0: 2020-2023→2024, Fold 1: 2021-2024→2025)をそのまま使用
- **D-17:** 過学習検出ロジック・特徴量安定性ロジックは変更なし

### Claude's Discretion
- `build_win_payout_map()`の具体的な実装（payouts DataFrameからのマップ構築方法）
- `get_win_candidates()`の返り値の型（DataFrame or list of Bet objects）
- `select_bets()`へのwin path追加方法（既存メソッドの拡張 vs 新規メソッド）
- `BacktestEngine.run()`内のwin_payout_map/win_odds_map構築タイミング
- `_settle_bet()`のwin対応（BetType.WINの場合のwin_payout_map参照）
- ETL type rulesへの追加カラム(paytansyoumaban2/3, paytansyopay2/3)対応の要否
- MLflowログへの単勝ROI記録フォーマット

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### バックテストエンジン（主変更対象）
- `src/backtest/engine.py` lines 102-168 — build_payout_map()/build_wide_payout_map()。build_win_payout_map()の参照実装
- `src/backtest/engine.py` lines 210-820 — BacktestEngine.run()。race loop全体。win payout map構築とbetting_target分岐の追加ポイント
- `src/backtest/engine.py` lines 909-950 — _settle_bet()。BetType.WIN決済の修正ポイント。現在payout_map(複勝)を参照しているバグあり
- `src/backtest/engine.py` lines 601-609 — final_odds_map割当。final_win_odds_mapの追加ポイント

### RacePredictor（候補選択追加）
- `src/backtest/race_predictor.py` lines 51-222 — predict()フロー。推論チェーン全体
- `src/backtest/race_predictor.py` lines 408-525 — get_place_candidates()。get_win_candidates()の参照実装
- `src/backtest/race_predictor.py` lines 532-642 — select_bets()。win path追加ポイント

### WinSelectionGate（既存、参照のみ）
- `src/models/win_selection_gate.py` lines 982-1001 — score()。win_selection_ev/edge/probの計算
- `src/models/win_selection_gate.py` lines 804-878 — training。walk-forward OOF scoring
- `src/models/win_selection_gate.py` lines 19, 434-481 — build_win_selection_ev(), threshold grid search

### データソース
- `src/db/everydb2_queries.py` lines 267-274 — get_payouts() SQL。paytansyoumaban1/paytansyopay1追加ポイント
- `src/db/etl.py` lines 111-114 — _TABLE_TYPE_RULES["payouts"]。既存type rules参照
- `src/db/readers.py` line 241-243 — load_payouts()。Parquetからのpayouts読み込み
- `data/raw/payouts.parquet` — 201列、38,835行。paytansyoumaban1/paytansyopay1を含む

### CLI スクリプト
- `scripts/run_backtest.py` lines 61-88 — CLI arguments。--betting-target追加ポイント
- `scripts/run_backtest.py` — BacktestEngine生成箇所。betting_target渡し
- `scripts/run_wf_validation.py` lines 45-58 — フォールド定義。変更不要
- `scripts/run_wf_validation.py` lines 142-238 — フォールド実行。--betting-target追加

### ドメイン型
- `src/domain/types.py` — BetType enum (WIN, PLACE, WIDE)。変更不要
- `src/domain/models.py` lines 229-250 — SubmodelSet dataclass

### 要件定義
- `.planning/REQUIREMENTS.md` — WIN-01, WIN-02, WIN-03, WIN-04, WIN-05
- `.planning/ROADMAP.md` — Phase 8 Success Criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **build_payout_map()** (`engine.py:102-125`): 複勝payout mapの構築パターン。payfukusyoumaban/payfukusyopayのループ処理。build_win_payout_map()は同じパターンでpaytansyoumaban1/paytansyopay1を使用（1着のみなのでループ不要）
- **get_place_candidates()** (`race_predictor.py:408-525`): 複勝候補選択の完全実装。place_selection_edge/place_selection_prob/fukuoddslowでフィルタリング。get_win_candidates()は対称的な設計
- **WinSelectionGate.score()** (`win_selection_gate.py:982`): 既にwin_selection_ev/edge/probを計算済み。tanoddslowベースで動作。候補選択にそのまま利用可能
- **_settle_bet()** (`engine.py:909-950`): BetType.WINの分岐が既に存在（finish == 1の判定）。現在はpayout_map(複勝)を誤参照しているだけ。win_payout_mapを追加すれば修正完了
- **Conformal confidence** (`robust_confidence_estimator.py`): predict_lower_bound()とpredict_interval()が利用可能。Phase 6で実装済み

### Established Patterns
- **payout_map構築パターン**: payouts DataFrameから`(race_id, umaban) → multiplier`辞書を構築。groupbyまたはiterrowsで処理
- **final_odds_map構築パターン**: entries DataFrameから`(race_id, umaban) → odds`辞書を構築。現在はfukuoddslowを使用
- **RacePredictor候補選択パターン**: DataFrameフィルタ → sort → head(max_bets) → Bet objects生成
- **CLI引数追加パターン**: argparse.ArgumentParser.add_argument()。--betting-mode(flat/kelly)が既存例
- **BacktestEngineパラメータ渡し**: __init__()で受け取り、run()内で参照

### Integration Points
- **BacktestEngine.__init__()** (`engine.py`): betting_target="win"パラメータ追加。win_payout_map, final_win_odds_mapの初期化
- **BacktestEngine.run() lines 225-289** (`engine.py`): データロード部。win payout map + win odds mapの構築タイミング
- **BacktestEngine.run() lines 420-786** (`engine.py`): レースループ。betting_targetに応じた候補選択呼び分け
- **RacePredictor.predict()** (`race_predictor.py:51`): win_selection_ev/edge/probは既に計算済み。get_win_candidates()はこの結果を利用
- **RacePredictor.select_bets()** (`race_predictor.py:532`): win path追加。betting_target == "win"時にget_win_candidates()を呼び出し
- **_settle_bet()** (`engine.py:909`): BetType.WINの場合にwin_payout_mapを参照するよう修正

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質優先で実装する
- 決済精度を最優先: 実際のJRA払戻金を使用し、オッズベース近似はフォールバックのみ
- 候補選択はバランス型: edge>0で足切りし、win_gate_scoreでランキング。保守的すぎず積極的すぎない
- RacePredictorを経由した責務分散がベストプラクティス。BacktestEngineはオーケストレーター、RacePredictorが選択ロジック
- WF検証は最小修正。派手な機能追加より、正しい単勝ROI検証に集中
- ETL SQL修正は将来の再ETLに備えた予防措置。現在のParquetには既にデータが存在

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 8-Win Backtest Core*
*Context gathered: 2026-05-04*
