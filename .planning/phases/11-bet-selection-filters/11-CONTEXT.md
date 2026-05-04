# Phase 11: Bet Selection Filters - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

低信頼ベット・不安定レジーム・赤字オッズバンドを自動除外し、バックテストのベット品質が向上する。

**In scope (from ROADMAP.md):**
- BSEL-01: バックテスト実行時、EV_lower_win_corrected >= 1.0 を満たさないベットが自動除外される
- BSEL-02: RegimeDetectorがCOLLAPSEDと判定したレースでベットが完全スキップされる
- BSEL-03: オッズバンド別ROI分析に基づき、赤字バンドのベットがOddsBandFilterで除外される
- 成功基準4: 全フィルター適用後の残存ベット数が年間1,000件以上を維持する

**Out of scope:**
- ステークサイジング最適化 (Phase 12: Stake Sizing Enhancement)
- パラメータチューニング/Optuna最適化 (Phase 13: Risk Calibration & Parameter Optimization)
- 複勝/ワイドモデルの変更
- 新データ源の導入
- モデル再学習

**Plans:** TBD

</domain>

<decisions>
## Implementation Decisions

### EV下限フィルター戦略 (BSEL-01)
- **D-01:** 既存 `win_selection_edge > 0` に加えて `EV_lower_win_corrected >= 1.0` のハードフィルターを追加。二重フィルター構成で安全性を確保
- **D-02:** フィルターは `get_win_candidates()` 内で早期適用。候補選択段階で除外し、後続のランキング・スコア計算が除外分に無駄な計算をしない
- **D-03:** `EV_lower_win_corrected` が NaN の場合、既存 `win_selection_edge > 0` のみで判定。Conformal推定が未学習・データ不足のレースでは既存の複合EVで候補を選択する（フォールバック）
- **D-04:** 除外統計ログ出力: 除外件数・EV_lower < 1.0 の割合をINFO レベルでログ出力。レポートのAI診断セクションにも反映。BSEL-01 Success Criteria「除外件数がログ/レポートに出力される」に対応

### OddsBandFilter のバンド特定 (BSEL-03)
- **D-05:** 動的解析アプローチ。バックテスト実行時にトレーニング期間のベットデータから各オッズバンドのROIを自動計算し、赤字バンドを自動特定。ルックアヘッドバイアスを防ぐためテスト期間外のデータのみ使用
- **D-06:** 新規クラス `OddsBandFilter` を `src/betting/odds_band_filter.py` に作成。BacktestEngine.run() のレースループ内で候補選択後に呼び出す。独立コンポーネントとしてテスト容易性を確保
- **D-07:** 赤字判定条件: トレーニング期間ROI < 100% のバンドを除外。シンプルで理解しやすい基準
- **D-08:** 除外統計ログ出力: 除外バンド名・件数・各バンドのROIをINFO レベルでログ出力。レポートのオッズバンド分析セクションに除外済みバンドを明示。BSEL-03 Success Criteria「除外バンド・件数がレポートに出力される」に対応

### フィルター連鎖とベット数ガード
- **D-09:** フィルター適用順序（レベル順）:
  1. COLLAPSEDレジームスキップ（レース全体 — BacktestEngine.run() 内、regime検出直後の early-return）
  2. EV下限フィルター（候補レベル — get_win_candidates() 内）
  3. OddsBandFilter（候補レベル — BacktestEngine.run() 内、候補選択後）
  レース全体除外を先に実行することで、候補選択の計算コストを節約
- **D-10:** ベット数ガード: バックテスト実行後に残存ベット数をログ出力し、1,000件/年未満なら WARNING を出す。自動緩和は行わない（過度な自動調整はルックアヘッドバイアスを生む）。パラメータ調整はPhase 13のOptuna最適化で対応
- **D-11:** COLLAPSEDスキップ実装: BacktestEngine.run() のレースループ内で `regime == RegimeState.COLLAPSED` なら `continue` でスキップ。スキップ件数をカウントしてログ出力。RegimeDetector.get_strategy_params() を拡張して COLLAPSED 時に `skip=True` を返す

### Claude's Discretion
- EV_lowerフィルターの具体的なpandasフィルター条件の実装
- OddsBandFilterのインターフェース設計（calibrate() + filter() メソッド等）
- バンド境界定義（Phase 9レポートと同じ 1.0-3.0/3.0-10.0/10.0-30.0/30.0+）
- 除外統計ログのフォーマット（INFO レベル、構造化ログ）
- WARNING の出力条件とフォーマット
- レポート拡張の具体的なコード変更（report.py への除外済みバンド表示追加）
- テスト戦略（フィルターごとの単体テスト + 統合テスト）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### RacePredictor（主変更対象 — EVフィルター追加）
- `src/backtest/race_predictor.py` lines 408-470 — get_win_candidates()。EV_lower >= 1.0 フィルターの追加ポイント
- `src/backtest/race_predictor.py` lines 472-589 — get_place_candidates()。参照実装（regime params活用パターン）
- `src/backtest/race_predictor.py` lines 596-753 — select_bets()。ベット生成フロー

### BacktestEngine（COLLAPSEDスキップ + OddsBandFilter呼び出し）
- `src/backtest/engine.py` lines 603-820 — run() レースループ全体。COLLAPSED early-return + OddsBandFilter呼び出しポイント
- `src/backtest/engine.py` lines 682-687 — regime検出。COLLAPSED スキップ分岐の追加ポイント
- `src/backtest/engine.py` lines 689-702 — 候補選択呼び出し。OddsBandFilterの呼び出しポイント（候補選択後）

### RegimeDetector（COLLAPSED スキップ拡張）
- `src/models/regime_detector.py` lines 133-176 — detect()。3状態分類 + hysteresis
- `src/models/regime_detector.py` lines 178-232 — get_strategy_params()。COLLAPSED 時の skip=True 追加ポイント

### WinSelectionGate（EV計算 — 参照のみ）
- `src/models/win_selection_gate.py` lines 19-30 — build_win_selection_ev()。EV優先順位
- `src/models/win_selection_gate.py` lines 33-54 — ensure_win_selection_columns()。win_selection_ev/edge/probの計算

### Conformal Confidence（EV_lower のソース — 参照のみ）
- `src/models/robust_confidence_estimator.py` lines 96-234 — predict_interval()。EV_lower_win_corrected の計算
- `src/models/robust_confidence_estimator.py` lines 204-218 — conformal_confidence_score の計算

### バックテストレポート（除外統計出力）
- `src/backtest/report.py` lines 17-99 — BacktestReportGenerator。除外統計の表示ポイント
- `src/backtest/report.py` lines 155-248 — _compute_condition_stats()。オッズバンド分析（1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+）
- `src/backtest/report.py` lines 90-193 — save_ai_diagnostics()。AI診断セクション

### ドメイン型
- `src/domain/types.py` lines 29-34 — RegimeState enum (AGGRESSIVE, CONSERVATIVE, COLLAPSED)
- `src/betting/stake_calculator.py` — StakeCalculator。参照のみ（変更なし）
- `src/betting/drawdown_controller.py` — DrawdownController。参照のみ（変更なし）

### 新規ファイル
- `src/betting/odds_band_filter.py` (新規) — OddsBandFilter クラス

### 要件定義
- `.planning/REQUIREMENTS.md` — BSEL-01, BSEL-02, BSEL-03
- `.planning/ROADMAP.md` — Phase 11 Success Criteria
- `.planning/STATE.md` — 決定済みアーキテクチャ（BettingOrchestrator非対象、新規コンポーネントはOddsBandFilterのみ）

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **get_win_candidates()** (`race_predictor.py:408-470`): 既存の候補選択ロジック。`win_selection_edge > 0 AND tanodds >= 1.0` でフィルタ → `win_gate_score DESC` でソート → max 2候補。EV_lowerフィルターはここに追加
- **_compute_condition_stats()** (`report.py:155-248`): 既存のオッズバンド分析（1.0-3.0, 3.0-10.0, 10.0-30.0, 30.0+）。OddsBandFilterのバンド定義と同じパターン
- **RegimeDetector.get_strategy_params()** (`regime_detector.py:178-232`): COLLAPSED時の戦略パラメータ返却。skip=Trueの追加ポイント
- **BacktestEngine.run() レースループ** (`engine.py:603-820`): COLLAPSED早期リターン + OddsBandFilter呼び出しの挿入ポイント

### Established Patterns
- **候補フィルターパターン**: get_win_candidates() / get_place_candidates() で DataFrame条件フィルタ → sort → head。EV_lowerフィルターも同じパターン
- **regime パラメータ活用パターン**: get_strategy_params() → params["ev_threshold"] 等を候補選択で使用。COLLAPSEDスキップもこのパターンの拡張
- **除外統計ログパターン**: engine.py のレースループでカウンタをインクリメント → run() 終了時にINFO ログ出力
- **オッズバンド分析パターン**: pd.cut() でバン化 → groupby → ROI計算。OddsBandFilterの動的解析も同じパターン

### Integration Points
- **race_predictor.py:408** — get_win_candidates() に EV_lower >= 1.0 フィルター追加
- **engine.py:687** — regime検出直後に COLLAPSED スキップ分岐追加
- **engine.py:689-702** — 候補選択後に OddsBandFilter.filter() 呼び出し追加
- **regime_detector.py:221-232** — get_strategy_params() COLLAPSED分岐に skip=True 追加
- **report.py** — 除外統計（EV除外・COLLAPSEDスキップ・オッズバンド除外）のレポート表示追加
- **src/betting/odds_band_filter.py** (新規) — OddsBandFilter クラス

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- 二重フィルター（edge>0 + EV_lower >= 1.0）は安全性を最大化するアプローチ。NaN フォールバックも含めて堅牢な設計
- 動的オッズバンド解析はルックアヘッドバイアスを防ぐ。トレーニング期間データのみでバンドROIを計算
- フィルター過剰除外の自動緩和は行わない（ルックアヘッドバイアス回避）。Phase 13 の Optuna 最適化で全体最適化
- COLLAPSED スキップ件数は成功基準「スキップレース数がレポートに記録される」に対応するため必ずログ出力

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-Bet Selection Filters*
*Context gathered: 2026-05-04*
