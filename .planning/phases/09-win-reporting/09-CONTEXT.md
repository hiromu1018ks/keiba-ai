# Phase 9: Win Reporting - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

ユーザーが単勝バックテスト結果のベット履歴・ROI診断・オッズバンド別内訳を確認できるようにする。

**In scope (from ROADMAP.md):**
- RPT-01: バックテスト結果のベット履歴に単勝ベットの馬番・オッズ・EV・結果を記録できる
- RPT-02: 単勝ROI・回収率・的中率・ベット数の集計診断を出力できる
- RPT-03: オッズバンド別(人気・中穴・大穴)のROI内訳を分析・表示できる

**Out of scope:**
- パフォーマンス最適化 (Phase 10: Pipeline Performance)
- 複勝/ワイドレポートの変更
- ベッティング戦略の変更
- 新データ源の導入

**Plans:** 1 plan
- 09-01: Win bet history + ROI diagnostics + odds band analysis (RPT-01, RPT-02, RPT-03)

</domain>

<decisions>
## Implementation Decisions

### オッズバンド分析の定義 (RPT-03)
- **D-01:** 人気順位バンド(1-3番人気/4-6番人気/7番人気以降)に加えて、オッズ倍率バンドも実装。2次元の分析で収益構造を深く理解する
- **D-02:** 人気順位バンドは既存の `_compute_condition_stats()` の popularity bands (1-3, 4-6, 7+) をそのまま利用。RPT-03と完全一致
- **D-03:** オッズ倍率バンドの区分はClaude裁量で最適化。JRA控除率25%と実用的な投資リスク区分を考慮

### レポート拡張方針
- **D-04:** 既存 `BacktestReportGenerator` を拡張し、betting_targetで条件分岐。新規クラスは作らない
- **D-05:** win指定時は単勝専用セクションを出力。place/wide時は既存ロジックを維持
- **D-06:** 2層出力: (1) 人間向け HTML + CLI標準出力（視覚的にわかりやすい）、(2) AI分析向け 構造化JSON（改善点を自動特定しやすい形式）

### 診断出力の詳細度 (RPT-02)
- **D-07:** 包括的診断を実装。基本4指標(ROI・回収率・的中率・ベット数)に加えて以下を含める:
  - 月別推移 (monthly ROI/bets/wins)
  - 表面×距離別内訳 (turf/dirt × sprint/mile/intermediate/long)
  - Regime別内訳 (aggressive/conservative/collapsed)
  - EVバンド別内訳 (<1.0, 1.0-1.2, 1.2-1.5, 1.5+)
  - オッズバンド別内訳 (popularity + odds multiplier)
- **D-08:** AI分析用JSONには、改善点自動特定のための構造化データを含める:
  - 各バンドのROIとベースラインROIとの比較
  - 最も寄与度の高い/低いバンドのハイライト
  - 月別ROI推移のトレンド情報

### bet_history追加フィールド (RPT-01)
- **D-09:** 包括的フィールドを記録。基本フィールドに加えて以下を追加:
  - `win_selection_ev`: WinSelectionGateによるEV評価値
  - `win_selection_edge`: モデルエッジ(p_model - p_market)
  - `win_selection_prob`: モデル予測確率
  - `win_gate_score`: ゲートスコア(ランキング指標)
  - `conformal_confidence_score`: Conformal信頼性スコア
  - `tanoddslow`: 単勝オッズ(確定値)
  - `kakuteijyuni`: 確定着位
  - `popularity`: 人気順位
- **D-10:** これらのフィールドにより、どのスコア成分がROIに寄与しているか、どのconfidence閾値が有効かを事後分析可能

### Claude's Discretion
- オッズ倍率バンドの具体的な区分（JRA控除率と実データ分布を考慮）
- AI分析用JSONのスキーマ詳細（改善点自動特定のロジック）
- HTMLレポートの視覚デザイン（既存スタイルに準拠）
- CLI標準出力のフォーマット（既存display_single_year_result()パターンに準拠）
- bet_historyフィールドのengine.pyでの取得方法（race_predictorのDataFrameから参照）
- MultiYearReportGeneratorへの対応範囲

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### レポート生成（主変更対象）
- `src/backtest/report.py` lines 17-99 — BacktestReportGenerator。HTML/JSON/CLI生成のメインクラス。betting_target条件分岐の追加ポイント
- `src/backtest/report.py` lines 106-135 — _compute_monthly_stats()。月別集計パターン
- `src/backtest/report.py` lines 155-248 — _compute_condition_stats()。既存popularity bands (1-3, 4-6, 7+) + EV bands + surface×distance。オッズ倍率バンドの追加ポイント
- `src/backtest/report.py` lines 250-298 — _compute_daily_stats()。日別集計
- `src/backtest/report.py` lines 301-377 — MultiYearReportGenerator。マルチ年度レポート

### bet_history構築（フィールド追加）
- `src/backtest/engine.py` lines 758-812 — bet_history dict構築。win-specificフィールドの追加ポイント
- `src/backtest/engine.py` lines 56-99 — BacktestResult dataclass。summary()メソッド
- `src/backtest/engine.py` lines 849-879 — ROI計算・BacktestResult構築

### CLIスクリプト（表示・保存）
- `scripts/run_backtest.py` lines 215-263 — display_single_year_result()。CLI標準出力の表示関数
- `scripts/run_backtest.py` lines 343-365 — 単一年レポート出力（JSON/HTML/Parquet保存）
- `scripts/run_backtest.py` lines 500-542 — マルチ年度レポート出力

### WinSelectionGate（追加フィールドのソース）
- `src/models/win_selection_gate.py` lines 982-1001 — score()。win_selection_ev/edge/probの計算
- `src/models/robust_confidence_estimator.py` — conformal_confidence_scoreの計算

### RacePredictor（bet_historyフィールド参照元）
- `src/backtest/race_predictor.py` lines 51-222 — predict()フロー。win_selection_*フィールドがDataFrameに格納される場所
- `src/backtest/race_predictor.py` lines 408-525 — get_win_candidates()。win候補選択ロジック

### ドメイン型
- `src/domain/types.py` — BetType enum (WIN, PLACE, WIDE)
- `src/domain/models.py` lines 151-176 — Bet dataclass

### 要件定義
- `.planning/REQUIREMENTS.md` — RPT-01, RPT-02, RPT-03
- `.planning/ROADMAP.md` — Phase 9 Success Criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **_compute_condition_stats()** (`report.py:155-248`): 既存のpopularity bands (1-3, 4-6, 7+)とEV bands (<1.0, 1.0-1.2, 1.2-1.5, 1.5+)。オッズ倍率バンドは同じパターンで追加可能
- **_compute_monthly_stats()** (`report.py:106-135`): YYYY-MM groupby → bets/wins/stake/return/ROI。regime別分析も同じパターン
- **display_single_year_result()** (`run_backtest.py:215-263`): CLI出力フォーマットの参照実装。win版は同パターンで単勝指標を出力
- **BacktestReportGenerator._derive_fields()** (`report.py:93-104`): bet_history dictにderived fields追加のパターン。win版フィールド追加に適用可能
- **BacktestResult.summary()** (`engine.py:56-99`): 既存テキストサマリー。win版診断の追加ポイント

### Established Patterns
- **ROI計算パターン**: `total_return / total_stake`。report.py/engine.py/run_backtest.pyで統一
- **band分析パターン**: bet_history listをpd.DataFrame化 → カラムでcut/bucketize → groupby → 計算
- **HTML/JSON二層出力**: BacktestReportGenerator.generate()がHTML、save_bet_history()がJSON
- **CLI表示パターン**: f-string整形でターミナルにテーブル出力。display_single_year_result()参照

### Integration Points
- **engine.py bet_history構築部** (lines 758-812): win-specificフィールド追加のメインポイント。race_predictorの出力DataFrameから値を取得
- **report.py BacktestReportGenerator** (lines 17-99): betting_target条件分岐の追加。win時の表示項目・バンド分析変更
- **run_backtest.py display_single_year_result()** (lines 215-263): CLI出力のwin化。ROI・的中率等の単勝指標表示
- **report.py MultiYearReportGenerator** (lines 301-377): マルチ年度レポートのwin対応

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・情報量を優先
- 2層出力（人間向け + AI分析向け）により、レポート結果から改善点を自動特定できる設計を目指す
- 人気順位バンド + オッズ倍率バンドの2次元分析で、収益構造を多角的に理解
- 包括的フィールド記録により、事後分析で「どのスコア成分がROIに寄与したか」を特定可能にする
- 既存report.pyの拡張で一元管理。新規クラスは作らない（コード重複回避）

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 9-Win Reporting*
*Context gathered: 2026-05-04*
