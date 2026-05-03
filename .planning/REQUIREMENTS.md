# Requirements: keiba-ai v1.2

**Defined:** 2026-05-04
**Core Value:** 単勝モデルのバックテストROIを100%超えにすること

## v1.2 Requirements

### Win Backtest Core

- [ ] **WIN-01**: build_win_payout_map()で単勝払戻しデータ(tan_umaban/tan_pay)を読み取り、payout_mapを構築できる
- [ ] **WIN-02**: final_odds_mapがtanoddslow(単勝オッズ)を使用し、単勝ベットの正しい決済を行える
- [ ] **WIN-03**: get_win_candidates()がwin_selection_ev/edge/prob列で候補をフィルタリングし、単勝ベット候補を生成できる
- [ ] **WIN-04**: BacktestEngineにbetting_targetパラメータを追加し、単勝/複勝モードを切り替えられる(デフォルト=WIN)
- [ ] **WIN-05**: Conformal信頼性スコア(conformal_confidence_score)を単勝ベット判定に組み込み、高信頼度ベットのみを生成できる

### Win Reporting

- [ ] **RPT-01**: バックテスト結果のベット履歴に単勝ベットの馬番・オッズ・EV・結果を記録できる
- [ ] **RPT-02**: 単勝ROI・回収率・的中率・ベット数の集計診断を出力できる
- [ ] **RPT-03**: オッズバンド別(人気・中穴・大穴)のROI内訳を分析・表示できる

### Pipeline Performance

- [ ] **PERF-01**: build_payout_map()/build_wide_payout_map()のiterrows()をベクトル化pandas操作に置き換えられる
- [ ] **PERF-02**: レースごとのDataFrameフィルタリングをgroupby辞書の前処理に置き換え、O(n_races * n_rows)→O(1)ルックアップにできる
- [ ] **PERF-03**: HorseHistoryFeatures等の履歴特徴量をParquetキャッシュし、バックテスト再実行時に再計算をスキップできる
- [ ] **PERF-04**: pyinstrumentによるプロファイリングを統合し、ボトルネックの定量測定ができる

## Future Requirements

### Win vs Place Comparison

- **COMP-01**: 同一条件下での単勝/複勝ROI比較レポートを生成できる
- **COMP-02**: ベッティング戦略の単勝/複勝切替による感度分析ができる

## Out of Scope

| Feature | Reason |
|---------|--------|
| 複勝/ワイドモデルの変更 | 単勝に集中するため |
| 新データ源の導入 | 既存EveryDB2データで十分 |
| 実馬券購入機能 | ペーパートレードまで |
| Web UI | CLIベースで十分 |
| LSTM/Transformer時系列モデリング | 過去5-15走では過学習リスク高 |
| 複雑メタラーナー(GBM/NN) | 特徴量3個ではRidgeが最適 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| WIN-01 | — | Pending |
| WIN-02 | — | Pending |
| WIN-03 | — | Pending |
| WIN-04 | — | Pending |
| WIN-05 | — | Pending |
| RPT-01 | — | Pending |
| RPT-02 | — | Pending |
| RPT-03 | — | Pending |
| PERF-01 | — | Pending |
| PERF-02 | — | Pending |
| PERF-03 | — | Pending |
| PERF-04 | — | Pending |

**Coverage:**
- v1.2 requirements: 12 total
- Mapped to phases: 0
- Unmapped: 12 ⚠️

---
*Requirements defined: 2026-05-04*
*Last updated: 2026-05-04 after initial definition*
