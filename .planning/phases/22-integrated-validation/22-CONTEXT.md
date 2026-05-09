# Phase 22: 統合検証とバックテスト - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 19（Isotonic EVキャリブレーション）、Phase 20（高オッズ的中パターン特徴量）、Phase 21（CQR Conformal EV予測区間）の全改善を適用したバックテストを実行し、ROI改善を確認する。包括的セグメント分析でv1.4ベースラインとの比較を行う。

**In scope:**
- VAL-01: 全改善適用後のバックテスト実行（2024単年、flat、calibration-bt有効）
- VAL-02: EVキャリブレーション品質検証（包括的セグメント分析）
- バックテスト結果の分析レポート生成（--report）
- 既存テスト全通過確認（1,393テスト、回帰なし）
- v1.4ベースライン（ROI 83.1%、EV過大評価2.42倍）との比較

**Out of scope:**
- WF検証（run_wf_validation.py、~4時間）— 別セッションで実行
- Optuna戦略パラメータ再最適化 — 次マイルストーンで対応
- 複勝/ワイドモデルの検証
- 新しい特徴量やモデルの追加実装
- validation_suite.pyのTODO（logloss, spearman_rho）対応
- 追加の差分比較スクリプト作成

</domain>

<decisions>
## Implementation Decisions

### バックテスト実行設定
- **D-01:** テスト年度は2024単年。学習期間2020-2023、テスト期間2024。v1.4ベースライン（2024テスト）との直接比較が可能。
- **D-02:** Betting modeはflat（100円固定）。Kelly sizingの影響を除外し、純粋なモデル改善効果を測定する。
- **D-03:** `--calibration-bt`を有効化。OddsBandFilterの再キャリブレーションを実行（直近12ヶ月の軽量BT、~16分追加）。Phase 19のIsotonic改善後はEV分布が変化しているため、再キャリブレーションに意味がある可能性。
- **D-04:** `--report`を有効化。HTMLレポート + parquet出力 + validation_report.jsonを自動生成。
- **D-05:** Strategy manifestは既存のもの（ユーザー変更済み）をそのまま使用。manifestはキャリブレーションとは独立。

### 比較ベースラインと評価指標
- **D-06:** 包括的セグメント分析を実施。測定指標: ROI、高オッズ帯ROI(20+)、EV過大評価倍率、レジーム別ROI、オッズバンド別ROI、的中率、平均オッズ。Success Criteria（ROI 95%+、高オッズ帯ROI 50%+）の2項目で判定。
- **D-07:** v1.4ベースラインは既存数値（ROI 83.1%、EV過大評価2.42倍）を使用。再実行なし。Phase 19.1のP0-P2最適化は結果完全一致、P3は<5%差異という前提。
- **D-08:** レポート出力は既存の`--report`機構を利用。BacktestReportGenerator/MultiYearReportGeneratorがセグメント別内訳を自動生成。追加の差分比較スクリプトは不要。

### 検証スコープ
- **D-09:** WF検証はスキップ。Phase 22はバックテスト単発に集中。WF検証は別セッションで実行可能（run_wf_validation.py --ensemble）。
- **D-10:** EV診断は既存機構のみ。ev_diagnostics.pyがECE/Brier/Reliability/CQRカバレッジを自動計算。validation_suite.pyのTODO（logloss, spearman_rho）は対応しない。

### 不達時対応とテスト戦略
- **D-11:** ROI 95%不達時は現状でv1.5完了とする。バックテスト結果の分析レポートを出力して終了。改善（パラメータ調整、フィルター調整等）は次マイルストーンで対応。
- **D-12:** テストは既存1,393テストの通過確認のみ。統合E2Eテストの追加はしない。Phase 19/20/21の各単体テスト（計48テスト）が包括的。

### Claude's Discretion
- バックテスト実行の具体的な手順（スクリプト実行順序、結果の検証方法）
- レポート結果の解釈とサマリの提示方法
- セグメント別分析の具体的な出力形式

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### バックテスト実行（主変更対象 — VAL-01）
- `scripts/run_backtest.py` — バックテストCLIエントリーポイント。--calibration-bt, --report, --ensemble等のフラグ
- `scripts/run_backtest.py:110-115` — --calibration-bt 引数定義
- `scripts/run_backtest.py:225-261` — _collect_training_bet_history() キャリブレーションBT実行箇所

### BacktestEngine（バックテスト本体）
- `src/backtest/engine.py` — BacktestEngine。run()でレースループ実行
- `src/backtest/engine.py:77-81` — BacktestResultデータクラス。ROI/bet_count等の結果格納
- `src/backtest/engine.py:900-905` — n_ev_excluded集計

### 推論パイプライン（Phase 19+20+21統合チェーン）
- `src/backtest/race_predictor.py` — RacePredictor。predict() → get_win_candidates()
- `src/backtest/race_predictor.py:166-186` — ConformalEV予測区間の適用
- `src/backtest/race_predictor.py:420-480` — get_win_candidates() EV_lowerフィルター

### EV診断（品質評価 — VAL-02）
- `src/models/ev_diagnostics.py` — compute_ev_diagnostics() ECE/Brier/Reliability/CQRカバレッジ
- `src/models/ev_diagnostics.py:160-282` — オッズバンド別EV過大評価分析

### レポート生成
- `src/backtest/report.py` — BacktestReportGenerator / MultiYearReportGenerator
- `src/backtest/validation_report.py` — generate_validation_report() ROI/ベットカウント PASS/FAIL評価

### 学習パイプライン（Phase 19/20/21統合状態）
- `src/pipelines/training_pipeline.py` — TrainingPipelineV5。全モデル学習チェーン
- `src/pipelines/training_pipeline.py:559-585` — OOF EV予測生成 + Isotonic学習
- `src/pipelines/training_pipeline.py:851-898` — ConformalEV学習統合

### 前フェーズのCONTEXT（必読 — 決定の連続性）
- `.planning/phases/19-ev-calibration/19-CONTEXT.md` — Phase 19決定（Isotonic、OOF生成、オッズバンド補正）
- `.planning/phases/20-high-odds-pattern-features/20-CONTEXT.md` — Phase 20決定（高オッズ特徴量）
- `.planning/phases/21-conformal-ev/21-CONTEXT.md` — Phase 21決定（CQR、動的フィルタリング）
- `.planning/phases/19.1-backtest-speedup-optimization/19.1-CONTEXT.md` — Phase 19.1決定（バックテスト高速化）

### 既存テストパターン
- `tests/test_backtest_engine.py` — BacktestEngine既存テスト（~50テスト）
- `tests/test_ev_isotonic.py` — Isotonicキャリブレーションテスト（15テスト）
- `tests/test_high_odds_features.py` — 高オッズ特徴量テスト（17テスト）
- `tests/test_conformal_ev_model.py` — ConformalEVテスト（16テスト）

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **BacktestEngine** (`src/backtest/engine.py`): run()が学習→推論→精算の全フローを実行。--calibration-bt, --reportフラグで挙動制御。Phase 19.1の高速化適用済み
- **BacktestReportGenerator** (`src/backtest/report.py`): HTMLレポート + parquet出力。セグメント別内訳（オッズバンド、レジーム、芝/ダート）を自動生成
- **generate_validation_report()** (`src/backtest/validation_report.py`): ROI/ベットカウントのPASS/FAIL自動判定
- **compute_ev_diagnostics()** (`src/models/ev_diagnostics.py`): ECE/Brier/Reliability/CQRカバレッジを含む包括的EV品質評価

### Established Patterns
- **mockベーステスト**: 全テストがDB不要。unittest.mock使用。Phase 22でも回帰テストはこのパターン
- **--reportフラグ**: バックテスト結果のHTML/parquet/JSON出力を制御。既存機構でセグメント別分析を自動生成
- **v1.4ベースライン**: ROI 83.1%、EV過大評価2.42倍。高オッズ帯(20+)のP過大評価1.98倍

### Integration Points
- **scripts/run_backtest.py** — CLIエントリーポイント。--calibration-bt + --report + --ensemble構成で実行
- **src/backtest/engine.py:run()** — 学習→特徴量→推論→精算→レポートの全フロー
- **src/backtest/validation_report.py** — 結果のPASS/FAIL判定とvalidation_report.json出力

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針。品質・堅牢性を優先
- Phase 22は実装フェーズではなく検証フェーズ。コード変更は最小限（バックテスト実行 + テスト確認 + 結果レポート）
- バックテスト実行はPostgreSQL環境（localhost:5432/everydb2）が必要。所要時間~57分
- Phase 19.1の高速化でバックテスト実行時間は64-81%短縮済み
- CQRのPlace EV対応は未実装（Phase 21 D-11の「TODO: implement CQR for place EV if needed」）。単勝検証には影響なし

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 22-統合検証とバックテスト*
*Context gathered: 2026-05-09*
