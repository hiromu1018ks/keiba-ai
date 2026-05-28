# Phase 43: Shadow Diagnosis - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 41 Shadow Comparison Framework の出力成果物を入力として、baseline vs shadow の確率品質・選定パターン・キャリブレーション乖離を全面比較し、ROI劣化(97.8%→87.8%)の次元を特定する診断フェーズ。Phase 44 (ROI Bisect) と Phase 45 (Structural Fix) の入力となる。

**In scope:** DIAG-01 (確率品質比較)、DIAG-02 (選定パターン差分)、DIAG-03 (actual/predicted比率乖離)。
**Out of scope:** 再学習・BacktestEngine再実行、レジーム別分析、ROI最適化、新特徴量追加、Phase 41 Framework自体の変更。

</domain>

<decisions>
## Implementation Decisions

### 診断実行形態

- **D-01:** 後処理スクリプトとして実装。`scripts/run_shadow_diagnosis.py` (CLI) + `src/backtest/shadow_diagnosis.py` (ロジック)。Phase 41成果物のみを入力とし、再学習・BacktestEngine再実行は行わない。
- 入力ファイル: `shadow_comparison_result.json`, `shadow_race_diff.parquet`, `shadow_horse_diff.parquet`, `shadow_manifest.json` (全てPhase 41出力)。
- Phase 41成果物に必要列(popularity, probability_rank等)が不足している場合は、`missing_inputs` として診断レポートに明示し、Phase 41の出力拡張候補として記録する。

### 劣化次元の分解手法 (段階的除外アプローチ)

- **D-02:** 3ステップの段階的除外で劣化次元を特定:
  1. **確率品質次元**: 全馬ベースでbaseline vs shadowのBrier/logloss/ECE/actual_predicted_ratioを比較。確率品質そのものの劣化の有無を判定。
  2. **選定次元**: selected_changed vs unchanged レースに分け、ROI/的中率/avg_odds/actual_predicted_ratioの差分を比較。選定変更が劣化に寄与しているかを判定。
  3. **キャリブレーション次元**: surface/odds_band/popularity_band/probability_rank_band/selected_changed別にactual/predicted比率とECEを比較。キャリブレーション乖離箇所を特定。
- ΔBrier/Δlogloss/ΔROIのセグメント別寄与度は診断指標として出力。因果分解とは呼ばず、診断レポート内で参考値として扱う。

### セグメント定義

- **D-03:** セグメント境界:
  - `popularity_band` (単勝オッズ順位): [1-3, 4-6, 7-9, 10-14, 15+] の5段階
  - `probability_rank_band` (レース内p_win順位): [top1, 2-3, 4-6, 7+] の4段階
  - `odds_band`: Phase 41既存定義を流用
  - `surface`: turf / dirt (Phase 41既存)
  - `selected_changed`: True / False (Phase 41既存)
- 欠損やfield_size不足(例: 出走頭数<4でprobability_rank_band 7+が存在しない)は `unknown` にフォールバックし、診断レポートにmissing/unknown件数を出す。
- レジーム別分解は行わない(REQUIREMENTS.md Out of Scope準拠)。

### 出力フォーマット

- **D-04:** 3ファイル構成:
  - `shadow_diagnosis_result.json` — 機械可読な全分析結果(確率品質比較・選定差分・セグメント別キャリブレーション乖離) + `missing_inputs` (Phase 41出力に不足していた列のリスト)。Phase 44/45が消費する主要成果物。
  - `shadow_diagnosis_report.html` — Phase 41 Jinja2パターンに従うHTMLレポート。3ステップ段階的分析をセクション化表示(全馬確率品質 → selected_changed/unchanged → セグメント別calibration乖離)。ROI/HR差分も可視化。
  - `shadow_diagnosis_summary.md` — レビュー/コミット/PR用Markdown要約。主要な劣化次元、上位悪化セグメント、missing_inputs、Phase 44/45への推奨調査対象を短く記録。

### Claude's Discretion

- ShadowDiagnosis クラスの内部メソッド・データフロー設計
- Jinja2 HTMLテンプレートのレイアウト・スタイリング(Phase 41パターンに従う)
- テスト構造・命名(既存規約に従う)
- JSON出力のスキーマ設計(Phase 44/45が消費しやすい構造)
- missing_inputs 検出ロジックの実装詳細
- popularity_band / probability_rank_band 計算のエッジケース処理

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 41 成果物 (入力)
- `src/backtest/shadow_comparison.py` — ShadowComparisonFramework。出力する JSON/Parquet 成果物のスキーマ定義。
- `scripts/run_shadow_comparison.py` — Phase 41 CLI。出力ディレクトリ構造と成果物配置。

### Pipeline Integration Points
- `src/backtest/race_predictor.py` — RacePredictor。shadow診断ブロック(lines 860-884)でbaseline/shadow選定比較を記録。MAWC(lines 269-277)、Ranker(lines 279-285)。
- `src/backtest/engine.py` — BacktestEngine。BacktestResult と bet_history の列定義。
- `src/backtest/deployment_gates.py` — DeploymentGateEvaluator (Phase 42)。GatePolicy定義。

### Model Sources
- `src/models/market_aware_win_calibrator.py` — Phase 39 calibrator。shadow mode pattern。
- `src/models/race_level_ranker.py` — Phase 40 ranker。investment_score コンポーネント。
- `src/domain/models.py` — SubmodelSet, TrainedModelsV5。

### Prior Phase Context
- `.planning/phases/41-shadow-comparison-framework/41-CONTEXT.md` — Phase 41 比較基盤の決定事項。D-10(5出力成果物)、D-12(メトリクス)、D-13(集計次元)。
- `.planning/phases/42-feature-routing-audit-safety-gates/42-CONTEXT.md` — Phase 42 GatePolicy定義。D-11(ゲート条件と閾値)。
- `.planning/phases/40-race-level-ranker/40-CONTEXT.md` — Phase 40 ranker。D-03(investment_score組合せ)、D-05(コンポーネント別レポート)。

### Requirements
- `.planning/REQUIREMENTS.md` — DIAG-01, DIAG-02, DIAG-03 (Phase 43 requirements)。
- `.planning/ROADMAP.md` — Phase 43 success criteria (3 items)。
- `.planning/PROJECT.md` — Key Decisions (配備条件=確率品質, selection agreement = diagnostic not gate)。

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **ShadowComparisonFramework** (`src/backtest/shadow_comparison.py`): Phase 41 比較基盤。save_results() が JSON/Parquet/HTML/Manifest を出力。shadow_diagnosis も同じパターンで出力。
- **BacktestReportGenerator** (`src/backtest/report.py`): Jinja2 + HTMLテンプレートパターン。ShadowDiagnosisReportGenerator も同じパターンに従う。
- **save_results()** (`shadow_comparison.py`): JSON + Parquet + CSV + HTML + Manifest の5成果物出力パターン。ShadowDiagnosis の出力構造の参考。
- **Phase 41 shadow_race_diff.parquet**: race_id, baseline_selected_umaban, shadow_selected_umaban, selected_changed, baseline/shadow odds/p/EV/score/result/return/stake 列。DIAG-02 の主要入力。
- **Phase 41 shadow_horse_diff.parquet**: race_id, umaban, baseline/shadow p_win, p_win_market_aware, investment_score, rank, selected フラグ。DIAG-01/03 の主要入力。

### Established Patterns
- **Jinja2 HTML report**: BacktestReportGenerator + ShadowComparisonReportGenerator。ShadowDiagnosisReportGenerator も同じパターン。
- **JSON + Parquet + HTML 複数出力**: Phase 41 パターン。JSON は自動消費、Parquet は大規模データ、HTML は人間レビュー。
- **CLI引数パターン**: run_shadow_comparison.py の --baseline-root, --folds, --output-dir パターンを流用。
- **Manifest 再利用**: shadow_manifest.json を入力として使用し、診断結果にも参照を含める。

### Integration Points
- **入力**: `data/backtest/shadow/{fold_year}/` 内の Phase 41 成果物。
- **出力**: `data/backtest/shadow/diagnosis/` ディレクトリに配置。
- **消費者**: Phase 44 (ROI Bisect) が shadow_diagnosis_result.json を読み込んで劣化フェーズ特定に使用。

</code_context>

<specifics>
## Specific Ideas

- 段階的除外の3ステップ: 確率品質(全馬) → 選定(selected_changed/unchanged) → キャリブレーション(セグメント別actual/predicted比率+ECE)
- popularity_band: 単勝オッズ順位ベース [1-3, 4-6, 7-9, 10-14, 15+] — JRA競馬の本命/対抗/単穴/大穴パターンに対応
- probability_rank_band: レース内p_win順位ベース [top1, 2-3, 4-6, 7+] — モデル確信度の高低をキャプチャ
- missing_inputs フィールド: Phase 41出力に不足列があれば記録し、Phase 41拡張候補として扱う
- ΔBrier/Δlogloss/ΔROIのセグメント別寄与度を診断指標として出力(因果分解ではなく参考値)

</specifics>

<deferred>
## Deferred Ideas

- **Phase 41出力拡張**: popularity_rank, probability_rank 列の追加。Phase 43 の missing_inputs で特定された不足列をPhase 41で追加するかは別フェーズで判断。
- **レジーム別分析**: REQUIREMENTS.md で明示的に除外。v2.3+で検討。
- **LightGBM LambdaRank shadow variant**: Phase 41 D-09でv2.2+に延期済み。
- **因果分解的アプローチ**: ΔBrierの厳密な寄与度分解は将来検討。Phase 43では診断指標として扱う。

</deferred>

---

*Phase: 43-Shadow Diagnosis*
*Context gathered: 2026-05-28*
