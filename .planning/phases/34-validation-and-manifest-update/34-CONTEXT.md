# Phase 34: Validation and Manifest Update - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

新特徴量追加後（Phase 31-32）の統合検証フェーズ。フルバックテストでROI改善を確認し、IC値を新ベースラインとして記録し、GPD診断で新特徴量の効果を検証し、FEATURE_COLS manifestを凍結する。

**In scope:**
- VAL-01: 2024年度単年バックテスト（ensemble、flat 100円固定、manifestなし）
- VAL-02: v1.7 IC値の記録（turf/dirt/all 3サーフェス別、B差分/C直交/E増分/Per-raceの4定式化）。ベースライン比較はスキップし、v1.7 IC値を将来のベースラインとする
- VAL-03: GPD診断の実行とClaudeによる判定（MDR > 0 and FAD <= 5）
- VAL-04: FEATURE_COLS manifest SHA256凍結 + 全12モデル更新
- VAL-05: POST_RACE情報漏洩テストの再実行（新特徴量に対する漏洩チェック）
- 検証結果が目標未達の場合: 結果をありのまま記録し、manifestを凍結して次マイルストーンで改善を計画

**Out of scope:**
- 新特徴量の設計・実装 (Phase 31/32 complete)
- Optuna戦略パラメータ最適化 (将来フェーズ)
- モデル再学習・ハイパーパラメータチューニング
- v1.6ベースラインIC値の取得（OOF予測未保存のため）
- ETL拡張 (Phase 29 complete)
- IC評価モジュールの実装 (Phase 30 complete)
- GPD診断ツールの実装 (Phase 33 complete)

</domain>

<decisions>
## Implementation Decisions

### バックテスト設定
- **D-01:** 対象年度は **2024年のみ**（単年BT）。v1.6結果(85.7%)のテスト年度2024との直接比較が可能
- **D-02:** strategy_manifestなしでバックテスト実行。デフォルトパラメータ使用。v1.6もmanifestなしで85.7%だったため条件は同一
- **D-03:** betting-modeは **flat（100円固定）**。v1.6と同じ条件でROIの純粋なモデル改善を測定
- **D-04:** --calibration-btは **スキップ**。OddsBandFilterの再キャリブレーションは行わない

### ベースラインIC比較方法
- **D-05:** v1.6ベースラインIC比較はスキップ。v1.6学習時にOOF save hookが未存在のため
- **D-06:** v1.7 IC値を **turf/dirt/all 3サーフェス別** に4定式化（B差分/C直交/E増分/Per-race）で記録。将来の改善測定のベースラインとする
- **D-07:** IC評価は **BT後に実行**。BT学習時のOOF save hookで保存されたoof_predictions.parquetを使用

### GPD診断と判定
- **D-08:** GPDレポートを実行し **Claudeが判定**。MDR > 0（Marketがshallow優位）かつ FAD <= 5（Fundamentalがdepth 5以下で活性化）を成功基準とする
- **D-09:** 新特徴量（race-level 6個 + market-cross 5個）がMarket分類としてshallow depthで機能していることを確認

### 検証失敗時の対応
- **D-10:** 検証結果が目標未達の場合、**結果をありのまま記録**してmanifestを凍結。Phase 34内でのOptuna再最適化や特徴量削除・再BTは行わない
- **D-11:** 改善が必要な場合は次マイルストーン（v1.8）で計画

### 実行フロー
- **D-12:** 検証の実行順序:
  1. POST_RACE漏洩テスト（VAL-05）— 最速、問題あれば早期発見
  2. マルチ年度BT 2024（VAL-01）— OOF予測を保存
  3. IC評価（VAL-02）— BT保存OOFからIC値計算
  4. GPD診断（VAL-03）— 学習済みモデルでdepth分析
  5. Manifest凍結（VAL-04）— 全検証完了後にSHA256ハッシュ更新

### Claude's Discretion
- BT結果のROI値とv1.6(85.7%)の比較解釈
- IC値の「良い/悪い」の判定（ベースラインがないため、Claudeの判断で記録）
- GPD結果のMDR/FAD値に基づくPASS/WARN判定
- 各検証ステップ間のエラーハンドリング（BT失敗時のIC評価スキップ等）
- 検証結果レポートの形式（JSON、console summary等）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### バックテスト実行 (VAL-01)
- `scripts/run_backtest.py` — マルチ年度BT実行スクリプト。`--years 2024 --train-window 4 --ensemble --betting-mode flat --betting-target win`
- `src/backtest/engine.py` — BacktestEngine。BT実行のメインロジック
- `src/pipelines/training_pipeline.py` — TrainingPipeline。学習＋OOF保存（Phase 30 hook）

### IC評価 (VAL-02)
- `src/models/ic_evaluator.py` — IC evaluator（Phase 30作成）。4定式化 + turf/dirt/all別計算
- `scripts/run_ic_eval.py` — IC評価CLI。OOF Parquetパスを受け取りIC値を計算
- `data/oof/oof_predictions.parquet` — BT学習時に保存されるOOF予測

### GPD診断 (VAL-03)
- `src/models/gpd_diagnostics.py` — GPD診断モジュール（Phase 33作成）。FEATURE_CATEGORY_MAP + depth別gain集計
- `scripts/run_gpd.py` — GPD診断CLI。学習済みモデルでMDR/FADを計算

### Manifest凍結 (VAL-04)
- `scripts/freeze_feature_manifest.py` — manifest生成スクリプト。12モデルのFEATURE_COLSをSHA256で凍結
- `src/models/stage1_ability_model.py` — AbilityModel.FEATURE_COLS
- `src/models/two_stage_return_model.py` — WinTwoStageModel / PlaceTwoStageModel FEATURE_COLS
- `src/models/ev_correction_model.py` — EVCorrectionModel / PlaceEVCorrectionModel FEATURE_COLS
- `src/models/conformal_ev_model.py` — ConformalEVModel.FEATURE_COLS
- `src/models/market_model.py` — MarketModel.FEATURE_COLS
- `src/models/stacked_ensemble.py` — StackedEnsemble.FEATURE_COLS
- `src/models/regime_detector.py` — RegimeDetector.FEATURE_COLS

### POST_RACE安全性 (VAL-05)
- `tests/test_post_race_leakage.py` — 3層テストアーキテクチャ
- `src/domain/types.py` (lines 38-55) — POST_RACE_COLS定義

### 要件定義
- `.planning/REQUIREMENTS.md` §Validation — VAL-01~05

### Prior Phase Context
- `.planning/phases/31-race-level-aggregation-features/31-CONTEXT.md` — race-level特徴量定義
- `.planning/phases/32-market-cross-consistency-features/32-CONTEXT.md` — market-cross特徴量定義
- `.planning/phases/33-gain-per-depth-diagnostic/33-CONTEXT.md` — GPD診断設計（MDR/FAD指標）
- `.planning/phases/30-residual-ic-evaluation-framework/30-CONTEXT.md` — IC evaluator設計

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/run_backtest.py`: 既存のマルチ年度BTスクリプト。`--years 2024 --train-window 4 --ensemble` で単年BT実行可能
- `src/models/ic_evaluator.py`: 4定式化IC計算。`run_ic_eval.py` でOOF Parquetから直接実行可能
- `src/models/gpd_diagnostics.py`: FEATURE_CATEGORY_MAP（179特徴量をMarket/Fundamental/Categoricalに分類）+ MDR/FAD計算
- `scripts/freeze_feature_manifest.py`: 12モデルFEATURE_COLS SHA256凍結。既存スクリプトをそのまま実行
- `tests/test_post_race_leakage.py`: 3層漏洩検出テスト。Phase 31/32の新特徴量も自動的にテスト対象

### Established Patterns
- 検証スクリプトパターン: CLI argparse → ModelLoader → 診断/評価関数 → JSON出力
- OOF save hook: TrainingPipeline.run()内でfull_features_dfをParquet保存（Phase 30で追加）
- Manifest凍結: freeze_feature_manifest.py → data/feature_manifest.json (SHA256)
- BT結果出力: backtest_result.json + data/validation/validation_report.json

### Integration Points
- `src/pipelines/training_pipeline.py`: BT実行時にOOF予測を `data/oof/oof_predictions.parquet` に保存
- `scripts/run_backtest.py --years 2024 --train-window 4 --ensemble --betting-mode flat`: BT実行コマンド
- `scripts/run_ic_eval.py --oof-path data/oof/oof_predictions.parquet`: IC評価コマンド
- `scripts/run_gpd.py --models-dir data/models-backtest`: GPD診断コマンド
- `python scripts/freeze_feature_manifest.py`: Manifest凍結コマンド
- `python -m pytest tests/test_post_race_leakage.py -v`: 漏洩テストコマンド

</code_context>

<specifics>
## Specific Ideas

- BT対象が2024年単年のため、実行時間は~41分（manifestなし）と短い。全検証ステップを含めても1-2時間以内で完了可能
- v1.6のROI 85.7%は2023/2024/2025の3年平均。2024年単独のROIは異なる可能性があるため、厳密な比較には注意が必要
- GPD診断はBT学習済みモデル（data/models-backtest/）を使用。学習直後に実行するのが最も効率的
- IC値の記録はJSON出力されるため、MLflowにも同時ログ記録可能（Phase 30の機能）

</specifics>

<deferred>
## Deferred Ideas

- Optuna戦略パラメータ最適化 — 将来フェーズで実行。Phase 34ではmanifestなしのデフォルトパラメータで検証
- v1.6ベースラインIC値の取得 — OOF予測が未保存のため不可能。v1.7 IC値を新ベースラインとする
- 2023/2025年度の追加BT — 時間が許せば追加実行可能だが、Phase 34の最小スコープは2024年のみ

</deferred>

---

*Phase: 34-Validation and Manifest Update*
*Context gathered: 2026-05-18*
