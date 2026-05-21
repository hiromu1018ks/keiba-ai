# Phase 28: Validation & Freeze - Context

**Gathered:** 2026-05-15
**Status:** Ready for planning

<domain>
## Phase Boundary

全特徴量変更（Phase 23-27）を統合したマルチ年度バックテストでROI改善を確認し、特徴量セットを凍結する。v1.6マイルストーンの最終検証フェーズ。

**In scope:**
- 統合マルチ年度バックテスト（2023/2024/2025）の実行
- pytest全テスト通過確認（1,392+テスト）
- Feature importance再計算（analyze_feature_importance.py --all-models）
- FEATURE_COLS凍結: JSON manifest + SHA256 hash（モデル毎に記録）
- ROI改善幅の記録（v1.5ベースライン84.4%からの差分）
- 結果レポート・ドキュメント更新

**Out of scope:**
- モデル再学習・ハイパーパラメータ調整
- Optuna戦略パラメータ再最適化（既存manifestを使用）
- WF検証（~4時間、別タスク）
- 複勝/ワイドモデルの変更
- 新しい特徴量の追加
- バックテスト結果に基づく追加改善Phase

</domain>

<decisions>
## Implementation Decisions

### バックテスト構成
- **D-01:** マルチ年度3年テスト（2023/2024/2025）で検証する。`--train-window 4`で学習。最も信頼性が高い（~3時間）
- **D-02:** 既存のstrategy_manifest(data/strategy_manifest.json)をそのまま使用する。新特徴量追加後のOptuna再最適化は行わない
- **D-03:** バックテストフラグ: `--ensemble --calibration-bt --report --strategy-manifest data/strategy_manifest.json`。Phase 25 D-04と同じ構成
- **D-04:** 完全なバックテストコマンド: `run_backtest.py --years 2023 2024 2025 --train-window 4 --ensemble --calibration-bt --report --strategy-manifest data/strategy_manifest.json`

### ROI評価基準
- **D-05:** ROIの絶対値100%到達を目標とするが、100%未達でも改善幅ベースで記録して完了とする。「v1.5: 84.4% → v1.6: XX% (+Y.Ypp)」の形式
- **D-06:** ROI未達時は追加チューニングや追加Phaseを行わず、結果を記録してPhase 28を完了する。次マイルストーンでの改善に委ねる

### 特徴量凍結方法
- **D-07:** ParameterFreezeProtocol（Phase 13）のパターンを踏襲: JSON manifest + SHA256 hash。sort_keys=True + indent=2で決定論的
- **D-08:** SHA256 hashは各モデルのFEATURE_COLS毎に記録する（AbilityModel, WinTwoStage, Place HIT/RETURN, EV等）

### 検証範囲
- **D-09:** pytest全テスト通過確認 + マルチ年度バックテスト + Feature importance再計算の3本柱
- **D-10:** Feature importance再計算はPhase 23の監査スクリプト（`analyze_feature_importance.py --all-models`）を使用する
- **D-11:** WF検証（~4時間）はPhase 28のスコープ外

### Claude's Discretion
- バックテスト結果の具体的な分析・解釈
- Feature importanceの結果に基づく推奨事項の記述
- 凍結manifestファイルの出力パス
- テスト結果レポートのフォーマット
- ROADMAP.md/PROJECT.mdの更新内容

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### バックテスト・ROI検証
- `scripts/run_backtest.py` — バックテストCLI。--years, --train-window, --ensemble, --calibration-bt, --report, --strategy-manifest
- `src/backtest/engine.py` — BacktestEngine。run()でフルBT実行
- `src/backtest/validation_report.py` — generate_validation_report() ROI PASS/FAIL判定
- `data/strategy_manifest.json` — Optuna最適化済み16次元パラメータmanifest

### Feature Importance監査
- `scripts/analyze_feature_importance.py` — feature importance監査CLI。--all-modelsで全モデル計算
- `feature_importance_report.json` — 直近のimportanceレポート出力

### FEATURE_COLS定義（凍結対象）
- `src/models/stage1_ability_model.py:28-148` — AbilityModel.FEATURE_COLS (~121特徴量)
- `src/models/two_stage_return_model.py:48-156` — WinTwoStageModel.FEATURE_COLS (~109特徴量)
- `src/models/two_stage_return_model.py:289-340` — PlaceTwoStageModel.HIT_FEATURE_COLS
- `src/models/two_stage_return_model.py:345-400` — PlaceTwoStageModel.RETURN_FEATURE_COLS
- `src/models/ev_correction_model.py:151-` — EVCorrectionModel.FEATURE_COLS
- `src/models/ev_correction_model.py:405-` — PlaceEVCorrectionModel.FEATURE_COLS
- `src/models/conformal_ev_model.py:81-` — ConformalEVModel.FEATURE_COLS
- `src/models/regime_detector.py:49-` — RegimeDetector.FEATURE_COLS
- `src/models/market_model.py:21-` — MarketModel.FEATURE_COLS

### ParameterFreezeProtocol（凍結パターン参照）
- `src/domain/models.py` — TrainedModelsV5, SubmodelSet定義
- `data/strategy_manifest.json` — 既存manifestパターン（SHA256 sort_keys=True + indent=2）

### テスト
- `tests/` — 全テスト（81ファイル、1,392+テスト、mock使用、DB不要）

### 前フェーズのCONTEXT（決定の連続性）
- `.planning/phases/27-feature-interactions/27-CONTEXT.md` — Phase 27決定（交互作用・TE）
- `.planning/phases/26-everydb2-new-features/26-CONTEXT.md` — Phase 26決定（血統・相対・mining・record）
- `.planning/phases/25-quick-win-wire-existing/25-CONTEXT.md` — Phase 25決定（12特徴量配線）
- `.planning/phases/24-feature-audit-pruning/24-CONTEXT.md` — Phase 24決定（Tier分類、監査パターン）
- `.planning/phases/23-safety-gate/23-CONTEXT.md` — Phase 23決定（漏洩防止、監査スクリプト）

### 要件・進捗
- `.planning/REQUIREMENTS.md` — v1.6全要件。Phase 28は全Prior Phase出力の検証
- `.planning/ROADMAP.md` — Phase 28 Success Criteria
- `.planning/PROJECT.md` — プロジェクト概要、Core Value（ROI 100%超え）

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **ParameterFreezeProtocol**: JSON manifest + SHA256 hashパターン。Phase 13で確立済み。sort_keys=True + indent=2でdeterministic。特徴量凍結にそのまま適用可能
- **analyze_feature_importance.py**: Phase 23で構築した監査スクリプト。--all-modelsで全モデルのpermutation+gain重要度を計算。新特徴量の効果確認に使用
- **run_backtest.py**: マルチ年度対応済み（--years + --train-window）。--strategy-manifest + --calibration-bt + --report全対応
- **BacktestResult**: ROI、bankroll curve、bet historyを含む結果オブジェクト

### Established Patterns
- **マルチ年度バックテスト**: `--years 2023 2024 2025 --train-window 4` でPhase 22/v1.4で実績あり
- **pytest全テスト**: `python -m pytest tests/ -v` で1,392+テストがDB不要で実行可能
- **FEATURE_COLS list[str]**: モデルクラスに定義。len()で特徴量数を確認可能
- **JSON manifest**: sort_keys=True + indent=2でdeterministicなJSON。hashlib.sha256でhash生成

### Integration Points
- **バックテスト実行**: run_backtest.py → BacktestEngine.run() → BacktestResult
- **Feature importance**: analyze_feature_importance.py → 各モデルのpermutation+gain重要度 → JSON/CSV出力
- **特徴量凍結**: 各モデル.FEATURE_COLS → JSON manifest → SHA256 hash → ファイル出力
- **結果記録**: ROADMAP.md更新（Phase 28完了記録）+ PROJECT.md更新（ROI結果）

</code_context>

<specifics>
## Specific Ideas

- ユーザーは一貫して「ベストプラクティスを追求」「実装難易度は問わない」方針
- ROI改善は絶対値100%到達が目標だが、改善幅ベースで評価する（16pp改善は大きな目標）
- Optuna再最適化は行わない（既存manifestを信頼する判断）
- WF検証は別タスク（~4時間、PostgreSQL環境必要）
- Stage1 ~121特徴量、Stage2 ~109特徴量はプロジェクト史上最大の特徴量セット
- PostgreSQL環境依存のバックテスト実行はユーザーがローカルで行う（CI不可）

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 28-Validation & Freeze*
*Context gathered: 2026-05-15*
