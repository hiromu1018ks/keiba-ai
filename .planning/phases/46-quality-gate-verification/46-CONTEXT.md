# Phase 46: Quality Gate Verification - Context

**Gathered:** 2026-05-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 45で実装されたMAWC保守的再学習CLIを本番データで実行して保守的variantを生成し、4つの品質ゲート（OOFHealthValidator, FeatureRoutingAudit, DeploymentGateEvaluator, ROI回復傾向）を包括的に検証するv2.2マイルストーン最終品質保証フェーズ。

Phase 46は**判定フェーズ**であり、モデル修正・C値再選択・閾値調整は行わない。例外は実行系バグ（ファイルパス、manifest不整合、CLI引数ミス、保存漏れ）の修正のみ。

**In scope:**
- QUAL-01: OOFHealthValidator PASS確認
- QUAL-02: FeatureRoutingAudit PASS確認（50+28禁止特徴量CI安全監査）
- QUAL-03: DeploymentGateEvaluator PASS確認（確率品質・ベット数維持・再現性・診断の4ゲート）
- QUAL-04: ROI回復傾向確認（Brier/logloss/ECE非悪化、actual/predicted非悪化、bet_count維持）
- v2.2マイルストーン完了証明（Phase 43-46 requirement traceability）
- orchestration CLI (`scripts/run_phase46_quality_gates.py`) の作成
- runbook (`.planning/phases/46-quality-gate-verification/46-RUNBOOK.md`) の作成

**Out of scope:**
- モデル再学習・C値再探索（Phase 45 or Phase 45b）
- Selection閾値変更
- Ranker/OBF変更
- 新特徴量追加
- レジーム別分析・パラメータ調整
- ROI単独最適化

</domain>

<decisions>
## Implementation Decisions

### 実行フロー設計

- **D-01:** 2段階実行フローを採用。
  - **Stage 1:** `scripts/run_mawc_conservative_retrain.py` を実行して `data/models-backtest-mawc-conservative/` を生成。manifest.json / retrain_summary.md / HTML report を確認。全surface/yearでdeployed候補が存在しない、またはfavorite band guardがFAILなら、Stage 2のShadow Comparisonは実行せずPhase 46をBLOCKED/FAILとして記録。
  - **Stage 2:** Stage 1が通った場合のみ実行。推奨順序: FeatureRoutingAudit → OOFHealthValidator → Shadow Comparison → Shadow Diagnosis → DeploymentGateEvaluator → Final summary。全自動、中間成果物全保存、途中FAILで停止。skip/resume可能設計。

- **D-02:** Stage 2の実装形態は orchestration CLI + runbook の両方作成。
  - `scripts/run_phase46_quality_gates.py` — Stage 2の既存CLI/関数を順次呼び出す。各ステップの成果物パスとPASS/FAILをJSON/Markdownに記録。途中FAIL時は停止するが失敗理由と既存成果物を残す。再実行時に既存成果物を検出してskip/resume可能。
  - `.planning/phases/46-quality-gate-verification/46-RUNBOOK.md` — 各ステップを手動再現できるコマンド列。長時間実行のShadow Comparison単独再実行にも対応。トラブル時の判断基準を明記。
  - Phase 46 CLIは既存コンポーネントをラップするだけで、新しい判定ロジックを追加しない。

### ROI判定フレームワーク

- **D-03:** 3ラベル分離判定フレームワーク。
  - **Quality Gate:** PASS/FAIL（Primary conditions: Brier/logloss/ECE baseline比非悪化、actual/predicted ratio非悪化、shadow bet_count >= baseline * 0.95）
  - **ROI Trend:** recovered（90%以上）/ weak_recovery（87.8%-90%）/ not_recovered（87.8%未満）
  - **Deployment:** deployable / not_deployable / manual_review
  - ROI 100%超えはv2.2の目標だがPhase 46の必須PASS条件ではない。
  - ROIが87.8%未満でも品質ゲート全PASSなら「品質ゲートPASS・ROI回復未達」として記録し、配備判断はmanual_review。
  - ROIは配備ゲートではなく診断指標として扱う。品質ゲート判定とは分離。

### ゲートFAIL時対応

- **D-04:** Phase 46内ではリトライ・再探索・候補差し替えを一切しない。
  - Stage 1 FAIL → Shadow Comparison実行せず、Phase 46 FAIL/BLOCKEDとして記録。
  - Stage 2 FAIL → FAIL内容をnot_deployedとして記録して終了。
  - 既存C grid内の次点候補差し替えも不可。
  - 例外: ファイル欠損、CLI引数ミス、manifest pathミス等の実行系バグのみPhase 46内で修正・再実行可能。
  - モデル修正が必要な場合はPhase 45bまたは次フェーズで扱う。

### 成果物構造

- **D-05:** Phase 46成果物 + マイルストーン完了証明を生成。HTMLは新規必須成果物にしない。
  - **Phase 46成果物:**
    - `data/backtest/phase46_quality_gates/phase46_quality_gate_result.json`
    - `data/backtest/phase46_quality_gates/phase46_quality_gate_summary.md`
    - `.planning/phases/46-quality-gate-verification/46-RUNBOOK.md`
    - `.planning/phases/46-quality-gate-verification/46-VERIFICATION.md`
  - **検証入力/副産物:**
    - `data/models-backtest-mawc-conservative/manifest.json`
    - `data/models-backtest-mawc-conservative/retrain_summary.md`
    - `data/backtest/shadow_mawc_conservative/shadow_comparison_result.json`
    - `data/backtest/shadow_mawc_conservative/diagnosis/shadow_diagnosis_result.json`
    - `data/backtest/shadow_mawc_conservative/gates/deployment_gate_result.json`
  - **マイルストーン完了証明:**
    - `.planning/v2.2-MILESTONE-SUMMARY.md`
    - Phase 43-46 requirement traceability (DIAG-01~03, BISECT-01~02, FIX-01~02, QUAL-01~04) 最終状態一覧
    - 最終判定（deployable/not_deployed/manual_review）明記
    - ROIは参考指標、品質ゲート判定とは分離
  - 既存MAWC/Shadow/Diagnosis HTMLへのリンクをsummaryに記載。

### Claude's Discretion

- FeatureRoutingAudit/OOFHealthValidatorの具体的呼び出し方法（関数呼び出し vs CLI化）はplan時にコード確認して決定
- orchestration CLIのskip/resume判定ロジックの詳細設計
- Stage 2内の各ステップ間の成果物パス受け渡し設計
- phase46_quality_gate_result.json のスキーマ設計
- v2.2-MILESTONE-SUMMARY.md の構造設計
- テスト構造・命名（既存規約に従う）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### 品質検証インフラ（主入力）
- `src/validation/oof_health_validator.py` — OOFHealthValidator。QUAL-01で使用。fail-fast OOF検証 + SHA256 manifest。
- `src/audit/feature_routing_registry.py` — FeatureRoutingAuditRegistry。QUAL-02で使用。50+28禁止特徴量CI監査。
- `src/backtest/deployment_gates.py` — DeploymentGateEvaluator。QUAL-03で使用。4ゲート評価（確率品質、ベット数、再現性、診断）。
- `src/backtest/shadow_comparison.py` — ShadowComparisonFramework。Stage 2でbaseline vs mawc_conservative比較に使用。
- `src/backtest/shadow_diagnosis.py` — ShadowDiagnosis。Stage 2で保守的variant診断に使用。

### Phase 45 成果物（Stage 1入力）
- `scripts/run_mawc_conservative_retrain.py` — MawcConservativeRetrainer CLI。Stage 1で実行。
- `src/models/mawc_conservative_retrainer.py` — MawcConservativeRetrainer。保守的再学習エンジン。
- `data/oof/oof_predictions.parquet` — MAWC再学習用OOFデータ。
- `data/models-backtest/` — 既存学習済みモデル（Stage 1 source）。

### Pipeline Integration Points
- `src/backtest/race_predictor.py` — RacePredictor。MAWC/Rankerフラグ制御。
- `src/db/model_loader.py` — ModelLoader。load_from_dir()で年ディレクトリ読み込み。
- `scripts/run_shadow_comparison.py` — Phase 41 CLI。Stage 2で--shadow-root data/models-backtest-mawc-conservative指定。
- `scripts/run_shadow_diagnosis.py` — Phase 43 CLI。Stage 2で保守的variant診断。

### 前フェーズコンテキスト
- `.planning/phases/45-structural-fix/45-CONTEXT.md` — Phase 45設計。MAWC保守的再学習の全決定事項。
- `.planning/phases/44-roi-bisect/44-CONTEXT.md` — Phase 44ビセクション結果。MAWC劣化原因特定。
- `.planning/phases/43-shadow-diagnosis/43-CONTEXT.md` — Phase 43診断設計。セグメント定義。
- `.planning/phases/42-feature-routing-audit-safety-gates/42-CONTEXT.md` — Phase 42 GatePolicy定義。

### Requirements & Project
- `.planning/REQUIREMENTS.md` — QUAL-01, QUAL-02, QUAL-03, QUAL-04 (Phase 46 requirements)。
- `.planning/ROADMAP.md` — Phase 46 success criteria (5 items)。
- `.planning/PROJECT.md` — Key Decisions (配備条件=確率品質)。

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **ShadowComparisonFramework** (`src/backtest/shadow_comparison.py`): N-way variant対応済み。`--shadow-root data/models-backtest-mawc-conservative` で保守的variant比較可能。
- **DeploymentGateEvaluator** (`src/backtest/deployment_gates.py`): 4ゲート評価。既存GatePolicy定義をそのまま適用。
- **OOFHealthValidator** (`src/validation/oof_health_validator.py`): SHA256 manifest検証。関数呼び出し前提（CLI未確認）。
- **FeatureRoutingAuditRegistry** (`src/audit/feature_routing_registry.py`): 50+28禁止特徴量監査。`run_feature_audit()` 関数 or `scripts/run_feature_routing_audit.py` CLI。
- **ShadowDiagnosis** (`src/backtest/shadow_diagnosis.py`): 3ステップ段階的除外診断。CLI `scripts/run_shadow_diagnosis.py` 存在。

### Established Patterns
- **CLI + JSON/MD/HTML複数出力**: Phase 41-45パターン。JSON機械可読、MD人間可読、HTML視覚的。
- **Manifest + SHA256**: Phase 42/45パターン。成果物の完全性担保。
- **orchesration**: 新規パターン。既存CLI/関数を順次呼び出すwrapper CLI。

### Integration Points
- **Stage 1 入力**: data/models-backtest/ (既存モデル), data/oof/oof_predictions.parquet (OOFデータ)
- **Stage 1 出力**: data/models-backtest-mawc-conservative/{year}/ (保守的variant)
- **Stage 2 入力**: data/models-backtest-mawc-conservative/ (保守的variant), data/models-backtest/ (baseline)
- **Stage 2 出力**: data/backtest/shadow_mawc_conservative/ (Shadow成果物), data/backtest/phase46_quality_gates/ (Phase46成果物)
- **マイルストーン出力**: .planning/v2.2-MILESTONE-SUMMARY.md

</code_context>

<specifics>
## Specific Ideas

- Stage 1とStage 2は別スクリプトではなく、Phase 46 CLI内でstage=1/2フラグまたは自動判定で切り替える設計が適切。Stage 1結果（manifest）が既に存在すればStage 2からresume可能。
- skip/resume設計: 各ステップの出力ファイルが存在する場合は「既存成果物を検出、skip可能」としてログに記録。強制再実行フラグ(--force)も用意。
- bet_count閾値 0.95 (baseline比) はD-03のQuality Gate条件。shadow_selected_count >= baseline_selected_count * 0.95。
- Stage 2のShadow Comparison出力先は `data/backtest/shadow_mawc_conservative/` を推奨（Phase 41の `data/backtest/shadow/` と区別）。
- v2.2-MILESTONE-SUMMARY.md は Phase 43-46 の各VERIFICATION.md と CONTEXT.md を集約する位置づけ。各requirementの最終状態はVERIFICATION.mdから抽出可能。

</specifics>

<deferred>
## Deferred Ideas

- **Phase 45b (MAWC再調整)**: Phase 46でFAIL/not_deployedとなった場合、C値範囲拡大・特徴量構成変更等の追加修正フェーズ。
- **Ranker修正 (investment_score重み・閾値調整)**: Rankerはdormant。v2.3+で検討。
- **OddsBandFilter再学習・閾値調整**: v2.3+で検討。
- **デプロイゲート自動判定 (DEP-01)**: v2.3+に延期。
- **Optuna 19次元パラメータ最適化 (DEP-02)**: v2.3+に延期。
- **レジーム別分析・レジーム別パラメータ調整**: v2.3+で検討。
- **新特徴量追加**: v2.3+で検討。

</deferred>

---

*Phase: 46-Quality Gate Verification*
*Context gathered: 2026-05-31*
