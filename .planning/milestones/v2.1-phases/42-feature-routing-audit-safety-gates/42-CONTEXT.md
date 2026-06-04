# Phase 42: Feature Routing Audit & Safety Gates - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

v2.1マイルストーンの最終安全確認フェーズ。Phase 39-41で導入したMarketAwareWinCalibrator + RaceLevelRankerについて、(1) キャリブレータ/ランカー特徴量がMarketModel/RaceQualityScreenerに漏洩していないことを監査し、(2) OOF健全性を新コンポーネント込みで検証し、(3) 配備ゲート条件が全て通過するまで新パイプラインがベースラインを置き換えないことを保証する。

**In scope:** SAF-01（特徴量ルーティング監査）、SAF-02（OOF健全性検証）、SAF-03（配備ゲート条件の評価と文書化）。
**Out of scope:** 配備ゲート自動デプロイ判定（DEP-01, v2.2+）、MarketAwareWinCalibrator/RaceLevelRanker自体の実装変更、新規モデル/特徴量の追加、ROI最適化。

</domain>

<decisions>
## Implementation Decisions

### 特徴量ルーティング監査 (SAF-01)

- **D-01:** 両方実装: ユニットテストでfail-fast検出（CI常時保証）+ `scripts/run_feature_routing_audit.py` でJSON/Markdown監査レポート生成。最終的な安全保証はテストで担保。監査スクリプトは検証・レビュー用。
- **D-02:** 監査レジストリを `src/audit/feature_routing_registry.py` に単一定義。`FORBIDDEN_CALIBRATOR_FEATURES` / `FORBIDDEN_RANKER_FEATURES` / `CRITICAL_TARGET_MODELS` / `ADVISORY_TARGET_MODELS` を含む。テストとスクリプトが同じレジストリを参照。実際のモデルFEATURE_COLSとの差分テストで更新漏れも検出する。
- **D-03:** 必須監査対象（fail-fast）= MarketModel + RaceQualityScreener。参考監査対象（warning/report）= その他モデル（Stage1, Win, Place, Wide, EVCorrection, Regime等、FEATURE_COLSが容易に取得できるもの）。fail-fast対象以外の交差はwarning扱い。
- **D-04:** 監査スクリプト出力: JSON + Markdown。JSON: CI/自動判定用（モデル別status, forbidden_intersections, warning_intersections, checked_feature_count）。Markdown: レビュー用（fail-fast/warning分離表示、レジストリバージョン、実行日時、FEATURE_COLS取得元を記録）。

### OOF健全性検証 (SAF-02)

- **D-05:** 二層構成: (1) CI用mockベース検証 — Phase 39-40新規OOF生成パス（MarketAwareWinCalibrator/RaceLevelRanker）をmockで検証、OOFHealthValidatorがPASSすることを確認 + TrainingPipeline OOF保存ポイントからOOFHealthValidator呼び出しまでの軽量統合テスト、(2) 手動/nightly監査コマンド — フルE2Eバックテスト級の検証はCI必須にせず手動実行用。
- **D-06:** Phase 37のOOFHealthValidator既存anomaly定義を共通基盤として再利用。Phase 39/40固有チェックはartifact profile/pluginとして分離（OOFHealthValidator本体に汎用性の低いロジックを直書きしない）。
- **D-07:** MarketAwareWinCalibrator artifact profile: 確率NaN/inf検出、範囲[0,1]確認、race_id単位sum-to-1.0確認、p_win_pred混入禁止、fold列必須。
- **D-08:** RaceLevelRanker artifact profile: investment_score/component scoreのNaN/inf検出、race_id内順位の決定性確認、fold列必須。

### 配備ゲート条件 (SAF-03)

- **D-09:** 独立ゲート評価器 `src/backtest/deployment_gates.py` に `DeploymentGateEvaluator` を実装。Phase 41の `shadow_comparison_result.json` と manifest を入力として PASS/FAIL/WARN を出力。RacePredictorには閾値判定を入れない。
- **D-10:** ゲート条件と閾値は `GatePolicy` dataclass または小さなJSONで固定。v2.2で自動デプロイ判定へ拡張可能な構造にする。
- **D-11:** GatePolicyの具体条件:
  - 確率品質（非悪化必須）: Brier shadow <= baseline, logloss shadow <= baseline, ECE shadow <= baseline（各fold年度 + overall）。小数誤差 tolerance ~1e-6 許容。
  - actual/predicted ratio: 各年度でbaselineより悪化しないことをWARN以上の条件。
  - ベット数維持: shadow_bet_count >= baseline_bet_count * 0.95。
  - アーティファクト再現性: shadow/baseline manifest SHA256一致、feature_version/schema_hash一致。
  - 診断: OOFHealthValidator PASS、FeatureRoutingAudit PASS、shadow comparison manifest complete（全て必須）。
  - 非ゲート（レポートのみ）: Selection agreement（診断指標）, ROI（レポートのみ）。
- **D-12:** DeploymentGateEvaluatorは判定レポートのみを出力し、モデルやdeployment_statusを自動変更しない。CLI実行時はFAILなら非ゼロ終了コードを返す。RacePredictorは既存のfeature flag + deployment_statusを尊重するだけ。
- **D-13:** 自動shadow_only化はv2.2以降に延期。

### Claude's Discretion

- 監査レジストリの具体的な特徴量リスト内容（Phase 39/40のモデル定義から正確に抽出）
- OOF artifact profileの具体的な実装方法（OOFHealthValidatorへの統合方法）
- DeploymentGateEvaluatorの内部メソッド・データフロー設計
- テスト構造・命名（既存規約に従う）
- GatePolicy dataclassのフィールド設計

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Components Being Audited
- `src/models/market_aware_win_calibrator.py` — Phase 39 calibrator. FEATURE_COLS definition for forbidden feature set extraction.
- `src/models/race_level_ranker.py` — Phase 40 ranker. FEATURE_COLS definition for forbidden feature set extraction.
- `src/models/market_model.py` — MarketModel. CRITICAL_TARGET_MODELS member. FEATURE_COLS to audit against forbidden set.
- `src/models/race_quality_screener.py` — RaceQualityScreener. CRITICAL_TARGET_MODELS member. FEATURE_COLS to audit against forbidden set.

### Validation Infrastructure
- `src/validation/oof_health.py` — OOFHealthValidator (Phase 37). Base anomaly detection + fail-fast + SHA256 manifest. Extend with artifact profiles for Phase 39/40.
- `src/backtest/shadow_comparison.py` — ShadowComparisonFramework (Phase 41). Produces shadow_comparison_result.json consumed by DeploymentGateEvaluator.
- `src/backtest/deployment_gates.py` — NEW file. DeploymentGateEvaluator + GatePolicy.

### Audit Infrastructure (NEW)
- `src/audit/feature_routing_registry.py` — NEW file. FORBIDDEN_CALIBRATOR_FEATURES, FORBIDDEN_RANKER_FEATURES, CRITICAL_TARGET_MODELS, ADVISORY_TARGET_MODELS.
- `scripts/run_feature_routing_audit.py` — NEW script. Reads registry, audits all models, outputs JSON + Markdown.

### Pipeline Integration Points
- `src/backtest/race_predictor.py` — RacePredictor. Feature flag + deployment_status check pattern (lines 269-277 MAWC, 279-285 ranker). No changes to RacePredictor for gate logic.
- `src/pipelines/training_pipeline.py` — TrainingPipelineV5. OOF generation paths. OOFHealthValidator save-point hooks.
- `src/domain/models.py` — SubmodelSet (lines 234-273). market_aware_win_calibrator + win_race_level_ranker fields.

### Requirements
- `.planning/REQUIREMENTS.md` — SAF-01, SAF-02, SAF-03 (Phase 42 requirements).
- `.planning/ROADMAP.md` — Phase 42 success criteria (3 items).
- `.planning/PROJECT.md` — Key Decisions (配備条件=確率品質, selection agreement = diagnostic not gate).

### Prior Phase Context
- `.planning/phases/39-marketawarewincalibrator/39-CONTEXT.md` — Phase 39 calibrator architecture, ~51 feature dimensions, OOF generation.
- `.planning/phases/40-race-level-ranker/40-CONTEXT.md` — Phase 40 ranker architecture, feature sets, shadow mode pattern.
- `.planning/phases/41-shadow-comparison-framework/41-CONTEXT.md` — Phase 41 comparison framework, shadow_comparison_result.json format, manifest structure.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **OOFHealthValidator** (`src/validation/oof_health.py`): Phase 37 validation infrastructure. Fail-fast + anomaly detection + SHA256 manifest. Extend with artifact profiles for MAWC and Ranker.
- **ShadowComparisonFramework** (`src/backtest/shadow_comparison.py`): Phase 41 comparison runner. Produces shadow_comparison_result.json with per-fold/year metrics, surface/odds band breakdown. This is the primary input for DeploymentGateEvaluator.
- **FEATURE_COLS pattern**: Every model defines FEATURE_COLS as a class constant. Audit can introspect these directly.
- **GateKeeper / deployment_status pattern**: Existing RacePredictor checks is_trained + deployment_status before using new components. No changes needed.

### Established Patterns
- **Mock-based testing**: All 2,056 tests use unittest.mock. No DB dependency. New audit/OOF tests follow same pattern.
- **Class-based test organization**: Test<ComponentName> pattern. New tests: TestFeatureRoutingAudit, TestDeploymentGateEvaluator.
- **Script entry points**: scripts/ directory for CLI tools. run_feature_routing_audit.py follows run_shadow_comparison.py pattern.
- **JSON + Markdown output**: Phase 41 shadow comparison uses this pattern. Audit script follows same convention.
- **GatePolicy-style config**: Phase 38's InvestmentFeatureSpec uses frozen dataclass. GatePolicy follows same pattern.

### Integration Points
- **src/audit/** — NEW directory for audit infrastructure. feature_routing_registry.py is the single source of truth for forbidden feature sets.
- **src/backtest/deployment_gates.py** — NEW file. Reads shadow_comparison_result.json, applies GatePolicy, outputs gate evaluation.
- **src/validation/oof_health.py** — EXTEND with artifact profile registry. New profiles for MAWC and Ranker without modifying core validator logic.
- **scripts/run_feature_routing_audit.py** — NEW CLI script. Uses same audit registry as tests.

</code_context>

<specifics>
## Specific Ideas

- Audit registry as single source of truth: `src/audit/feature_routing_registry.py` with FORBIDDEN_CALIBRATOR_FEATURES (~51 features from Phase 39), FORBIDDEN_RANKER_FEATURES (~28 features from Phase 40), CRITICAL_TARGET_MODELS (MarketModel, RaceQualityScreener), ADVISORY_TARGET_MODELS (other models).
- Diff test: test verifies registry matches actual MarketAwareWinCalibrator.FEATURE_COLS / RaceLevelRanker.FEATURE_COLS — catches stale registries.
- GatePolicy frozen dataclass: brier_tolerance=1e-6, logloss_tolerance=1e-6, ece_tolerance=1e-6, bet_count_ratio_threshold=0.95, require_oof_pass=True, require_audit_pass=True, require_manifest_complete=True.
- DeploymentGateEvaluator: evaluate(shadow_result_path, manifest_path) -> GateEvaluationResult with overall PASS/FAIL/WARN + per-condition details. CLI exits non-zero on FAIL.
- OOF artifact profiles as plugin-like objects: CalibratorArtifactProfile, RankerArtifactProfile, each with validate(oof_df) -> list[Anomaly]. Registered with OOFHealthValidator.

</specifics>

<deferred>
## Deferred Ideas

- **自動デプロイ判定 (DEP-01):** DeploymentGateEvaluatorのPASS判定に基づくdeployment_status自動切り替え。v2.2+で実装。Phase 42では構造のみ拡張可能にする。
- **FAIL時自動shadow_only化:** RacePredictorがgate FAILを検知して自動的にshadow_onlyにする機能。v2.2+で実装。

</deferred>

---

*Phase: 42-Feature Routing Audit & Safety Gates*
*Context gathered: 2026-05-28*
