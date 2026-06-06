---
phase: 52-shared-feature-builder-consistency
verified: 2026-06-06T05:18:57Z
status: gaps_found
score: 7/9 must-haves verified
overrides_applied: 0
gaps:
  - truth: "run_paper_trading.py の predict/diagnose/dry-run モードが FeatureBuilder を使用せず、旧来の FeatureEngine + 手動エンリッチメントを使用しているため、BT と PT (スクリプト実行パス) の特徴量不一致が残存する"
    status: failed
    reason: "_run_predict (L463-531), _run_diagnose (L898), _run_dry_run (L1211-1277) はいずれも FeatureBuilder を使用していない。PaperPredictor.setup() は FeatureBuilder を使用するよう更新されたが、_run_predict は EveryDB2 から直接ロードする別コードパスを実行する。SC1「BT/PT/TrainingPipeline が同じ関数を呼び出す」は、スクリプト経由の PT 実行パスで満たされていない。REVIEW.md CR-02 として既に指摘済み。"
    artifacts:
      - path: "scripts/run_paper_trading.py"
        issue: "L463-531, L898, L1211-1277 が FeatureEngine + 手動 BloodlineFeatures/SireFeatures/PaceAptitudeFeatures/CourseFeatures を使用"
    missing:
      - "_run_predict(), _run_diagnose(), _run_dry_run() の特徴量生成を FeatureBuilder.build_for_training() に置換"
  - truth: "FeatureBuilder._enrich_features() に BloodlineFeatures が明示的に含まれていない (FeatureEngine.build_all() 経由で暗黙的に実行されるが、13モジュールの列挙に欠落)"
    status: partial
    reason: "FeatureEngine.build_all() の Group B で BloodlineFeatures が実行されるため、build_for_training/build_for_inference いずれも blood_* カラムを含む。実害はないが、_enrich_features() の13モジュール列挙に (o) BloodlineFeatures が含まれておらず、コードの意図が不明瞭。REVIEW.md CR-01 として指摘済み。"
    artifacts:
      - path: "src/features/feature_builder.py"
        issue: "_enrich_features() の docstring と実装が13モジュールと宣言しているが BloodlineFeatures は FeatureEngine.build_all() 経由で暗黙実行"
    missing:
      - "_enrich_features() のコメント/docstring に BloodlineFeatures が FeatureEngine.build_all() で実行される旨を明記、または _enrich_features() に明示的に追加"
---

# Phase 52: Shared Feature Builder & Consistency 検証レポート

**Phase Goal:** BT と PT と TrainingPipeline が同一の特徴量生成関数を呼び出し、パイプラインの同一実装・同一設定契約が検証可能であること
**Verified:** 2026-06-06T05:18:57Z
**Status:** gaps_found
**Re-verification:** No (initial verification)

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | BacktestEngine.prepare_data() が FeatureBuilder.build_for_training() に委譲し、同じ形の出力を返す | VERIFIED | engine.py L681-691: FeatureBuilder import + build_for_training() 呼出 |
| 2 | BacktestEngine.run() 内部パスが FeatureBuilder に委譲する | VERIFIED | engine.py L946-956: FeatureBuilder import + build_for_training() 呼出 |
| 3 | PaperPredictor.setup() が FeatureBuilder.build_for_inference() に委譲する (7ギャップ解消) | VERIFIED | predictor.py L53-107: FeatureBuilder + FeatureState による surface 毎ビルド |
| 4 | RacePredictor.predict() に track_condition/interaction/relative の重複計算が残存しない | VERIFIED | grep 結果: compute_track_condition_features, compute_interaction_features, compute_relative_features の呼出が全て 0 件 |
| 5 | TrainingPipeline._train_submodel() が特徴量構築コードを含まず track_stats 計算のみ | VERIFIED | training_pipeline.py L810-833: 13モジュールの import/compute が全て除去、track_stats 計算のみ残存 |
| 6 | 4呼び出し元にインライン特徴量構築コードが残存しない | VERIFIED | engine.py: HorseHistoryFeatures 等の import なし (キャッシュクリアのみ)。predictor.py: 旧モジュール import なし |
| 7 | run_paper_trading.py の predict/diagnose/dry-run モードが FeatureBuilder を使用する | FAILED | L463-531, L898, L1211-1277: FeatureEngine + 手動 BloodlineFeatures/SireFeatures/PaceAptitude/CourseFeatures を使用 |
| 8 | FeatureBuilder が13エンリッチメントモジュールを _train_submodel と同一順序で実行する | VERIFIED | feature_builder.py L216-418: (a) HorseHistory -> (b) transforms -> (c) PaceAptitude -> ... -> (n) JockeyTrainerCombo の順序 |
| 9 | FeatureBuilder._enrich_features() のモジュール列挙に BloodlineFeatures が含まれる | PARTIAL | BloodlineFeatures は FeatureEngine.build_all() Group B で暗黙実行。_enrich_features() の列挙には含まれていないが、出力には blood_* カラムが含まれる |

**Score:** 7/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/features/feature_manifest.py` | FeatureManifest, FeatureState, FeatureBuildResult | VERIFIED | 3つの frozen dataclass + compute_hash() + from_submodel_set() 実装済み |
| `src/features/feature_builder.py` | FeatureBuilder (build_for_training/build_for_inference) | VERIFIED | 13モジュール統合 + POST_RACE 除去 + manifest 生成 |
| `src/features/pit_registry.py` | PITModuleRegistry | VERIFIED | 13モジュール PIT 契約登録 + verify_pit_compliance() |
| `src/backtest/engine.py` | FeatureBuilder 委譲 | VERIFIED | prepare_data() L681, run() L946 |
| `src/backtest/race_predictor.py` | 重複除去 | VERIFIED | compute_track_condition/interaction/relative の呼出 0件 |
| `src/paper_trading/predictor.py` | FeatureBuilder 委譲 | VERIFIED | setup() L53: build_for_inference() 使用 |
| `src/pipelines/training_pipeline.py` | FeatureBuilder 委譲 + _train_submodel 簡略化 | VERIFIED | run() L379: build_for_training() 使用。_train_submodel: 13モジュール除去 |
| `src/features/data_cutoff_manifest.py` | DataCutoffManifest | VERIFIED | verify()/verify_strict()/from_config() 実装 |
| `src/features/pipeline_consistency.py` | PFPVerifier | VERIFIED | freeze()/verify()/get_frozen_state() 実装 |
| `src/features/session_manifest.py` | SessionManifest, get_code_version, write_session_manifest | VERIFIED | セッション記録 + git dirty 検出 + アトミック書き込み |
| `scripts/run_paper_trading.py` | 3点検証 + --allow-dirty | VERIFIED | startup (L345-416), --allow-dirty (L147) |
| `scripts/run_paper_trading.py` (_run_predict) | FeatureBuilder 使用 | FAILED | L463: FeatureEngine + 手動エンリッチメントを使用 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| src/backtest/engine.py | src/features/feature_builder.py | import FeatureBuilder | WIRED | L681, L946 |
| src/paper_trading/predictor.py | src/features/feature_builder.py | import FeatureBuilder | WIRED | L53 |
| src/pipelines/training_pipeline.py | src/features/feature_builder.py | import FeatureBuilder | WIRED | L379 |
| src/features/feature_builder.py | src/features/feature_engine.py | FeatureEngine.build_all() | WIRED | L193-201 |
| src/features/feature_builder.py | src/features/sire_features.py | SireFeatures.compute_batch | WIRED | L276-300 |
| src/features/feature_builder.py | src/features/track_condition_features.py | compute_track_condition_features | WIRED | L332-357 |
| src/features/pipeline_consistency.py | src/backtest/parameter_freeze_protocol.py | ParameterFreezeProtocol | WIRED | L14, L46 |
| src/features/session_manifest.py | src/features/feature_manifest.py | FeatureManifest hash | WIRED | set_model_identity() に manifest_hash |
| scripts/run_paper_trading.py | src/features/session_manifest.py | get_code_version, SessionManifest | WIRED | L349-416 |
| scripts/run_paper_trading.py | src/features/data_cutoff_manifest.py | DataCutoffManifest.from_config | WIRED | L370 |
| scripts/run_paper_trading.py | src/features/pipeline_consistency.py | PFPVerifier | WIRED | L396 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|---------------------|--------|
| FeatureBuilder.build_for_training() | feat_df | FeatureEngine.build_all() + 13 enrichment modules | Real modules invoked, merges applied | FLOWING |
| FeatureBuilder.build_for_inference() | feat_df (POST_RACE stripped) | Same as training + FeatureState.track_stats | track_stats from SubmodelSet | FLOWING |
| PFPVerifier.verify() | checks dict | ParameterFreezeProtocol + FeatureManifest.compute_hash() | Re-computes hash on verify | FLOWING |
| SessionManifest.to_dict() | manifest dict | get_code_version() + set_model_identity() | Git subprocess + stored fields | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All Phase 52 imports succeed | python -c "from features.feature_manifest import ..." | "All imports OK" | PASS |
| FeatureManifest hash determinism | python -c "assert m1.compute_hash() == m2.compute_hash()" | Passed | PASS |
| FeatureState fail-fast on None track_stats | python -c "FeatureState.from_submodel_set(sub, '1.0')" | ValueError with "TRN-04" | PASS |
| 46 unit tests pass | pytest tests/test_feature_manifest.py tests/test_feature_builder.py tests/test_pipeline_consistency.py -v | 46 passed in 1.62s | PASS |
| ruff lint on new files | ruff check src/features/*.py | All checks passed | PASS |
| --allow-dirty flag exists | grep -c "allow-dirty" scripts/run_paper_trading.py | 3 matches | PASS |

### Probe Execution

Step 7c: SKIPPED (no probe scripts defined for this phase)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PLN-01 | 52-01, 52-02 | 共通特徴量ビルダー抽出、7ギャップ解消 | SATISFIED (partial) | FeatureBuilder が 13モジュールを統一。PaperPredictor.setup() は FeatureBuilder 使用。ただし run_paper_trading.py の _run_predict/diagnose/dry-run は旧パス |
| PLN-02 | 52-03 | PT 実行記録に MLflow run ID/学習期間/コードハッシュ/manifest hash 保存 | SATISFIED | SessionManifest + run_paper_trading.py L403-416 に統合 |
| PLN-03 | 52-03 | データカットオフ検証 (予測日以降のデータ使用防止) | SATISFIED | DataCutoffManifest + run_paper_trading.py L366-382 に統合 |
| PLN-04 | 52-03 | PFP パラメータ不変性検証 | SATISFIED | PFPVerifier + run_paper_trading.py L384-401 に統合 |

**Orphaned requirements:** なし (PLN-01~04 は全て PLAN frontmatter で宣言)

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none in Phase 52 files) | - | - | - | 新規ファイルに TBD/FIXME/XXX なし |

### Human Verification Required

なし (46件のユニットテストが全て通過、インポートチェック・lint チェック通過)

### Gaps Summary

Phase 52 は FeatureBuilder による特徴量生成統一の基盤を構築した。FeatureManifest/FeatureState/FeatureBuildResult の各 dataclass、13エンリッチメントモジュールを統合する FeatureBuilder、PIT 契約レジストリ、DataCutoffManifest、PFPVerifier、SessionManifest が全て実装され、46件のテストが通過している。

**残存ギャップ (2件):**

1. **run_paper_trading.py の _run_predict/diagnose/dry_run が FeatureBuilder 未使用 (CR-02):** PaperPredictor.setup() は FeatureBuilder に移行したが、スクリプト内の3つの実行モード (predict, diagnose, dry_run) は旧来の FeatureEngine + 手動エンリッチメントを使用している。これは SC1「BT/PT/TrainingPipeline が同じ関数を呼び出す」の部分的な未達成。REVIEW.md で CR-02 として指摘済み。

2. **FeatureBuilder._enrich_features() に BloodlineFeatures の明示的列挙がない (CR-01):** FeatureEngine.build_all() 経由で暗黙的に実行されるため実害はないが、13モジュールの列挙に含まれておらず、コードの意図が不明瞭。REVIEW.md で CR-01 として指摘済み。

ギャップ #1 は SC1 の完全達成を妨げる BLOCKER。ギャップ #2 は実害がないものの、ドキュメント整合性の観点で WARNING。

---

_Verified: 2026-06-06T05:18:57Z_
_Verifier: Claude (gsd-verifier)_
