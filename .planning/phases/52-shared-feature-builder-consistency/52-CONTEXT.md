# Phase 52: Shared Feature Builder & Consistency - Context

**Gathered:** 2026-06-06
**Status:** Ready for planning

<domain>
## Phase Boundary

BT と PT と TrainingPipeline が同一の特徴量生成関数を呼び出し、パイプラインの同一実装・同一設定契約が検証可能であること。

具体的には:
1. **特徴量生成の共有実装** — FeatureBuilder クラスを新設し、BT/PT/Train が同一関数を呼ぶ。7つのギャップ(Sire/PaceAptitude/Course/DamPedigree/Record/Mining/interaction)を一括解消
2. **パイプライン一貫性検証** — MLflow run ID・学習期間・コードハッシュ・feature manifest hash の記録と検証
3. **データカットオフ検証** — 予測日以降のデータが使用されていないことの二段階検証
4. **PFP不変性検証** — PT実行中のパラメータ不変性検証

**v2.4対象は Win/Place のみ。Wide は拒否する。**

</domain>

<decisions>
## Implementation Decisions

### 抽出範囲の境界 (FeatureBuilder API Design)

- **D-01:** `FeatureBuilder` は `build_all()` を含む全体をカバーする唯一の公開エントリポイント。内部は `build_base_features()` と `enrich_features()` に分割。学習時のみ `preserve_columns` でターゲット列保持、推論時はPOST_RACE列を禁止。戻り値は `FeatureBuildResult` (DataFrame + manifest)。RacePredictor内の重複特徴量計算は撤去。
- **D-02:** `src/features/feature_builder.py` に `FeatureBuilder` クラスを配置。FeatureEngine は基礎特徴量生成の下請け。FeatureBuilder は全追加モジュールの実行順序・マージ・manifest生成を担当。Train/BT/PT は `FeatureBuilder` のメソッドのみを呼ぶ。DBアクセスは行わず、入力 DataFrame と ParquetStore を受け取る。
- **D-03:** `FeatureBuildResult` dataclass (frozen) を返す。`frame: pd.DataFrame` + `manifest: FeatureManifest`。manifest hash対象はモデル入力列の名前・順序・dtype・特徴量定義バージョン。race_id/ターゲット/POST_RACE/構築日時/データ値はhash除外。学習時manifest → モデル成果物に保存、PT → 完全一致検証 → 不一致時fail-fast。構築日時をハッシュに含めると毎回異なるため一貫性検証に使えない。
- **D-04:** `build_for_training()` と `build_for_inference()` の別メソッド分離。共通処理は非公開 `_build()` に集約。学習メソッドはPIT-safeなexpanding特徴量と学習期間統計を生成。推論メソッドは必須のfit済み `FeatureState` でtransformのみ行い、欠落時fail-fast。`FeatureState` 対象: track_stats, track_month_stats, 特徴量定義バージョン、その他fit済み統計。TargetEncoder / OOF予測 / モデル校正 → FeatureState対象外、モデル成果物として管理。
- **D-05:** PIT処理は既存特徴量モジュールに委ね、FeatureBuilderは検証層に徹する。各モジュールのPIT契約をregistryで管理、推論時に `max(race_date) < prediction_date` を検証。最終段: manifestに基づき列順・dtype正規化。変換不能・必須列欠落・未知列 → fail-fast。カテゴリ符号化規則もmanifestに保存。二重シフト防止のためFeatureBuilderではPITシフトを再実装しない。

### 検証基盤の戦略 (PLN-02/03/04)

- **D-06:** Git commit SHA + dirty検知。`code_version`: commit SHA、`git_dirty`: bool。dirty時は対象コード差分のSHA256を `dirty_diff_hash` として保存。対象: `src/`, `scripts/run_paper_trading.py`, 設定ファイル, feature manifest生成コード。未追跡ファイル有無も記録。通常PT runはdirty状態を拒否（非ゼロ終了）。開発用フラグ指定時のみ警告付き許可。
- **D-07:** 二段階データカットオフ検証。段階1 — PT起動時に `DataCutoffManifest` で一括検証。段階2 — FeatureBuilder実行時に参照した履歴データのmax日付を検証。検証対象: モデル学習終了日, 特徴量統計fit終了日, OddsBandFilter校正終了日, strategy manifest最適化データ終了日。不明・欠落 → fail-fast。作成日時やFeatureManifestバージョンはデータカットオフの代用にしない。
- **D-08:** PT起動時freeze + レース予測直前verify + 終了時verify。検証対象: モデルHP, FeatureState, feature manifest, strategy manifest, OddsBandFilter, betting target/mode。除外: RegimeDetector, DDController等の実行中に意図的に変化するランタイム状態。verify失敗時は以降予測停止 → 既存記録保存 → 非ゼロ終了。SHA256検証コストは数分間隔の競馬運用では無視可能。
- **D-09:** ローカル `session_manifest.json` を正本とし、同じ識別情報をPT用MLflow runへ複製。`bets.parquet` には `session_id` + `model_run_id` のみ保存し、詳細はmanifest参照。manifestはrun開始前にatomic write、終了状態・PFP検証結果・終了コードを追記。MLflow障害時もクラッシュ復旧・監査が可能。

### BT内部重複の解消方針

- **D-10:** `BacktestEngine.prepare_data()` と `run()` 内部パスの両方をFeatureBuilder呼出に変更。`prepare_data()` はWF向け薄い互換ラッパーとして維持。`run(prepared_data=None)` も同じ内部ヘルパー経由。回帰テスト: 両経路で同一入力に対する feature manifest hash + 主要列値が一致することを検証。旧インライン実装は完全削除、フォールバックとして残さない。

### Claude's Discretion

- FeatureBuilder 内部の `_build()` メソッドの詳細な実行順序・マージ方法
- FeatureManifest / FeatureState / FeatureBuildResult dataclass の内部フィールド定義
- DataCutoffManifest の具体的な検証ロジック
- FeatureBuilder PIT registry の実装形式
- session_manifest.json のスキーマ定義
- 各特徴量モジュールのFeatureBuilder統合時のエラー処理
- dtype正規化の具体的なcoerceルール
- FeatureBuilder と RacePredictor の境界線の細部（interaction/relative/track_conditionをどこまでFeatureBuilderに含めるか）

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements & Roadmap
- `.planning/REQUIREMENTS.md` — PLN-01~04 の要件定義。Traceability table あり
- `.planning/ROADMAP.md` §Phase 52 — Goal, Success Criteria, Requirements mapping
- `.planning/PROJECT.md` — v2.4 milestone context, Out of Scope 定義
- `.planning/STATE.md` — Phase 51 deliverables, blockers/concerns

### Existing Implementation (must-read for integration)
- `src/features/feature_engine.py` lines 199-484 — `build_all()`: 基礎特徴量オーケストレーター。FeatureBuilderの内部下請け
- `src/backtest/engine.py` lines 532-879 — `prepare_data()`: 特徴量構築コピーA (FeatureBuilderで置換対象)
- `src/backtest/engine.py` lines 965-1306 — `run()` 内部パス: 特徴量構築コピーB (FeatureBuilderで置換対象)
- `src/pipelines/training_pipeline.py` lines 797-1145 — `_train_submodel()`: 特徴量構築コピーC (最も完全)
- `src/paper_trading/predictor.py` lines 42-126 — `setup()`: PT特徴量構築 (7ギャップ対象)
- `src/backtest/race_predictor.py` lines 216-299 — `predict()`: RacePredictor内の特徴量計算 (撤去対象)

### Feature Modules (must integrate into FeatureBuilder)
- `src/features/horse_history.py` — HorseHistoryFeatures (PTにある)
- `src/features/jockey_context.py` — JockeyContextFeatures (PTにある)
- `src/features/trainer_context.py` — TrainerContextFeatures (PTにある)
- `src/features/jockey_trainer_combo.py` — JockeyTrainerComboFeatures (PTにある)
- `src/features/sire_features.py` — SireFeatures (PTにない、ギャップ)
- `src/features/pace_aptitude.py` — PaceAptitudeFeatures 6列 (PTにない、ギャップ)
- `src/features/course_features.py` — CourseFeatures (PTにない、ギャップ)
- `src/features/dam_pedigree.py` — DamPedigreeFeatures (PTにない、ギャップ)
- `src/features/record_features.py` — RecordFeatures (PTにない、ギャップ)
- `src/features/mining_features.py` — MiningFeatures (PTにない、ギャップ)

### Verification Infrastructure (must integrate/reuse)
- `src/backtest/parameter_freeze_protocol.py` — ParameterFreezeProtocol: SHA256 manifest検証パターン (PFP検証で再利用)
- `src/validation/oof_health_validator.py` — OOFHealthValidator: fail-fast検証パターン (DataCutoffManifestで参考)
- `src/audit/feature_routing_registry.py` — FeatureRoutingAuditRegistry: レジストリパターン (PIT registryで参考)

### Domain Types & Models
- `src/domain/types.py` — BetType, POST_RACE_COLS (POST_RACE列除外判定で使用)
- `src/domain/models.py` — SubmodelSet (track_statsフィールド), TrainedModelsV5

### Configuration
- `config/settings.yaml` — DB接続、データパス、feature_engine設定

### Prior Phase Context
- `.planning/phases/51-settlement-integrity-training-pipeline/51-CONTEXT.md` — Phase 51決定事項(精算アーキテクチャ, bet_id, 3-column state model, ModelLoader優先度)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FeatureEngine.build_all()` (feature_engine.py:199-484): 基礎特徴量生成。FeatureBuilderの `_build_base_features()` として内部利用。既にBT/PT/Trainで共用
- `SubModelManager.add_distance_band_features()`: 距離バンド特徴量。既にBT/PT/Trainで共用
- `ParameterFreezeProtocol` (parameter_freeze_protocol.py): SHA256 manifest検証。PFP検証(D-08)でそのまま再利用
- `OOFHealthValidator` (oof_health_validator.py): fail-fast検証パターン。DataCutoffManifest(D-07)の設計参考
- `build_win_payout_map()` / `build_place_payout_map()` (betting/payout_maps.py): Phase 51で抽出済みの共用パターン。FeatureBuilder抽出の先行事例
- `SubmodelSet` (domain/models.py): track_stats/track_month_stats フィールドが既に定義済み (dict | None)。FeatureState のデータ源

### Established Patterns
- **Parquetベースデータ層**: PostgreSQLはETL専用、推論時はParquetのみ。FeatureBuilderもParquetStoreを受け取る
- **Pure function / 純粋関数抽出**: payout_maps.py はI/Oなし純粋関数。FeatureBuilder もDBアクセスなし
- **Fail-fast on missing artifacts**: OOFHealthValidator, FeatureRoutingAudit と同じパターン。FeatureState/DataCutoffManifestでも適用
- **Atomic replace**: 一時ファイル経由で書込み、renameで原子性担保。session_manifest.json でも適用
- **SHA256 manifest**: ParameterFreezeProtocol と同じパターン。FeatureManifest, DataCutoffManifest でも適用
- **3箇所コピーセット**: BacktestEngine.prepare_data(), BacktestEngine.run()内部, TrainingPipeline._train_submodel() に同一コードが存在。全てFeatureBuilder呼出に置換

### Integration Points
- `scripts/run_paper_trading.py` `_run_predict()`: PaperPredictor.setup() → FeatureBuilder.build_for_inference() に変更
- `scripts/run_train.py` main(): TrainingPipeline.run() → FeatureBuilder.build_for_training() に変更
- `scripts/run_backtest.py` main(): BacktestEngine.run() → FeatureBuilder.build_for_inference() に変更
- `src/backtest/engine.py`: prepare_data() を薄いラッパーに変更、run()内部パスをFeatureBuilder呼出に変更
- `src/backtest/race_predictor.py` predict(): 重複特徴量計算(compute_track_condition/interaction/relative)を撤去
- `src/pipelines/training_pipeline.py` _train_submodel(): 特徴量構築をFeatureBuilder呼出に変更

</code_context>

<specifics>
## Specific Ideas

- FeatureBuilder は「入力から完成特徴量まで」を単一関数で保証する。後段だけ共有すると store渡し忘れやオッズ設定差の再発リスクがある
- 学習と推論の差は入力期間とターゲット保持だけにする。それ以外は完全に同一の `_build()` 内部処理
- TargetEncoder を FeatureBuilder に含めない。TargetEncoder は target 列に依存するため、特徴量生成と target依存処理の境界を明確にする
- 既存モジュールの PIT 処理を信頼し、FeatureBuilder は検証層に徹する。PIT を再実装すると二重シフトの危険
- FeatureManifest のハッシュに構築日時を含めると毎回異なるハッシュになり一貫性検証に使えない
- 作成日時やバージョン番号はデータカットオフの代用にしない。検証すべきは「データに含まれる最大日付」
- RegimeDetector/DDController は実行中に意図的に変化するランタイム状態であり、PFP検証から除外する
- 旧インライン実装はフォールバックとして残さない。残すと「どちらが使われるか」が不明確になり将来の分岐リスクが再発

</specifics>

<deferred>
## Deferred Ideas

- Strategy manifest integration — Phase 53 (STR-01~06)
- Live data fetcher — Phase 53 (LIV-01~03)
- Regime synchronization (AGGRESSIVE固定 vs 動体) — Phase 53 (STR-06)
- One-command run mode — Phase 54 (AUT-01~03)
- Weekly/cumulative reporting — Phase 54 (RPT-01~04)
- Conservative MAWC redesign — v2.5+
- WinSegmentCalibrator dead code removal — v2.5+ (WRN-01)
- Wide bet support — v2.5+ (WID-01, WID-02)

</deferred>

---

*Phase: 52-Shared Feature Builder & Consistency*
*Context gathered: 2026-06-06*
