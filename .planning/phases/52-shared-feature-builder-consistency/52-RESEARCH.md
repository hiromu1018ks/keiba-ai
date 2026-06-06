# Phase 52: Shared Feature Builder & Consistency - Research

**Researched:** 2026-06-06
**Domain:** Feature generation consolidation, pipeline consistency verification
**Confidence:** HIGH

## Summary

Phase 52 は、BT/PT/TrainingPipeline の3箇所に分散する特徴量構築コードを単一の `FeatureBuilder` クラスに統合する。現在、3箇所のコピーセットは「学習パイプライン（最も完全）」→「BacktestEngine.prepare_data()」→「BacktestEngine.run()内部パス」→「PaperPredictor.setup()」の順で特徴量モジュールが欠落しており、PT には7つの特徴量モジュール（Sire/PaceAptitude/Course/DamPedigree/Record/Mining/interaction）が完全に欠落している。更に RacePredictor 内には track_condition_features/interaction_features/relative_features の重複計算が存在し、これらを FeatureBuilder に統合・撤去する。

**Primary recommendation:** TrainingPipeline._train_submodel() (lines 797-1145) を「参照実装」とし、FeatureBuilder._build() に完全移植する。3箇所のコピーは FeatureBuilder 呼出に置換し、RacePredictor 内の重複計算を撤去する。FeatureBuildResult/FeatureManifest/FeatureState dataclass で一貫性検証基盤を構築する。

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** FeatureBuilder は build_all() を含む唯一の公開エントリポイント。内部は build_base_features() と enrich_features() に分割。学習時のみ preserve_columns でターゲット列保持、推論時はPOST_RACE列禁止。戻り値は FeatureBuildResult (DataFrame + manifest)。RacePredictor内の重複特徴量計算は撤去。
- **D-02:** FeatureBuilder クラスは src/features/feature_builder.py に配置。FeatureEngine は基礎特徴量生成の下請け。FeatureBuilder は全追加モジュールの実行順序・マージ・manifest生成を担当。DBアクセスは行わず、入力 DataFrame と ParquetStore を受け取る。
- **D-03:** FeatureBuildResult dataclass (frozen)。frame: pd.DataFrame + manifest: FeatureManifest。manifest hash対象はモデル入力列の名前・順序・dtype・特徴量定義バージョン。race_id/ターゲット/POST_RACE/構築日時/データ値はhash除外。
- **D-04:** build_for_training() と build_for_inference() の別メソッド分離。共通処理は非公開 _build() に集約。推論メソッドは必須のfit済み FeatureState でtransformのみ行い、欠落時fail-fast。FeatureState 対象: track_stats, track_month_stats, 特徴量定義バージョン、その他fit済み統計。TargetEncoder/OOF予測/モデル校正 → FeatureState対象外。
- **D-05:** PIT処理は既存特徴量モジュールに委ね、FeatureBuilderは検証層に徹する。各モジュールのPIT契約をregistryで管理。二重シフト防止のためFeatureBuilderではPITシフトを再実装しない。
- **D-06:** Git commit SHA + dirty検知。code_version: commit SHA、git_dirty: bool。dirty時は対象コード差分のSHA256を dirty_diff_hash として保存。通常PT runはdirty状態を拒否。
- **D-07:** 二段階データカットオフ検証。段階1 — PT起動時に DataCutoffManifest で一括検証。段階2 — FeatureBuilder実行時に参照した履歴データのmax日付を検証。
- **D-08:** PT起動時freeze + レース予測直前verify + 終了時verify。検証対象: モデルHP, FeatureState, feature manifest, strategy manifest, OddsBandFilter, betting target/mode。除外: RegimeDetector, DDController等の実行中に意図的に変化するランタイム状態。
- **D-09:** ローカル session_manifest.json を正本とし、同じ識別情報をPT用MLflow runへ複製。bets.parquet には session_id + model_run_id のみ保存。manifestはrun開始前にatomic write。
- **D-10:** BacktestEngine.prepare_data() と run() 内部パスの両方をFeatureBuilder呼出に変更。prepare_data() はWF向け薄い互換ラッパーとして維持。旧インライン実装は完全削除、フォールバックとして残さない。

### Claude's Discretion
- FeatureBuilder 内部の _build() メソッドの詳細な実行順序・マージ方法
- FeatureManifest / FeatureState / FeatureBuildResult dataclass の内部フィールド定義
- DataCutoffManifest の具体的な検証ロジック
- FeatureBuilder PIT registry の実装形式
- session_manifest.json のスキーマ定義
- 各特徴量モジュールのFeatureBuilder統合時のエラー処理
- dtype正規化の具体的なcoerceルール
- FeatureBuilder と RacePredictor の境界線の細部

### Deferred Ideas (OUT OF SCOPE)
- Strategy manifest integration — Phase 53
- Live data fetcher — Phase 53
- Regime synchronization — Phase 53
- One-command run mode — Phase 54
- Weekly/cumulative reporting — Phase 54
- Conservative MAWC redesign — v2.5+
- WinSegmentCalibrator dead code removal — v2.5+
- Wide bet support — v2.5+
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PLN-01 | Shared feature builder extraction — BT/PT/TrainingPipeline が共通の特徴量構築関数を呼ぶ。7つのギャップを一括解消 | 3箇所コピー差異分析完了、参照実装特定(_train_submodel)、7ギャップの内容特定 |
| PLN-02 | Pipeline identity recording — MLflow run ID・学習期間・コードハッシュ・feature manifest hash を PT 実行記録に保存 | ParameterFreezeProtocol SHA256パターン再利用可能、Git SHA取得方法確認済 |
| PLN-03 | Data cutoff validation — 2026年 PT では 2025年12月31日以前のデータのみ使用。予測日以降の情報を含まないことを検証 | OOFHealthValidator fail-fastパターン参考、二段階検証設計(D-07)完了 |
| PLN-04 | PFP parameter immutability — PT 実行中のパラメータ不変性を ParameterFreezeProtocol で検証 | PFPコード確認済、freeze/verifyパターンそのまま再利用 |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Feature generation (base features) | Python/ML Pipeline | -- | FeatureEngine.build_all() が基礎特徴量を生成 |
| Feature generation (enrichment) | Python/ML Pipeline | -- | FeatureBuilder が追加モジュール(Sire/Pace/Course等)を統合 |
| Feature manifest hash | Python/ML Pipeline | -- | 列名・順序・dtype・バージョンのSHA256ハッシュ |
| FeatureState (fit statistics) | Python/ML Pipeline | -- | track_stats/track_month_stats等の学習期間統計 |
| Pipeline identity recording | Python/ML Pipeline | MLflow | Git SHA + manifest hash を session_manifest + MLflow に記録 |
| Data cutoff validation | Python/ML Pipeline | -- | 予測日以前のデータのみ使用することを検証 |
| PFP immutability | Python/ML Pipeline | -- | ParameterFreezeProtocol の freeze/verify パターン |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | 2.x | DataFrame処理 | 既存依存 [VERIFIED: pyproject.toml] |
| numpy | 1.x | 数値計算 | 既存依存 [VERIFIED: pyproject.toml] |
| hashlib (stdlib) | -- | SHA256 manifest hash | ParameterFreezeProtocolと同一パターン [VERIFIED: stdlib] |
| json (stdlib) | -- | manifest serialization | 既存パターン [VERIFIED: stdlib] |
| dataclasses (stdlib) | -- | FeatureBuildResult/FeatureState | frozen dataclass [VERIFIED: stdlib] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pickle (stdlib) | -- | PFP serialization | ParameterFreezeProtocol再利用時 |
| gitpython | -- | Git dirty/state取得 | D-06 code_version取得 [ASSUMED] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| gitpython | subprocess git CLI | gitpython は追加依存。subprocess git は追加依存なしで同等機能。既存コードに gitpython 依存なし |

**Installation:**
```bash
# 新規外部パッケージインストールなし
# 全て stdlib + 既存依存で完結
```

## Package Legitimacy Audit

> このフェーズは外部パッケージをインストールしないため、Package Legitimacy Gate は不要。

**外部パッケージインストール:** なし

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────┐
                         │       Caller (BT/PT/Train)   │
                         └────────────┬────────────────┘
                                      │
                    build_for_training() / build_for_inference()
                                      │
                         ┌────────────▼────────────────┐
                         │      FeatureBuilder          │
                         │  (src/features/              │
                         │   feature_builder.py)        │
                         │                              │
                         │  _build():                   │
                         │   1. build_base_features()   │
                         │   2. enrich_features()       │
                         │   3. PIT validation          │
                         │   4. manifest generation     │
                         └────────────┬────────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                     │
      ┌──────────▼──────┐  ┌─────────▼─────────┐  ┌───────▼──────────┐
      │  FeatureEngine   │  │ Feature Modules   │  │ FeatureState     │
      │  (build_all)     │  │ (25 modules)      │  │ (fit statistics) │
      │                  │  │                   │  │                  │
      │ Base features:   │  │ SireFeatures      │  │ track_stats      │
      │ - race/entry     │  │ PaceAptitude      │  │ track_month_stats│
      │ - odds           │  │ CourseFeatures    │  │ version          │
      │ - intra_race     │  │ DamPedigree       │  └──────────────────┘
      │ - market_bias    │  │ RecordFeatures    │
      │ - bloodline      │  │ MiningFeatures    │
      │ - track_conds    │  │ interaction       │
      │ - horse_track_apt│  │ relative          │
      └──────────────────┘  │ track_condition   │
                            └───────────────────┘
                                      │
                         ┌────────────▼────────────────┐
                         │   FeatureBuildResult         │
                         │   (frame: DataFrame,         │
                         │    manifest: FeatureManifest)│
                         └─────────────────────────────┘
```

### Recommended Project Structure
```
src/features/
├── feature_builder.py       # NEW: FeatureBuilder class (single entry point)
├── feature_engine.py        # EXISTING: build_all() base features (internal subcontractor)
├── feature_manifest.py      # NEW: FeatureManifest/FeatureState/FeatureBuildResult dataclasses
├── pit_registry.py          # NEW: PIT contract registry per module
├── interaction_features.py  # EXISTING: compute_interaction_features
├── relative_features.py     # EXISTING: compute_relative_features
├── track_condition_features.py  # EXISTING: compute_track_condition_features
├── sire_features.py         # EXISTING: SireFeatures
├── pace_aptitude_features.py    # EXISTING: PaceAptitudeFeatures
├── course_features.py       # EXISTING: CourseFeatures
├── dam_pedigree_features.py # EXISTING: DamPedigreeFeatures
├── record_features.py       # EXISTING: RecordFeatures
├── mining_features.py       # EXISTING: MiningFeatures
├── horse_history_features.py    # EXISTING: HorseHistoryFeatures
├── jockey_context_features.py   # EXISTING: JockeyContextFeatures
├── trainer_context_features.py  # EXISTING: TrainerContextFeatures
├── jockey_trainer_combo.py      # EXISTING: JockeyTrainerComboFeatures
└── ... (25+ existing modules)
```

### Pattern 1: Feature Builder Pipeline (D-01/D-04)
**What:** 単一クラスが全特徴量モジュールの実行順序を制御し、FeatureBuildResult を返す
**When to use:** Train/BT/PT の全特徴量生成パス
**Example:**
```python
# D-01/D-04: 学習と推論のエントリポイント
class FeatureBuilder:
    def build_for_training(self, race_df, entry_df, odds_df, store, ...) -> FeatureBuildResult:
        """学習用: preserve_columns でターゲット列保持"""
        return self._build(preserve_columns=["kakuteijyuni", "confirmed_odds"], ...)

    def build_for_inference(self, race_df, entry_df, odds_df, store, feature_state, ...) -> FeatureBuildResult:
        """推論用: FeatureState必須、POST_RACE列禁止、欠落時fail-fast"""
        if feature_state is None:
            raise ValueError("FeatureState required for inference")
        return self._build(preserve_columns=None, feature_state=feature_state, ...)
```

### Pattern 2: Feature Manifest Hash (D-03)
**What:** モデル入力列の名前・順序・dtype・特徴量定義バージョンのSHA256ハッシュ
**When to use:** 学習時manifest保存 → PT時完全一致検証
**Example:**
```python
@dataclass(frozen=True)
class FeatureManifest:
    column_names: tuple[str, ...]
    column_dtypes: tuple[str, ...]
    feature_version: str  # 特徴量定義バージョン

    def compute_hash(self) -> str:
        data = json.dumps({
            "columns": list(self.column_names),
            "dtypes": list(self.column_dtypes),
            "version": self.feature_version,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
```

### Pattern 3: FeatureState for Inference (D-04)
**What:** 学習期間にfitした統計を推論時に渡す不変オブジェクト
**When to use:** PT/BT推論時。学習時は FeatureBuilder 内部で統計を計算・保存
**Example:**
```python
@dataclass(frozen=True)
class FeatureState:
    track_stats: dict[str, dict[str, float]]  # SubmodelSet.track_stats から
    track_month_stats: dict[str, dict[str, float]]  # SubmodelSet.track_month_stats から
    feature_version: str

    @classmethod
    def from_submodel_set(cls, submodel: SubmodelSet, version: str) -> FeatureState:
        if submodel.track_stats is None:
            raise ValueError("submodel.track_stats is None — train with Phase 51 TRN-04")
        return cls(
            track_stats=submodel.track_stats,
            track_month_stats=submodel.track_month_stats or {},
            feature_version=version,
        )
```

### Anti-Patterns to Avoid
- **Anti-pattern: FeatureBuilder内でのPITシフト再実装:** 既存モジュール(HorseHistoryFeatures, PaceAptitudeFeatures等)がPIT安全。FeatureBuilderで再実装すると二重シフトの危険。(D-05)
- **Anti-pattern: 旧インライン実装のフォールバック保持:** 残すと「どちらが使われるか」が不明確になり将来の分岐リスクが再発。(D-10)
- **Anti-pattern: 構築日時のハッシュへの含め:** 毎回異なるハッシュになり一貫性検証に使えない。(D-03 specifics)
- **Anti-pattern: 作成日時やバージョン番号によるデータカットオフ判定:** 検証すべきは「データに含まれる最大日付」。(specifics)
- **Anti-pattern: TargetEncoderのFeatureBuilder組み込み:** TargetEncoder は target 列に依存するため、特徴量生成と target依存処理の境界を明確にする。(specifics)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SHA256 manifest verification | 新規SHA256実装 | ParameterFreezeProtocol._serialize() + hashlib | 既にデプロイ済みパターン、pickle+SHA256の実績あり |
| Fail-fast validation framework | 新規validation | OOFHealthValidator パターン | fail-fast設計、ValidationResult dataclass 既存 |
| Feature routing audit | 新規registry | FeatureRoutingAuditRegistry パターン | registry バージョン管理、CI check 既存 |
| Atomic file write | 新規atomic実装 | 一時ファイル経由rename | 既存パターン(session_manifestで再利用) |
| Strategy manifest verification | 新規verification | verify_strategy_manifest() | SHA256照合実績あり、そのまま再利用 |

**Key insight:** Phase 42-43で構築した検証基盤(ParameterFreezeProtocol, OOFHealthValidator, FeatureRoutingAuditRegistry)の設計パターンをそのまま適用する。新規フレームワークは不要。

## Common Pitfalls

### Pitfall 1: 3箇所コピーの実行順序差異
**What goes wrong:** 各コピーの特徴量モジュール実行順序が異なり、interaction/track_condition/relative の依存関係が壊れる
**Why it happens:** _train_submodel は hist→pace→course→sire→dam→record→track_condition→interaction→mining→relative の順だが、prepare_dataは hist→pace→course→sire→dam→record→mining で track_condition/interaction/relative が欠落。RacePredictorは推論時のみ track_condition→interaction→relative を実行
**How to avoid:** FeatureBuilder._build() は _train_submodel の実行順序を忠実に再現する。依存関係: HorseHistoryFeatures → PaceAptitude → Course → Sire → DamPedigree → Record → TrackCondition → Interaction → Mining → Relative
**Warning signs:** 推論時に NaN 列が増える、interaction特徴量が欠落

### Pitfall 2: track_stats/track_month_stats の未保存による推論NaN
**What goes wrong:** track_condition_features の T1-02 (relative/zscore) と T3-04 (season deviation) が NaN になる
**Why it happens:** SubmodelSet.track_stats/track_month_stats が None の場合、compute_track_condition_features が statistics-free フォールバック計算になる
**How to avoid:** Phase 51 TRN-04 で track_stats JSON 持続性は実装済みだが、全サーフェスモデルで保存されていることを FeatureState 生成時に検証する
**Warning signs:** turf_cushion_track_relative, turf_cushion_track_zscore, cushion_season_deviation が推論時に NaN

### Pitfall 3: FeatureManifest hash の「不変でない」要素の混入
**What goes wrong:** 構築日時やデータ値をhashに含めると、同一コード・同一入力でも異なるhashになる
**Why it happens:** datetime.now() や DataFrame 値の微小差（浮動小数点丸め）をhash計算に含める
**How to avoid:** hash対象は列名・順序・dtype・特徴量定義バージョンのみ。race_id/ターゲット/POST_RACE/構築日時/データ値はhash除外 (D-03)
**Warning signs:** 同一モデル・同一コードで manifest hash が異なる

### Pitfall 4: FeatureBuilder への DB アクセス混入
**What goes wrong:** FeatureBuilder 内で直接DBクエリを実行するとテスト不可能になる
**Why it happens:** SireFeatures や CourseFeatures が ParquetStore にアクセスするが、FeatureBuilder 自体は DB アクセスを行わない設計
**How to avoid:** FeatureBuilder は入力 DataFrame と ParquetStore を受け取る。FeatureBuilder 内では load_horses や load_sire_stats は呼ばず、各特徴量モジュール内で store から読み込む (D-02)
**Warning signs:** FeatureBuilder のテストにDB接続が必要になる

### Pitfall 5: 回帰テスト不十分による BT 結果変化
**What goes wrong:** FeatureBuilder 移行後に BT結果が変化し、モデル精度が劣化する
**Why it happens:** マージ順序の違いによる NaN伝播、列名衝突、dtypeの暗黙変換
**How to avoid:** D-10: 移行前後に同一入力で feature manifest hash + 主要列値の一致検証。prepare_data() と run() 内部パスの両方で同一結果になることを確認
**Warning signs:** BT ROI の予期しない変化、特徴量数の増減

## Code Examples

### Feature Builder Core (参照実装からの移植)
```python
# Source: TrainingPipeline._train_submodel() lines 797-1015
# FeatureBuilder._build() に移植すべき実行順序:

# 1. FeatureEngine.build_all() → build_base_features()
feat_df = self._feat_engine.build_all(race_df, entry_df, odds_df, ...)

# 2. SubModelManager.add_distance_band_features()
feat_df = self._submodel_mgr.add_distance_band_features(feat_df)

# 3. HorseHistoryFeatures (hist_df) + add_race_transforms
hist_all = HorseHistoryFeatures(store=store)
hist_df = hist_all.compute(race_df, entry_df, race_ids)
# hist_df merge is deferred to after enrichment

# 4. enrich_features():
#   a. PaceAptitudeFeatures (6 columns)
#   b. CourseFeatures (2 columns)
#   c. SireFeatures (11 columns)
#   d. DamPedigreeFeatures (4 columns)
#   e. RecordFeatures (1 column)
#   f. TrackConditionFeatures (requires track_stats/track_month_stats from FeatureState)
#   g. InteractionFeatures (15 columns)
#   h. MiningFeatures (4 columns)
#   i. RelativeFeatures (11 columns)

# 5. hist_df merge + race_transforms
feat_df = feat_df.merge(hist_df, on=["race_id", "umaban"], how="left")
feat_df = HorseHistoryFeatures.add_race_transforms(feat_df)

# 6. POST_RACE exclusion (if not training)
# 7. FeatureManifest generation
```

### ParameterFreezeProtocol Reuse Pattern (D-08)
```python
# Source: src/backtest/parameter_freeze_protocol.py
# PFP検証の再利用パターン

class PFPVerifier:
    def __init__(self, models: TrainedModelsV5, feature_state: FeatureState, ...):
        self._pfp = ParameterFreezeProtocol(models)
        self._feature_state_hash = _compute_feature_state_hash(feature_state)
        self._manifest_hash = manifest.compute_hash()

    def freeze(self):
        self._pfp.freeze()

    def verify(self) -> dict[str, Any]:
        pfp_result = self._pfp.verify()
        if not pfp_result["passed"]:
            return pfp_result
        # Additional verification for FeatureState, manifest etc.
        return {"passed": True, "message": "All immutability checks passed"}
```

### DataCutoffManifest Pattern (D-07)
```python
# Source: OOFHealthValidator fail-fast pattern
@dataclass(frozen=True)
class DataCutoffManifest:
    model_train_end: str          # モデル学習終了日
    stats_fit_end: str            # 特徴量統計fit終了日
    odds_band_calibration_end: str  # OddsBandFilter校正終了日
    strategy_optimization_end: str  # strategy manifest最適化データ終了日
    prediction_date: str          # PT予測日

    def verify(self, actual_dates: dict[str, str]) -> list[str]:
        """全データソースの最大日付がprediction_date以前であることを検証"""
        failures = []
        for key, actual_end in actual_dates.items():
            expected = getattr(self, key, None)
            if expected and actual_end > expected:
                failures.append(f"{key}: data extends to {actual_end}, expected <= {expected}")
        return failures
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 3箇所特徴量コピーセット | FeatureBuilder単一エントリ | Phase 52 | 一貫性保証、7ギャップ解消 |
| PT の7モジュール欠落 | 全モジュール統合 | Phase 52 | PT推論精度向上 |
| RacePredictor内重複計算 | FeatureBuilder内に統合・撤去 | Phase 52 | コード重複解消 |
| 手動一貫性確認 | FeatureManifest hash自動検証 | Phase 52 | CI可能な一貫性保証 |

**Deprecated/outdated:**
- BacktestEngine.prepare_data() のインライン特徴量構築コード (lines 686-879): FeatureBuilder呼出に置換
- BacktestEngine.run() 内部パスのインライン特徴量構築コード (lines 1100-1306): FeatureBuilder呼出に置換
- RacePredictor.predict() 内の track_condition/interaction/relative 計算 (lines 279-299): FeatureBuilderに統合して撤去

## 3箇所コピー差異分析

### モジュール実行比較表

| 特徴量モジュール | TrainingPipeline._train_submodel | BacktestEngine.prepare_data | BacktestEngine.run()内部 | PaperPredictor.setup | RacePredictor.predict |
|---|---|---|---|---|---|
| FeatureEngine.build_all() | YES (line 797前) | YES (line 686) | YES (line 1100) | YES (line 91) | -- |
| SubModelManager.distance_band | YES | YES (line 694) | YES (line 1108) | YES (line 94) | -- |
| HorseHistoryFeatures | YES (line 818) | YES (line 739) | YES (line 1156) | YES (line 98) | 引数で受取 |
| JockeyContextFeatures | YES (line 1142) | YES (line 743) | YES (line 1160) | YES (line 101) | 引数で受取 |
| TrainerContextFeatures | YES (line 1147) | YES (line 747) | YES (line 1164) | YES (line 104) | 引数で受取 |
| JockeyTrainerCombo | YES (line 1155) | YES (line 751) | YES (line 1168) | YES (line 107) | 引数で受取 |
| **SireFeatures** | **YES (line 897)** | **YES (line 758)** | **YES (line 1175)** | **NO** | -- |
| **PaceAptitudeFeatures** | **YES (line 828, 6列)** | **YES (line 795, 6列)** | **YES (line 1212, 6列)** | **NO** | -- |
| **CourseFeatures** | **YES (line 875, 2列)** | **YES (line 817, 2列)** | **YES (line 1235, 2列)** | **NO** | -- |
| **DamPedigreeFeatures** | **YES (line 936, 4列)** | **YES (line 835, 4列)** | **YES (line 1254, 4列)** | **NO** | -- |
| **RecordFeatures** | **YES (line 952, 1列)** | **YES (line 846, 1列)** | **YES (line 1266, 1列)** | **NO** | -- |
| **MiningFeatures** | **YES (line 999, 4列)** | **YES (line 856, 4列)** | **YES (line 1278, 4列)** | **NO** | -- |
| **TrackConditionFeatures** | **YES (line 980)** | **NO** | **NO** | **NO** | **YES (line 288)** |
| **InteractionFeatures** | **YES (line 992)** | **NO** | **NO** | **NO** | **YES (line 294)** |
| **RelativeFeatures** | **YES (line 1014)** | **NO** | **NO** | **NO** | **YES (line 299)** |

### ギャップサマリ
- **PT (PaperPredictor.setup):** Sire/PaceAptitude/Course/DamPedigree/Record/Mining の6モジュール完全欠落 (TrackCondition/Interaction/Relative は RacePredictor.predict で代替計算)
- **prepare_data / run()内部:** TrackCondition/Interaction/Relative が欠落。RacePredictor で代替計算しているため BT結果は正しいが、コード重複がある
- **RacePredictor.predict:** TrackCondition/Interaction/Relative の重複計算。FeatureBuilder に統合後は撤去

## 7つのギャップの詳細

| ギャップ | モジュール | 出力列数 | 列名 | インターフェース | マージキー |
|---|---|---|---|---|---|
| Sire | SireFeatures | 11 | sire_wr, sire_place_rate, sire_surface_wr, sire_distance_wr, sire_prize_avg, bms_wr, bms_surface_wr, bms_distance_wr, bms_has_history, bms_starts_log, bms_surface_starts_log, bms_distance_starts_log | SireFeatures(sire_stats_df).compute_batch(df) | (kettonum → sire_id map) |
| PaceAptitude | PaceAptitudeFeatures | 6 | pace_aptitude, front_pace_wr, closing_pace_wr, pace_corner_stability, pace_closing_power, pace_position_consistency | PaceAptitudeFeatures(store).compute_batch(df) | [kettonum, race_id] |
| Course | CourseFeatures | 2 | course_wr, course_distance_wr | CourseFeatures(store).compute_batch(df) | [kettonum, race_id] |
| DamPedigree | DamPedigreeFeatures | 4 | dam_wr, dam_surface_wr, dam_prize_log, breeder_strength | DamPedigreeFeatures(store).compute(df) | [race_id, umaban] |
| Record | RecordFeatures | 1 | course_record_time | RecordFeatures(store).compute(df) | [race_id] |
| Mining | MiningFeatures | 4 | dm_time_rank, dm_time_zscore, dm_confidence_range, dm_time_margin_to_fav | MiningFeatures(store).compute(df) | [race_id, umaban] |
| Interaction | compute_interaction_features | 15 | kyakusitu_x_distance 等 (INTERACTION_COLS参照) | compute_interaction_features(df) | 列追加 (merge不要) |

## FeatureEngine.build_all() 現在の能力

### 既に統合済みのモジュール (build_all 内)
1. _map_basic_features() — 基本特徴量マッピング
2. compute_intra_race_features — レース内特徴量
3. compute_odds_dynamics — オッズ動態特徴量
4. compute_market_bias — 市場バイアス特徴量
5. compute_flb_slope — FLB スロープ特徴量
6. compute_difficulty_score — レース難易度スコア
7. compute_race_level_features — レースレベル特徴量
8. compute_market_cross_features — 市場クロス特徴量
9. BloodlineFeatures — 血統特徴量
10. Track conditions merge (生値マージ) — dirt_moisture, turf_cushion
11. Horse track aptitude merge — horse_dirt_wet_hit_rate 等

### FeatureBuilder が追加で統合すべきモジュール
1. HorseHistoryFeatures + add_race_transforms
2. JockeyContextFeatures
3. TrainerContextFeatures
4. JockeyTrainerComboFeatures
5. SireFeatures
6. PaceAptitudeFeatures (6 columns)
7. CourseFeatures
8. DamPedigreeFeatures
9. RecordFeatures
10. TrackConditionFeatures (track_stats/track_month_stats依存)
11. InteractionFeatures
12. MiningFeatures
13. RelativeFeatures

**計: 13モジュールをFeatureBuilder._build()に統合**

## FeatureState に必要な fit 済み統計

### 既存 (SubmodelSet に保存済み)
| 統計 | SubmodelSet フィールド | 型 | 内容 |
|---|---|---|---|
| track_stats | track_stats: dict \| None | dict[str, dict[str, float]] | trackcd → {mean, std} (turf_cushion) |
| track_month_stats | track_month_stats: dict \| None | dict[str, dict[str, float]] | trackcd_month → {cushion_mean, cushion_std, moisture_mean, moisture_std} |

### 保存場所 (Phase 51 TRN-04)
- meta.json の "track_stats" / "track_month_stats" キーに JSON として保存
- ModelLoader で復元 → SubmodelSet.track_stats/track_month_stats に設定

### FeatureState の追加候補 (Claude's Discretion)
- **feature_version:** 特徴量定義バージョン (manifest hash計算に使用)
- その他のfit統計は現在存在しないため、初期実装では track_stats, track_month_stats, feature_version のみで十分

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | gitpython は使わず subprocess git で code_version を取得する | Standard Stack | 低: stdlib の subprocess で代替可能 |
| A2 | FeatureBuilder の実行順序は _train_submodel の順序をそのまま踏襲する | Architecture Patterns | 中: 順序が異なるとNaN伝播の可能性。回帰テストで検証 |
| A3 | PaceAptitudeFeatures.compute_batch() は 6列すべてを返す | ギャップ詳細 | 低: engine.py で 6列マージ済み (BUG-FIX comment あり) |
| A4 | Phase 51 TRN-04 で track_stats JSON 持続性は既に動作する | FeatureState | 中: 動作確認必要。None の場合 FeatureState 生成で fail-fast |
| A5 | RelativeFeatures (stage2) は FeatureBuilder 対象外 — RacePredictor 内で計算される | Architecture Patterns | 低: stage2 は p_ability_win 依存のため FeatureBuilder の範囲外 |

## Open Questions

1. **HorseHistoryFeatures の add_race_transforms タイミング**
   - What we know: _train_submodel では hist_df merge 直後に add_race_transforms を呼ぶ (line 823)
   - What's unclear: prepare_data では hist_df merge を最後に実行し (line 869-883)、add_race_transforms を呼んでいない
   - Recommendation: FeatureBuilder._build() では _train_submodel の順序に従い、hist merge 直後に add_race_transforms を呼ぶ。回帰テストで確認

2. **JockeyContext/TrainerContext/JockeyTrainerCombo のマージタイミング**
   - What we know: _train_submodel では Stage2 直前 (line 1142-1158) にマージ。prepare_data では事前計算して BacktestPreparedData に格納
   - What's unclear: FeatureBuilder 内でいつマージするか
   - Recommendation: FeatureBuilder._build() の最終段（RelativeFeaturesの後）でマージ。Stage2 相対特徴量にこれらの列が依存しないため、順序は任意。ただし _train_submodel では後段でマージしているので、同一順序を維持する方が安全

3. **RacePredictor の predict() シグネチャ変更**
   - What we know: 現在 hist_features, jockey_features, trainer_features, jt_combo_features を引数で受け取る
   - What's unclear: FeatureBuilder 統合後、これらの引数が不要になるか
   - Recommendation: FeatureBuilder が全特徴量を含む DataFrame を返すため、RacePredictor.predict() は race_df のみを受け取るように変更。ただし、BT では prepare_data から事前計算済み特徴量を受け取るパターンを維持する必要があるため、後方互換性の検討が必要

## Environment Availability

Step 2.6: SKIPPED — 外部依存なし (コード/設定変更のみ)

## Sources

### Primary (HIGH confidence)
- src/backtest/engine.py lines 532-1306 — 3箇所コピーの全コードを直接確認 [VERIFIED: codebase]
- src/pipelines/training_pipeline.py lines 797-1145 — 参照実装を直接確認 [VERIFIED: codebase]
- src/paper_trading/predictor.py lines 42-126 — PT特徴量構築の全コードを直接確認 [VERIFIED: codebase]
- src/backtest/race_predictor.py lines 200-400 — 重複特徴量計算の全コードを直接確認 [VERIFIED: codebase]
- src/features/feature_engine.py lines 199-484 — build_all() の全コードを直接確認 [VERIFIED: codebase]
- src/domain/models.py — SubmodelSet.track_stats/track_month_stats 定義を直接確認 [VERIFIED: codebase]
- src/backtest/parameter_freeze_protocol.py — PFP freeze/verify パターンを直接確認 [VERIFIED: codebase]
- src/features/track_condition_features.py — _compute_track_stats/_compute_track_month_stats を直接確認 [VERIFIED: codebase]

### Secondary (MEDIUM confidence)
- src/features/sire_features.py — SireFeatures インターフェース確認 [VERIFIED: codebase]
- src/features/pace_aptitude_features.py — PaceAptitudeFeatures 6列確認 [VERIFIED: codebase]
- src/features/course_features.py — CourseFeatures 2列確認 [VERIFIED: codebase]
- src/features/dam_pedigree_features.py — DamPedigreeFeatures 4列確認 [VERIFIED: codebase]
- src/features/record_features.py — RecordFeatures 1列確認 [VERIFIED: codebase]
- src/features/mining_features.py — MiningFeatures 4列確認 [VERIFIED: codebase]
- src/features/interaction_features.py — 15列確認 [VERIFIED: codebase]
- src/features/relative_features.py — 11+3列確認 [VERIFIED: codebase]
- src/validation/oof_health_validator.py — fail-fast パターン確認 [VERIFIED: codebase]
- src/audit/feature_routing_registry.py — registry パターン確認 [VERIFIED: codebase]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — 外部パッケージなし、stdlib + 既存依存のみ
- Architecture: HIGH — 全コードを直接確認、3箇所差異を完全に把握
- Pitfalls: HIGH — _train_submodel の実行順序を直接確認、依存関係を明確化

**Research date:** 2026-06-06
**Valid until:** 2026-07-06 (stable codebase, 低変動)
