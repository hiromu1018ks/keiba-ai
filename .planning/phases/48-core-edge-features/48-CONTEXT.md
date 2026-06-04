# Phase 48: Core Edge Features - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning

<domain>
## Phase Boundary

含水率・クッション値のTier 1+2交互作用特徴量(T1-01, T1-02, T2-01, T2-02, T2-03)をFeatureEngineに実装・登録し、REG-01(12モデルFEATURE_COLS登録)を完了する。Phase 47で作成したtrack_conditions.parquetのdirt_moisture/turf_cushionを特徴量パイプラインに統合する最初のフェーズ。

**In scope:** T1-01 (dirt_moisture_x_kyakusitu), T1-02 (cushion_track_relative/zscore), T2-01 (moisture_x_barrier + flags), T2-02 (cushion_x_kyakusitu), T2-03 (sire_x_cushion_band), REG-01 (FEATURE_COLS 12モデル登録)
**Out of scope:** T3/T4特徴量 (Phase 49), Feature Routing Audit (Phase 50), BT ROI検証 (Phase 50), IC評価 (Phase 50)

</domain>

<decisions>
## Implementation Decisions

### Data Merge Strategy (ハイブリッド)
- **D-01:** 生値(dirt_moisture, turf_cushion)は `FeatureEngine.build_all()` 内でtrack_conditions.parquetからrace_dfに左結合。DataRepository(store)経由でload_track_conditions()を呼び出す。start/endはrace_dfのrace_date min/maxから導出
- **D-02:** 交互作用特徴量の計算は `_train_submodel()` / `BacktestEngine` 内で実行。kyakusitukubun_cdがHorseHistoryFeatures後処理でしか利用不可能なため、interaction_features.pyと同じ遅延タイミング
- **D-03:** build_all()でのマージにより、horse_features.parquetキャッシュに生値が保存される。BacktestEngineでの再マージ不要

### Surgical Routing (外科的ルーティング)
- **D-04:** トラック条件特徴量を登録するモデル: AbilityModel, WinTwoStage, PlaceTwoStage, WideTwoStage, EVCorrection, PlaceEVCorrection (各surface)
- **D-05:** トラック条件特徴量を除外するモデル: MarketModel, RaceQualityScreener, RegimeDetector
- **D-06:** 理由: Phase 36で強特徴量の全モデル一律登録によりMarketModelが支配された。馬場状態特徴量は市場残差モデルにはノイズになるため除外

### Feature Module Organization
- **D-07:** 新規モジュール `src/features/track_condition_features.py` を作成。公開関数 `compute_track_condition_features(df) -> pd.DataFrame` と列定数 `TRACK_CONDITION_COLS: list[str]`
- **D-08:** 既存 `interaction_features.py` は汎用交互作用のまま維持。馬場連続値に固有の正規化・ビニング・surface別処理・将来T3/T4拡張は専用モジュールに集約
- **D-09:** 呼び出しタイミング: `_train_submodel()` / `BacktestEngine` 内で、HorseHistoryFeatures完了後、interaction_featuresと同じ箇所

### Normalization & Binning
- **D-10:** `turf_cushion_track_relative`: 学習期間のみのtrackcd別meanをベースラインとし、`turf_cushion - track_mean` で計算。ルックアヘッド回避
- **D-11:** `turf_cushion_track_zscore`: `(turf_cushion - track_mean) / track_std`。std==0 or NaNの場合はzscoreもNaN
- **D-12:** `sire_x_cushion_band`: 固定5段階ビン `[0, 7, 8, 9, 10, inf]`、labels=`["very_soft","soft","standard","firm","very_firm"]`。分位数ビン(fold不安定)・3ビン(T2-03要件不適合)は不採用
- **D-13:** 交互作用: `sire_id + "_" + cushion_band` をcategory型として生成

### Claude's Discretion
- テスト構成・テストケースの詳細設計 (既存パターンに従う)
- TRACK_CONDITION_COLSの具体的な列名定義 (要件の仕様に従う)
- track_mean/track_stdの保存形式 (dict, DataFrame, or module-level cache)
- surface-aware計算の実装詳細 (ダートのみdirt_moisture、芝のみturf_cushionを意識したNaN処理)
- build_all()へのstore経由DataRepository呼び出しの実装パターン (BloodlineFeaturesと同様)
- ログフォーマット・進捗表示の設計

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Feature Engine & Pipeline
- `src/features/feature_engine.py` — FeatureEngine.build_all() sequential pipeline、track_conditionsマージ統合ポイント、BloodlineFeatures(store)呼び出しパターン参照
- `src/pipelines/training_pipeline.py` — _train_submodel() 内の HorseHistoryFeatures → interaction_features 呼び出し順序 (track_condition_featuresの挿入位置)
- `src/backtest/engine.py` — BacktestEngine.run() 内の feature pre-computation (training_pipelineと対称な実装が必要)

### Existing Feature Module Patterns
- `src/features/interaction_features.py` — 純粋関数パターン + INTERACTION_COLS定数 + column existence guard
- `src/features/bloodline_features.py` — クラスベース + store注入 + merge パターン
- `src/features/track_condition_data.py` — ETL用モジュール (precompute用)。Phase 48の特徴量計算モジュールとは別物

### Data Access
- `src/db/repository.py` — DataRepository.load_track_conditions(start, end) の戻り値と日付フィルタパターン
- `src/db/parquet_store.py` — ParquetStore read/write/exists パターン

### Domain & Registration
- `src/domain/types.py` — POST_RACE_COLS定義 (含水率/クッション値は含めない)
- `src/models/stage1_ability_model.py` — AbilityModel.FEATURE_COLS 登録先
- `src/models/two_stage_return_model.py` — Win/Place/Wide 2-Stage FEATURE_COLS 登録先
- `src/models/ev_correction_model.py` — EVCorrection FEATURE_COLS 登録先

### Sire & History Context
- `src/features/horse_history_features.py` — kyakusitukubun_cd の生成 (line 1346-1355)、HorseHistoryFeatures出力列
- `src/features/sire_features.py` — SireFeaturesパターン、sire_wr/sire_surface_wr 等のsire関連特徴量
- `training_pipeline.py` lines 903-904 — sire_id/bms_id mapping: `horses_df.set_index("kettonum")["ketto3infohansyokunum1"]`

### Configuration & Requirements
- `.planning/REQUIREMENTS.md` — T1-01, T1-02, T2-01, T2-02, T2-03, REG-01 要件定義
- `.planning/phases/47-etl-data-pipeline/47-CONTEXT.md` — Phase 47の決定事項(track_conditions.parquet schema、NaN処理、異常値NaN化範囲)
- `config/settings.yaml` — paths/data セクション

### Phase 36 Lesson
- Phase 36外科的ルーティングの教訓: 全モデル一律登録 → MarketModel支配 →除外で修正。Feature Routing Audit (Phase 42) は50+28の禁止特徴量をCI検証

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ParquetStore`: read/write/exists でParquet I/O。track_conditions.parquet読み込みに使用
- `DataRepository.load_track_conditions(start, end)`: 日付フィルタ付きtrack conditionsローダー。空DataFrame返却ガード付き
- `coerce_types()`: readers.py内の型強制関数
- `date_filters()`: DataRepository内の日付フィルタ生成
- `compute_interaction_features(df)`: 既存交互作用パターン — column existence guard + category型キャスト
- `frame_number`: `_map_basic_features()` で wakuban → frame_number に既に変換済み
- `kyakusitukubun_cd`: HorseHistoryFeatures出力。_train_submodel()内で利用可能
- `sire_id` / `bms_id`: _train_submodel()内で kettonum → ketto3infohansyokunum1 マップ済み

### Established Patterns
- Feature moduleパターン: `src/features/*.py` に純粋関数で実装、定数で列名管理、column existence guard
- 遅延importパターン: build_all() / _train_submodel() 内で `from features.xxx import compute_xxx` と遅延import
- TimingContext: `with TimingContext("build_all/track_condition")` でステップ計測
- Guard clause: 空DataFrame早期リターン、列存在チェック後の計算スキップ
- NaN処理: `pd.to_numeric(errors="coerce")` で安全な数値変換、LightGBMネイティブNaN対応

### Integration Points
- `FeatureEngine.build_all()`: track_conditions のマージポイント (BloodlineFeatures(store)パターンに倣う)
- `_train_submodel()` / `BacktestEngine.run()`: 交互作用特徴量の計算ポイント (HorseHistoryFeatures → track_condition_features → interaction_features の順)
- 12モデルの `FEATURE_COLS: list[str]`: 外科的ルーティングに基づく列追加
- `_register_features()` または等価の登録機構: 新特徴量のFEATURE_COLS追加

### Key Data Flow
```
build_all():
  race_df + entry_df + odds_df → merge
  → DataRepository(store).load_track_conditions() → merge on race_id  (D-01: 生値)
  → 各種特徴量モジュール (intra_race, market_bias, bloodline, ...)
  → horse_features.parquet (dirt_moisture, turf_cushion 含む)

_train_submodel() / BacktestEngine:
  horse_features + HorseHistoryFeatures → kyakusitukubun_cd 利用可能
  → compute_track_condition_features(df)  (D-02: 交互作用)
  → compute_interaction_features(df)      (既存)
  → モデル学習/推論 (外科的ルーティング適用済みFEATURE_COLS)
```

</code_context>

<specifics>
## Specific Ideas

- クッション値ビン境界 `[0, 7, 8, 9, 10, inf]` は実データ分布に基づく。元の `[0, 5]` は実データでほぼ空になるため調整
- surface-aware設計: dirt_moistureはダートレースのみ非NaN、turf_cushionは芝レースのみ非NaN。LightGBMはNaNを自動処理するが、交互作用特徴量では「関係ないsurfaceの値はNaN」を意識した設計が必要
- track_mean/track_stdは学習期間のみから計算し、dictまたはDataFrameとして保持。テスト期間へはlookup適用
- クッション値データは2020/09開始 → 2018-2020年8月の芝レースはturf_cushionがNaN (VLD-03で検証)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope
</deferred>

---

*Phase: 48-Core Edge Features*
*Context gathered: 2026-06-04*
