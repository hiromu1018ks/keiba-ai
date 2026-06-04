# Phase 47: ETL Data Pipeline - Context

**Gathered:** 2026-06-04
**Status:** Ready for planning

<domain>
## Phase Boundary

外部CSVデータ(ダート含水率189K行・芝クッション値133K行)をParquetに変換し、DataRepository経由でFeatureEngineにマージ可能な単一race-level表として利用可能にする。ETL-01~04の要件を実装する。

</domain>

<decisions>
## Implementation Decisions

### Script Architecture
- **D-01:** 1スクリプト構成 — `scripts/precompute_track_condition.py` で両CSV(含水率+クッション値)を処理する。共通ロジック(ヘッダーなしCSV読み込み、集約、検証)を共有
- **D-02:** 変換ロジックは `src/features/track_condition_data.py` に配置。スクリプトはthin orchestrator(CSVパス指定・ParquetStore読み書きのみ)。Phase 48以降のFeatureEngine統合と単体テスト容易性を優先
- **D-03:** 出力は単一race-level表 `data/raw/track_conditions.parquet`。列構成: `race_id, race_date, dirt_moisture, turf_cushion`。2ファイル(dirt_moisture + turf_cushion)ではなく1テーブルに統合し、FeatureEngineからのマージを1回で完了させる

### ID Mapping & Aggregation
- **D-04:** CSVのentry_id(18桁) = race_id(先頭16桁) + umaban(末尾2桁)。race_idは先頭16桁で切り出す。※REQUIREMENTS.mdの「先頭14桁」は誤記 — Phase 47計画時に修正する
- **D-05:** 同一race_id内で非NaN値が複数種類ある場合はエラー(停止)
- **D-06:** NaNと非NaNの混在時は非NaN値を採用
- **D-07:** 既存races.parquetのrace_idとの交差検証を実施(ログ出力のみ、CSV→Parquet変換自体はraces.parquetに強依存させない)

### Missing Value & Outlier Handling
- **D-08:** 含水率/クッション値のNaNはそのまま保持。LightGBMネイティブNaN対応に任せる。統計的補完は行わない
- **D-09:** 物理的異常値のみNaN化 — 含水率は `0 < value < 100`、クッション値は `value > 0` を有効値とし、範囲外はNaN化して件数をログ出力。±3σ等の統計的外れ値処理や異常フラグ化はPhase 49で実装

### DataRepository Integration
- **D-10:** `DataRepository.load_track_conditions(start, end)` メソッドを追加。ETL-03要件を尊重
- **D-11:** 内部ロジック: `store.exists("raw", "track_conditions")` → 存在しなければ空DataFrame → 存在する場合は `date_filters(start, end)` + `store.read()` + `coerce_types()`
- **D-12:** FeatureEngineからは `DataRepository(store)` 経由で呼び出す

### Claude's Discretion
- テスト構成・テストケースの詳細設計 (既存パターンに従う)
- `_compute_race_id()` との整合性検証の実装詳細
- ログフォーマット・進捗表示の設計

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Data Pipeline Patterns
- `scripts/precompute_career_stats.py` — thin orchestratorパターン + ParquetStore書き込み
- `scripts/precompute_sire_stats.py` — スクリプト内完結パターン + to_parquet直接出力
- `src/db/readers.py` (lines 310-331) — load_career_stats/load_sire_stats standalone関数パターン
- `src/db/repository.py` — DataRepositoryクラス、既存odds loaderメソッドのパターン(start/end filter)
- `src/db/parquet_store.py` — ParquetStore read/write/exists、_optimize_dtypes

### Domain & Feature Integration
- `src/domain/types.py` (lines 38-67) — POST_RACE_COLS定義、追加方法
- `src/features/feature_engine.py` — FeatureEngine.build_all() sequential pipeline パターン、TimingContext使用

### Configuration
- `config/settings.yaml` — paths セクション(data directory設定)
- `.planning/REQUIREMENTS.md` — ETL-01~04 要件定義 (※「先頭14桁」は誤記、正しくは16桁)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ParquetStore`: read/write/exists メソッドでParquet I/O。Categorical最適化付き
- `coerce_types()`: readers.py内の型強制関数。新しいローダーでも使用
- `date_filters()`: DataRepository内の日付フィルタ生成。start/end → pyarrow pushdown filters
- `_compute_race_id()`: src/db/etl.py 内のrace_id計算ロジック。CSV IDとの整合性確認に参照

### Established Patterns
- Precompute scriptパターン: `sys.path` 設定 → ParquetStore() → load → compute → write → logging + timing
- Feature moduleパターン: `src/features/*.py` に純粋関数で実装、FeatureEngine.build_all()から遅延import
- Guard clause: 空DataFrame早期リターン、exists()チェック
- NaN処理: `pd.to_numeric(errors="coerce")` で安全な数値変換

### Integration Points
- `DataRepository.__init__()`: store注入ポイント。load_track_conditions()を追加
- `FeatureEngine.build_all()`: 新しいtrack_conditionsデータのmergeポイント (Phase 48)
- `src/domain/types.py:POST_RACE_COLS`: 含水率/クッション値はPOST_RACEに含めない(レース当日JRA発表値 = 締切前利用可能)

</code_context>

<specifics>
## Specific Ideas

- 単一Parquet `track_conditions.parquet` に含水率・クッション値を統合することで、FeatureEngine側のマージを1回で完了させる
- race_id = 先頭16桁 (REQUIREMENTS.mdの14桁は誤記)。CSV entry_id は `race_id(16桁) + umaban(2桁)` 構造
- クッション値データは2020/09開始 → 2020年以前のレースはturf_cushionがNaN。これはVLD-03(WF Fold0 NaN率検証)で確認する既知の制約

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope
</deferred>

---

*Phase: 47-ETL Data Pipeline*
*Context gathered: 2026-06-04*
