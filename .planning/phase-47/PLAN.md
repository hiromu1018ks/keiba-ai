# Phase 47: ETL Data Pipeline

**Goal**: 外部CSVデータ(含水率・クッション値)がParquetとしてDataRepository経由で利用可能になる
**Status**: Not started
**Depends on**: Nothing (first phase of v2.3)

## Summary

外部CSVファイル(ダート含水率・芝クッション値)を変換してParquet化し、DataRepositoryからロード可能にするETLデータパイプラインを構築する。Phase 47はデータ配管のみ — 特徴量計算はPhase 48で実装。

## Decisions

| ID | Decision | Choice | Rationale |
|----|----------|--------|-----------|
| D-01 | スクリプト構成 | 1スクリプトで両CSV処理 | データフォーマット同一、ロジック共通化可能 |
| D-02 | モジュール分離 | ロジック=src/features/track_condition_data.py、スクリプト=thin orchestrator | precompute_career_stats.pyと同じパターン |
| D-03 | Parquet構成 | 単一track_conditions.parquet (dirt_moisture + turf_cushion) | 同一race_idキーのrace-levelデータ、結合不要 |
| D-04 | race_id抽出 | entry_id.str[:16] | 既存_compute_race_idと同一16桁フォーマット |
| D-05 | 値矛盾検証 | 同一race内で値が異なる場合はValueError | データ品質担保 (理論上は全エントリ同値) |
| D-06 | NaN扱い | NaN(欠損)をそのまま保持 | 欠損は正当な状態(2018年はクッション値なし等) |
| D-07 | 物理範囲検証 | dirt_moisture: 0-15%, turf_cushion: 5-20 | JRA発表値の物理的範囲外はNaN化 |
| D-08 | DataRepository API | load_track_conditions(start, end)で一括ロード | races/entriesと同じ日付フィルタパターン |
| D-09 | FeatureEngine統合 | build_all()でrace-level left merge | Phase 48の特徴量計算のためにmerge準備 |

## Plans

### Plan 47-01: CSV-to-Parquet Conversion Module

**Objective**: 含水率・クッション値CSVを検証付きでParquetに変換するモジュールとプリコンピュートスクリプト
**Requirements**: ETL-01, ETL-02 (SC#1, SC#2)
**Autonomous**: yes

#### Task 47-01-A: 変換モジュール (src/features/track_condition_data.py)

**Create** `src/features/track_condition_data.py`

```python
# 主要関数:
def parse_track_condition_csv(csv_path: Path, value_name: str) -> pd.DataFrame:
    """CSV(18桁ID, float)をパースしてrace-levelに集約"""
    # 1. pd.read_csv(header=None, names=["entry_id", value_name])
    # 2. entry_id.str[:16] → race_id
    # 3. entry_id.str[16:18] → umaban (ログ用、保持しない)
    # 4. pd.to_datetime(race_id.str[:8]) → race_date
    # 5. race_id単位で値の一意性検証 (D-05: 異なる値があればValueError)
    # 6. drop_duplicates(subset=["race_id"]) でrace-level化

def validate_physical_range(df: pd.DataFrame, col: str, low: float, high: float) -> pd.DataFrame:
    """物理範囲外の値をNaN化 (D-07)"""
    # df.loc[~df[col].between(low, high), col] = np.nan

def convert_all(dirt_csv: Path, turf_csv: Path, output_path: Path) -> dict[str, int]:
    """両CSVを読み込み、単一Parquetに統合して出力 (D-03)"""
    # 1. parse_track_condition_csv(dirt_csv, "dirt_moisture")
    # 2. parse_track_condition_csv(turf_csv, "turf_cushion")
    # 3. validate_physical_range(dirt, "dirt_moisture", 0, 15)
    # 4. validate_physical_range(turf, "turf_cushion", 5, 20)
    # 5. pd.merge(dirt, turf, on=["race_id", "race_date"], how="outer")
    # 6. ParquetStore().write("raw", "track_conditions", merged)
    # 7. 統計ログ: 行数、NaN数、日付範囲、重複排除数
    # 8. return stats dict
```

**Output schema** (track_conditions.parquet):
```
race_id: str (16-digit, merge key)
race_date: datetime64 (for filtering)
dirt_moisture: float64 (NaN for non-dirt races or pre-2018)
turf_cushion: float64 (NaN for non-turf races or pre-2020)
```

**Files:**
- CREATE `src/features/track_condition_data.py`

#### Task 47-01-B: プリコンピュートスクリプト (scripts/precompute_track_condition.py)

**Create** `scripts/precompute_track_condition.py`

```python
"""外部CSV(含水率・クッション値)をParquetに変換するスクリプト。

Usage:
    python scripts/precompute_track_condition.py
    python scripts/precompute_track_condition.py --dirt-csv path/to/dirt.csv --turf-csv path/to/turf.csv
"""
# Thin orchestrator:
# 1. argparse: --dirt-csv, --turf-csv, --output (defaults to data/raw/track_conditions.parquet)
# 2. Resolve default CSV paths via _PROJECT_ROOT
# 3. Call convert_all()
# 4. Print summary stats
```

**Files:**
- CREATE `scripts/precompute_track_condition.py`

#### Task 47-01-C: ユニットテスト (tests/test_track_condition_data.py)

**Create** `tests/test_track_condition_data.py`

```python
# テストケース:
# 1. test_parse_extracts_race_id: 18桁ID → 16桁race_id + 2桁umaban
# 2. test_deduplicate_race_level: 同一race_id内で同じ値 → 1行に集約
# 3. test_conflicting_values_raise: 同一race_id内で異なる値 → ValueError
# 4. test_physical_range_validation: 範囲外値がNaN化される
# 5. test_convert_all_produces_parquet: convert_all → Parquet出力検証
# 6. test_merge_outer_join: dirt-only + turf-only → 外部結合でNaN列あり
# 7. test_race_date_from_id: 20200912... → 2020-09-12
# 全てmock使用、実際のCSVファイル不要
```

**Files:**
- CREATE `tests/test_track_condition_data.py`

---

### Plan 47-02: DataRepository Integration + CI Verification

**Objective**: DataRepositoryにロードメソッドを追加し、POST_RACE_COLS非包含をCI検証
**Requirements**: ETL-03, ETL-04 (SC#3, SC#4)
**Autonomous**: yes

#### Task 47-02-A: readers.py + DataRepository ロードメソッド

**Modify** `src/db/readers.py`:

```python
def load_track_conditions(store: ParquetStore, start: str, end: str) -> pd.DataFrame:
    """馬場状態(含水率・クッション値)を読み込む。"""
    df = store.read("raw", "track_conditions", filters=date_filters(start, end))
    return coerce_types(df)
```

**Modify** `src/db/repository.py`:

```python
def load_track_conditions(self, start: str, end: str) -> pd.DataFrame:
    """馬場状態(含水率・クッション値)を読み込む。

    Args:
        start: 開始日 (YYYYMMDD)
        end: 終了日 (YYYYMMDD)

    Returns:
        race_id, race_date, dirt_moisture, turf_cushionを含むDataFrame
    """
    df = self._store.read("raw", "track_conditions", filters=date_filters(start, end))
    return coerce_types(df)
```

**Also add** readers.py import in repository.py if not already present.

**Files:**
- MODIFY `src/db/readers.py` (add `load_track_conditions`)
- MODIFY `src/db/repository.py` (add `load_track_conditions` method)

#### Task 47-02-B: FeatureEngine race-level merge 準備

**Modify** `src/features/feature_engine.py`:

`build_all()` のStep 3b (steeple/excluded horse filtering, 約line 321) の直後、Step 4 (_map_basic_features, 約line 324) の前に、track_conditionデータをleft merge on race_id:

```python
# 3c. 馬場状態データのマージ (Phase 47: ETL data for Phase 48 features)
if store is not None:
    try:
        from db.repository import DataRepository
        repo = DataRepository(store)
        rd = pd.to_datetime(result_df["race_date"], errors="coerce")
        rd_valid = rd.dropna()
        if len(rd_valid) > 0:
            start_str = rd_valid.min().strftime("%Y%m%d")
            end_str = rd_valid.max().strftime("%Y%m%d")
            tc_df = repo.load_track_conditions(start_str, end_str)
            if not tc_df.empty:
                tc_cols = ["race_id", "dirt_moisture", "turf_cushion"]
                tc_available = [c for c in tc_cols if c in tc_df.columns]
                result_df = pd.merge(
                    result_df,
                    tc_df[tc_available],
                    on="race_id",
                    how="left",
                )
    except Exception:
        logger.warning("Track condition data merge skipped (non-fatal)")
```

This merge is **non-breaking**: if track_conditions.parquet doesn't exist, the try/except silently skips it. Existing behavior unchanged.

**Files:**
- MODIFY `src/features/feature_engine.py` (add track condition merge in build_all())

#### Task 47-02-C: POST_RACE_COLS CI検証テスト

**Modify** `tests/test_etl_type_conversion.py`:

```python
def test_track_condition_not_post_race(self):
    """含水率・クッション値はJRA締切前発表値なのでPOST_RACE_COLSに含まれないこと"""
    from domain.types import POST_RACE_COLS
    assert "dirt_moisture" not in POST_RACE_COLS
    assert "turf_cushion" not in POST_RACE_COLS
```

**Files:**
- MODIFY `tests/test_etl_type_conversion.py` (add test_track_condition_not_post_race)

---

## Verification Checklist

- [ ] `python scripts/precompute_track_condition.py` が両CSVを読み込み `data/raw/track_conditions.parquet` を生成
- [ ] 生成Parquetに race_id, race_date, dirt_moisture, turf_cushion 列が存在
- [ ] `python -m pytest tests/test_track_condition_data.py -v` が全テストPASS
- [ ] `python -m pytest tests/test_etl_type_conversion.py -v` がPOST_RACEテスト含めPASS
- [ ] `DataRepository().load_track_conditions("20200101", "20251231")` がDataFrameを返す
- [ ] `FeatureEngine.build_all()` がtrack_conditionデータをleft merge (既存テストが壊れない)
- [ ] `ruff check src/features/track_condition_data.py scripts/precompute_track_condition.py` PASS
- [ ] `mypy src/features/track_condition_data.py` PASS

## Execution Order

```
Plan 47-01 (Wave 1):
  47-01-A → 47-01-B (serial: module first, then script)
  47-01-C (parallel with 47-01-B: tests can be written alongside script)

Plan 47-02 (Wave 2, depends on 47-01 output schema):
  47-02-A → 47-02-B → 47-02-C (serial: readers → FeatureEngine → CI test)
```
