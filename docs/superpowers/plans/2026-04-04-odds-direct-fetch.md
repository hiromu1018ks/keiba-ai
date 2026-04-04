# predict でオッズを EveryDB2 から直接取得 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** predict / dry-run 実行時にオッズデータを EveryDB2 (PostgreSQL) から直接取得し、ETL delta が不要になるようにする。

**Architecture:** EveryDB2Queries に s_/n_ フォールバック付きのオッズ取得メソッドを追加し、readers.py に DB 版ローダー + 型変換 helper を追加。_run_predict() と _run_dry_run() で Parquet 版の代わりに DB 版を呼ぶ。

**Tech Stack:** Python 3.11, psycopg2, pandas, unittest.mock

**Spec:** `docs/superpowers/specs/2026-04-04-odds-direct-fetch-design.md`

---

## File Structure

| ファイル | 役割 |
|---------|------|
| `src/db/everydb2_queries.py` | `get_odds_snapshots()`, `get_odds_time_series()` 追加 |
| `src/db/readers.py` | `load_odds_snapshots_from_db()`, `load_odds_time_series_from_db()`, `_apply_odds_type_conversions()` 追加 |
| `scripts/run_paper_trading.py` | `_run_predict()`, `_run_dry_run()` のオッズ取得を DB に変更 |
| `tests/test_everydb2_queries.py` | 新メソッドのテスト追加 |
| `tests/test_readers_db.py` | DB 版ローダーのテスト (新規) |

---

### Task 1: EveryDB2Queries に get_odds_snapshots() を追加

**Files:**
- Modify: `src/db/everydb2_queries.py:161` (ファイル末尾に追加)
- Test: `tests/test_everydb2_queries.py`

- [ ] **Step 1: テストを書く**

`tests/test_everydb2_queries.py` の `TestEveryDB2Queries` クラス末尾に追加:

```python
@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_odds_snapshots_returns_raw_dataframe(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    # EveryDB2 は全列 character varying
    mock_read_sql.return_value = pd.DataFrame(
        {
            "Year": ["2026"],
            "MonthDay": ["0405"],
            "JyoCD": ["05"],
            "Kaiji": ["01"],
            "Nichiji": ["01"],
            "RaceNum": ["01"],
            "Umaban": ["1", "2", "3"],
            "TanOdds": ["32", "51", "124"],
            "FukuOddsLow": ["13", "21", "45"],
        }
    )

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_snapshots("20260405")

    assert not result.empty
    assert len(result) == 3
    # 型変換は行わない — 全列 object のまま
    assert result["Umaban"].dtype == object
    assert result.iloc[0]["TanOdds"] == "32"
    # s_ テーブルが使われることを確認
    sql_called = mock_read_sql.call_args[0][0]
    assert "s_odds_tanpuku" in sql_called

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_odds_snapshots_falls_back_to_n_table(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    # 1回目 (s_テーブル) は空、2回目 (n_テーブル) はデータあり
    mock_read_sql.side_effect = [
        pd.DataFrame(),  # s_odds_tanpuku 空
        pd.DataFrame({"Year": ["2026"], "Umaban": ["1"], "TanOdds": ["50"]}),
    ]

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_snapshots("20260405")

    assert not result.empty
    assert mock_read_sql.call_count == 2
    # 2回目の呼び出しで n_ テーブルが使われる
    second_sql = mock_read_sql.call_args_list[1][0][0]
    assert "n_odds_tanpuku" in second_sql

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_odds_snapshots_returns_empty_when_both_empty(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.return_value = pd.DataFrame()

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_snapshots("20260405")

    assert result.empty
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_everydb2_queries.py::TestEveryDB2Queries::test_get_odds_snapshots_returns_raw_dataframe -v`
Expected: FAIL — `AttributeError: 'EveryDB2Queries' object has no attribute 'get_odds_snapshots'`

- [ ] **Step 3: 実装を書く**

`src/db/everydb2_queries.py` の `get_track_condition()` メソッドの後に追加:

```python
def get_odds_snapshots(self, date_str: str) -> pd.DataFrame:
    """当日の単勝・複勝オッズスナップショットを取得。

    s_odds_tanpuku (速報系) を優先し、空なら n_odds_tanpuku (蓄積系) にフォールバック。
    戻り値は EveryDB2 生データ (全列 character varying)。型変換は呼び出し側で行う。
    """
    sql = "SELECT * FROM s_odds_tanpuku WHERE Year || MonthDay = %s"
    try:
        df = self._query(sql, (date_str,))
        if not df.empty:
            return df
    except Exception:
        logger.exception("Failed to query s_odds_tanpuku for %s", date_str)

    # フォールバック: n_odds_tanpuku
    sql = "SELECT * FROM n_odds_tanpuku WHERE Year || MonthDay = %s"
    try:
        df = self._query(sql, (date_str,))
        return df
    except Exception:
        logger.exception("Failed to query n_odds_tanpuku for %s", date_str)
        return pd.DataFrame()
```

- [ ] **Step 4: テストを実行してパスを確認**

Run: `python -m pytest tests/test_everydb2_queries.py::TestEveryDB2Queries -k "get_odds_snapshots" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: コミット**

```bash
git add src/db/everydb2_queries.py tests/test_everydb2_queries.py
git commit -m "feat: EveryDB2Queries に get_odds_snapshots() を追加"
```

---

### Task 2: EveryDB2Queries に get_odds_time_series() を追加

**Files:**
- Modify: `src/db/everydb2_queries.py`
- Test: `tests/test_everydb2_queries.py`

- [ ] **Step 1: テストを書く**

`tests/test_everydb2_queries.py` の `TestEveryDB2Queries` クラス末尾に追加:

```python
@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_odds_time_series_returns_raw_dataframe(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.return_value = pd.DataFrame(
        {
            "Year": ["2026", "2026"],
            "MonthDay": ["0405", "0405"],
            "JyoCD": ["05", "05"],
            "Kaiji": ["01", "01"],
            "Nichiji": ["01", "01"],
            "RaceNum": ["01", "01"],
            "Umaban": ["1", "1"],
            "HappyoTime": ["03101500", "03101530"],
            "TanOdds": ["35", "33"],
            "FukuOddsLow": ["14", "12"],
            "Tanninki": ["1", "1"],
        }
    )

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_time_series("20260405")

    assert not result.empty
    assert len(result) == 2
    assert result["HappyoTime"].dtype == object
    sql_called = mock_read_sql.call_args[0][0]
    assert "s_jodds_tanpuku" in sql_called

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_odds_time_series_falls_back_to_n_table(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.side_effect = [
        pd.DataFrame(),  # s_jodds_tanpuku 空
        pd.DataFrame(
            {
                "Year": ["2026"], "Umaban": ["1"], "HappyoTime": ["03101500"],
                "TanOdds": ["35"], "Tanninki": ["1"],
            }
        ),
    ]

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_time_series("20260405")

    assert not result.empty
    assert mock_read_sql.call_count == 2
    second_sql = mock_read_sql.call_args_list[1][0][0]
    assert "n_jodds_tanpuku" in second_sql
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_everydb2_queries.py -k "get_odds_time_series" -v`
Expected: FAIL

- [ ] **Step 3: 実装を書く**

`src/db/everydb2_queries.py` の `get_odds_snapshots()` の後に追加:

```python
def get_odds_time_series(self, date_str: str) -> pd.DataFrame:
    """当日の時系列オッズを取得。

    s_jodds_tanpuku (速報系) を優先し、空なら n_jodds_tanpuku (蓄積系) にフォールバック。
    戻り値は EveryDB2 生データ (全列 character varying)。型変換は呼び出し側で行う。
    """
    sql = "SELECT * FROM s_jodds_tanpuku WHERE Year || MonthDay = %s"
    try:
        df = self._query(sql, (date_str,))
        if not df.empty:
            return df
    except Exception:
        logger.exception("Failed to query s_jodds_tanpuku for %s", date_str)

    # フォールバック: n_jodds_tanpuku
    sql = "SELECT * FROM n_jodds_tanpuku WHERE Year || MonthDay = %s"
    try:
        df = self._query(sql, (date_str,))
        return df
    except Exception:
        logger.exception("Failed to query n_jodds_tanpuku for %s", date_str)
        return pd.DataFrame()
```

- [ ] **Step 4: テストを実行してパスを確認**

Run: `python -m pytest tests/test_everydb2_queries.py -k "get_odds_time_series" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: コミット**

```bash
git add src/db/everydb2_queries.py tests/test_everydb2_queries.py
git commit -m "feat: EveryDB2Queries に get_odds_time_series() を追加"
```

---

### Task 3: readers.py に _apply_odds_type_conversions() を追加

**Files:**
- Modify: `src/db/readers.py`
- Test: `tests/test_readers_db.py` (新規)

- [ ] **Step 1: テストを書く**

`tests/test_readers_db.py` を新規作成:

```python
"""readers.py の DB 版ローダー関連テスト"""

import pandas as pd
import pytest

from db.readers import _apply_odds_type_conversions


class TestApplyOddsTypeConversions:
    """_apply_odds_type_conversions のテスト"""

    def test_odds_tanpuku_converts_tanodds(self) -> None:
        """tanodds を _to_odds(v, 10) で変換: "150" -> 15.0"""
        df = pd.DataFrame(
            {
                "Year": ["2026"], "MonthDay": ["0405"],
                "Umaban": ["1"], "TanOdds": ["150"], "FukuOddsLow": ["80"],
                "JyoCD": ["05"], "Kaiji": ["01"], "Nichiji": ["01"], "RaceNum": ["01"],
            }
        )
        result = _apply_odds_type_conversions(df, "odds_tanpuku")
        assert result["tanodds"].iloc[0] == 15.0
        assert result["fukuoddslow"].iloc[0] == 8.0
        assert result["umaban"].dtype.name == "Int64"
        assert result["umaban"].iloc[0] == 1

    def test_jodds_tanpuku_converts_tanninki(self) -> None:
        """jodds_tanpuku は tanninki も Int64 に変換"""
        df = pd.DataFrame(
            {
                "Year": ["2026"], "MonthDay": ["0405"],
                "Umaban": ["3"], "TanOdds": ["55"], "FukuOddsLow": ["22"],
                "Tanninki": ["2"], "HappyoTime": ["03101500"],
                "JyoCD": ["05"], "Kaiji": ["01"], "Nichiji": ["01"], "RaceNum": ["01"],
            }
        )
        result = _apply_odds_type_conversions(df, "jodds_tanpuku")
        assert result["tanodds"].iloc[0] == 5.5
        assert result["fukuoddslow"].iloc[0] == 2.2
        assert result["umaban"].iloc[0] == 3
        assert result["tanninki"].iloc[0] == 2
        assert result["tanninki"].dtype.name == "Int64"
        # happyotime は文字列のまま
        assert result["happyotime"].iloc[0] == "03101500"
        assert result["happyotime"].dtype == object

    def test_handles_empty_values(self) -> None:
        """空文字は None になる"""
        df = pd.DataFrame(
            {
                "Year": ["2026"], "MonthDay": ["0405"],
                "Umaban": [""], "TanOdds": [""], "FukuOddsLow": [""],
                "JyoCD": ["05"], "Kaiji": ["01"], "Nichiji": ["01"], "RaceNum": ["01"],
            }
        )
        result = _apply_odds_type_conversions(df, "odds_tanpuku")
        assert pd.isna(result["tanodds"].iloc[0])
        assert pd.isna(result["umaban"].iloc[0])

    def test_unknown_table_key_returns_unchanged(self) -> None:
        """未知の table_key では何もしない"""
        df = pd.DataFrame({"Year": ["2026"], "TanOdds": ["150"]})
        result = _apply_odds_type_conversions(df, "unknown_table")
        assert result["TanOdds"].iloc[0] == "150"  # 変換されない
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_readers_db.py -v`
Expected: FAIL — `ImportError: cannot import name '_apply_odds_type_conversions'`

- [ ] **Step 3: 実装を書く**

`src/db/readers.py` の `_STRING_COLUMNS` 定義の直後、`_to_dt()` の前に追加:

```python
def _to_odds(val: object, divisor: int) -> float | None:
    """EveryDB2 のオッズ文字列を float に変換 (整数倍保存を除算)."""
    if val is None or val == "":
        return None
    try:
        return float(val) / divisor
    except (ValueError, TypeError):
        return None


def _apply_odds_type_conversions(
    df: pd.DataFrame, table_key: str
) -> pd.DataFrame:
    """EveryDB2 生データのオッズ列を Parquet 互換型に変換。

    EveryDB2 は全列 character varying で保存するため、ETL と同じ型変換を適用する。
    同時に EveryDB2 の PascalCase 列名を Parquet の snake_case にリネームする。
    """
    if table_key not in ("odds_tanpuku", "jodds_tanpuku"):
        return df

    # PascalCase → snake_case リネーム (EveryDB2 列名 → ETL 列名)
    rename_map = {
        "Year": "year", "MonthDay": "monthday", "JyoCD": "jyocd",
        "Kaiji": "kaiji", "Nichiji": "nichiji", "RaceNum": "racenum",
        "Umaban": "umaban", "TanOdds": "tanodds",
        "FukuOddsLow": "fukuoddslow", "Tanninki": "tanninki",
        "HappyoTime": "happyotime",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)

    # 型変換 (etl.py の _TABLE_TYPE_RULES と同じロジック)
    if "umaban" in df.columns:
        df["umaban"] = df["umaban"].apply(
            lambda v: int(v) if v is not None and v != "" else None
        ).astype("Int64")

    if table_key == "jodds_tanpuku" and "tanninki" in df.columns:
        df["tanninki"] = df["tanninki"].apply(
            lambda v: int(v) if v is not None and v != "" else None
        ).astype("Int64")

    for col in ("tanodds", "fukuoddslow"):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: _to_odds(v, 10))

    return df
```

- [ ] **Step 4: テストを実行してパスを確認**

Run: `python -m pytest tests/test_readers_db.py::TestApplyOddsTypeConversions -v`
Expected: PASS (4 tests)

- [ ] **Step 5: コミット**

```bash
git add src/db/readers.py tests/test_readers_db.py
git commit -m "feat: readers.py に _apply_odds_type_conversions() を追加"
```

---

### Task 4: readers.py に load_odds_snapshots_from_db() を追加

**Files:**
- Modify: `src/db/readers.py`
- Test: `tests/test_readers_db.py`

- [ ] **Step 1: テストを書く**

`tests/test_readers_db.py` に追加:

```python
from unittest.mock import MagicMock

import pandas as pd
import pytest


class TestLoadOddsSnapshotsFromDb:
    """load_odds_snapshots_from_db のテスト"""

    def test_returns_dataframe_with_race_date_and_race_id(self) -> None:
        """race_date と race_id 派生列が計算される"""
        mock_db = MagicMock()
        mock_db.get_odds_snapshots.return_value = pd.DataFrame(
            {
                "Year": ["2026"], "MonthDay": ["0405"],
                "JyoCD": ["05"], "Kaiji": ["01"], "Nichiji": ["01"], "RaceNum": ["01"],
                "Umaban": ["1"], "TanOdds": ["150"], "FukuOddsLow": ["80"],
            }
        )

        from db.readers import load_odds_snapshots_from_db

        result = load_odds_snapshots_from_db(mock_db, "20260405")

        assert "race_date" in result.columns
        assert "race_id" in result.columns
        assert result["race_id"].iloc[0] == "2026040505010101"
        assert result["tanodds"].iloc[0] == 15.0

    def test_empty_result_returns_empty_dataframe(self) -> None:
        """DB が空 DataFrame を返した場合は空のまま"""
        mock_db = MagicMock()
        mock_db.get_odds_snapshots.return_value = pd.DataFrame()

        from db.readers import load_odds_snapshots_from_db

        result = load_odds_snapshots_from_db(mock_db, "20260405")
        assert result.empty


class TestLoadOddsTimeSeriesFromDb:
    """load_odds_time_series_from_db のテスト"""

    def test_happyotime_preserved_as_string(self) -> None:
        """happyotime が _coerce_types で数値変換されない"""
        mock_db = MagicMock()
        mock_db.get_odds_time_series.return_value = pd.DataFrame(
            {
                "Year": ["2026"], "MonthDay": ["0405"],
                "JyoCD": ["05"], "Kaiji": ["01"], "Nichiji": ["01"], "RaceNum": ["01"],
                "Umaban": ["1"], "TanOdds": ["150"], "FukuOddsLow": ["80"],
                "Tanninki": ["1"], "HappyoTime": ["03101500"],
            }
        )

        from db.readers import load_odds_time_series_from_db

        result = load_odds_time_series_from_db(mock_db, "20260405")
        assert result["happyotime"].dtype == object
        assert result["happyotime"].iloc[0] == "03101500"

    def test_returns_dataframe_with_race_id(self) -> None:
        """race_id 派生列が計算される"""
        mock_db = MagicMock()
        mock_db.get_odds_time_series.return_value = pd.DataFrame(
            {
                "Year": ["2026"], "MonthDay": ["0405"],
                "JyoCD": ["05"], "Kaiji": ["01"], "Nichiji": ["01"], "RaceNum": ["01"],
                "Umaban": ["1"], "TanOdds": ["150"], "FukuOddsLow": ["80"],
                "Tanninki": ["1"], "HappyoTime": ["03101500"],
            }
        )

        from db.readers import load_odds_time_series_from_db

        result = load_odds_time_series_from_db(mock_db, "20260405")
        assert "race_id" in result.columns
        assert result["race_id"].iloc[0] == "2026040505010101"
        assert result["tanninki"].iloc[0] == 1
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_readers_db.py::TestLoadOddsSnapshotsFromDb -v`
Expected: FAIL — `ImportError: cannot import name 'load_odds_snapshots_from_db'`

- [ ] **Step 3: 実装を書く**

`src/db/readers.py` の既存 import ブロックに `TYPE_CHECKING` を追加:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.everydb2_queries import EveryDB2Queries
```

`src/db/readers.py` に `load_wide_odds()` の後に追加:

```python
def load_odds_snapshots_from_db(
    db: EveryDB2Queries, ymd: str
) -> pd.DataFrame:
    """EveryDB2 から単勝・複勝オッズスナップショットを読み込む。

    実行順序: 型変換 → race_date → race_id → _coerce_types
    (race_id のゼロパディングのため _coerce_types は最後に実行)
    """
    from db.etl import _compute_race_date, _compute_race_id

    raw = db.get_odds_snapshots(ymd)
    if raw.empty:
        return raw

    df = _apply_odds_type_conversions(raw, "odds_tanpuku")
    df = _compute_race_date(df)
    df = _compute_race_id(df)
    return _coerce_types(df)


def load_odds_time_series_from_db(
    db: EveryDB2Queries, ymd: str
) -> pd.DataFrame:
    """EveryDB2 から時系列オッズを読み込む。

    happyotime を _coerce_types からの数値変換から保護する。
    """
    from db.etl import _compute_race_date, _compute_race_id

    raw = db.get_odds_time_series(ymd)
    if raw.empty:
        return raw

    df = _apply_odds_type_conversions(raw, "jodds_tanpuku")
    df = _compute_race_date(df)
    df = _compute_race_id(df)

    # happyotime 保護: _STRING_COLUMNS に一時追加してから _coerce_types を呼ぶ
    _protected_cols = {"happyotime"} - _STRING_COLUMNS
    _STRING_COLUMNS.update(_protected_cols)
    try:
        df = _coerce_types(df)
    finally:
        _STRING_COLUMNS.difference_update(_protected_cols)

    return df
```

- [ ] **Step 4: テストを実行してパスを確認**

Run: `python -m pytest tests/test_readers_db.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: コミット**

```bash
git add src/db/readers.py tests/test_readers_db.py
git commit -m "feat: readers.py に load_odds_snapshots_from_db(), load_odds_time_series_from_db() を追加"
```

---

### Task 5: _run_predict() を DB オッズ取得に変更

**Files:**
- Modify: `scripts/run_paper_trading.py:214,229-230`

- [ ] **Step 1: _run_predict() の import とオッズ取得を変更**

`scripts/run_paper_trading.py` の `_run_predict()` 内で:

1. import 行を変更:
```python
# 変更前
from db.readers import load_entries, load_odds_snapshots, load_odds_time_series_range, load_races

# 変更後
from db.everydb2_queries import EveryDB2Queries
from db.readers import (
    load_entries,
    load_odds_snapshots_from_db,
    load_odds_time_series_from_db,
    load_races,
)
```

2. オッズ取得を変更:
```python
# 変更前
odds_df = load_odds_snapshots(store, ymd, ymd)
odds_ts_df = load_odds_time_series_range(store, ymd, ymd)

# 変更後
db = EveryDB2Queries(config.everydb2_connection_string)
odds_df = load_odds_snapshots_from_db(db, ymd)
odds_ts_df = load_odds_time_series_from_db(db, ymd)

if odds_df.empty or odds_ts_df.empty:
    logger.error("No odds data available from EveryDB2 for %s", ymd)
    return
```

3. `config` パラメータの型アノテーションを確認 — 既に `config.everydb2_connection_string` が `load_config()` で設定されているため、追加の変更は不要。

- [ ] **Step 2: 全テストを実行して回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: PASS (既存テストは Parquet 版のままなので影響なし)

- [ ] **Step 3: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: _run_predict() でオッズを EveryDB2 から直接取得"
```

---

### Task 6: _run_dry_run() を DB オッズ取得に変更

**Files:**
- Modify: `scripts/run_paper_trading.py:490,522-523`

- [ ] **Step 1: _run_dry_run() の import とオッズ取得を変更**

`scripts/run_paper_trading.py` の `_run_dry_run()` 内で:

1. import 行を変更:
```python
# 変更前
from db.readers import load_entries, load_odds_snapshots, load_odds_time_series_range, load_races

# 変更後
from db.everydb2_queries import EveryDB2Queries
from db.readers import (
    load_entries,
    load_odds_snapshots_from_db,
    load_odds_time_series_from_db,
    load_races,
)
```

2. オッズ取得を変更 (一括取得 → 連結):
```python
# 変更前
odds_df = load_odds_snapshots(store, all_start, all_end)
odds_ts_df = load_odds_time_series_range(store, all_start, all_end)

# 変更後
db = EveryDB2Queries(config.everydb2_connection_string)
odds_frames = []
odds_ts_frames = []
for d in dates:
    ymd = d.strftime("%Y%m%d")
    odds_frames.append(load_odds_snapshots_from_db(db, ymd))
    odds_ts_frames.append(load_odds_time_series_from_db(db, ymd))

odds_df = pd.concat(odds_frames, ignore_index=True) if odds_frames else pd.DataFrame()
odds_ts_df = pd.concat(odds_ts_frames, ignore_index=True) if odds_ts_frames else pd.DataFrame()

if odds_df.empty or odds_ts_df.empty:
    logger.error("No odds data available from EveryDB2 for %s ~ %s", all_start, all_end)
    return
```

- [ ] **Step 2: 全テストを実行して回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 3: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: _run_dry_run() でオッズを EveryDB2 から直接取得"
```

---

### Task 7: リント・型チェック・全テスト

**Files:** なし (検証のみ)

- [ ] **Step 1: リント**

Run: `ruff check src/db/everydb2_queries.py src/db/readers.py scripts/run_paper_trading.py`
Expected: PASS

- [ ] **Step 2: フォーマット**

Run: `ruff format --check src/db/everydb2_queries.py src/db/readers.py scripts/run_paper_trading.py`
Expected: PASS

- [ ] **Step 3: 型チェック**

Run: `mypy src/db/everydb2_queries.py src/db/readers.py`
Expected: PASS

- [ ] **Step 4: 全テスト**

Run: `python -m pytest tests/ -v`
Expected: PASS (全テスト)
