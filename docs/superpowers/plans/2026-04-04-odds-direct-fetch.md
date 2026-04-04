# predict で EveryDB2 から直接データ取得 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** predict / dry-run / setup 実行時に races, entries, odds を EveryDB2 から直接取得し、ETL delta が不要になるようにする。

**Architecture:** EveryDB2Queries に s_/n_ フォールバック付きの4つのクエリメソッドを追加し、readers.py に DB 版ローダーを追加。`etl._apply_type_conversions` を再利用して型変換を行い、_run_setup/predict/dry_run で Parquet 版を DB 版に置き換える。

**Tech Stack:** Python 3.11, psycopg2, pandas, unittest.mock

**Spec:** `docs/superpowers/specs/2026-04-04-odds-direct-fetch-design.md`

---

## File Structure

| ファイル | 役割 |
|---------|------|
| `src/db/everydb2_queries.py` | `get_races()`, `get_entries()`, `get_odds_snapshots()`, `get_odds_time_series()` 追加 |
| `src/db/readers.py` | `load_races_from_db()`, `load_entries_from_db()`, `load_odds_snapshots_from_db()`, `load_odds_time_series_from_db()` 追加 |
| `scripts/run_paper_trading.py` | `_run_setup()`, `_run_predict()`, `_run_dry_run()` のデータ取得を DB に変更 |
| `tests/test_everydb2_queries.py` | 新メソッドのテスト追加 |
| `tests/test_readers_db.py` | DB 版ローダーのテスト (新規) |

---

### Task 1: EveryDB2Queries に get_races() と get_entries() を追加

**Files:**
- Modify: `src/db/everydb2_queries.py:160` (ファイル末尾に追加)
- Test: `tests/test_everydb2_queries.py`

- [ ] **Step 1: テストを書く**

`tests/test_everydb2_queries.py` の `TestEveryDB2Queries` クラス末尾に追加:

```python
# --- get_races / get_entries テスト ---

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_races_returns_raw_dataframe(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.return_value = pd.DataFrame(
        {
            "year": ["2026"], "monthday": ["0405"],
            "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
            "trackcd": ["11"], "kyori": ["1200"], "tenkocd": ["2"],
        }
    )

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_races("20260405")

    assert not result.empty
    assert len(result) == 1
    # 型変換は行わない — 全列 object のまま
    assert result["trackcd"].dtype == object
    assert result.iloc[0]["trackcd"] == "11"
    # s_ テーブルが使われることを確認
    sql_called = mock_read_sql.call_args[0][0]
    assert "s_race" in sql_called

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_races_falls_back_to_n_table(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.side_effect = [
        pd.DataFrame(),  # s_race 空
        pd.DataFrame({"year": ["2026"], "trackcd": ["11"]}),
    ]

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_races("20260405")

    assert not result.empty
    assert mock_read_sql.call_count == 2
    second_sql = mock_read_sql.call_args_list[1][0][0]
    assert "n_race" in second_sql

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_races_returns_empty_when_both_empty(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.return_value = pd.DataFrame()

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_races("20260405")

    assert result.empty

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_entries_returns_raw_dataframe(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.return_value = pd.DataFrame(
        {
            "year": ["2026"], "monthday": ["0405"],
            "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
            "umaban": ["1", "2", "3"], "kettonum": ["0012345678", "0012345679", "0012345680"],
        }
    )

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_entries("20260405")

    assert not result.empty
    assert len(result) == 3
    assert result["umaban"].dtype == object
    sql_called = mock_read_sql.call_args[0][0]
    assert "s_uma_race" in sql_called

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_entries_falls_back_to_n_table(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.side_effect = [
        pd.DataFrame(),
        pd.DataFrame({"year": ["2026"], "umaban": ["1"]}),
    ]

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_entries("20260405")

    assert not result.empty
    assert mock_read_sql.call_count == 2
    second_sql = mock_read_sql.call_args_list[1][0][0]
    assert "n_uma_race" in second_sql

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_races_returns_empty_on_db_error(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.side_effect = Exception("Connection refused")

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_races("20260405")
    assert result.empty

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_entries_returns_empty_on_db_error(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.side_effect = Exception("Connection refused")

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_entries("20260405")
    assert result.empty
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_everydb2_queries.py::TestEveryDB2Queries -k "get_races or get_entries" -v`
Expected: FAIL — `AttributeError: 'EveryDB2Queries' object has no attribute 'get_races'`

- [ ] **Step 3: 実装を書く**

`src/db/everydb2_queries.py` の `get_track_condition()` メソッドの後に追加:

```python
    def get_races(self, date_str: str) -> pd.DataFrame:
        """当日のレース情報を取得。s_race → n_race フォールバック。

        戻り値は EveryDB2 生データ (全列 character varying)。型変換は呼び出し側で行う。
        """
        sql = "SELECT * FROM s_race WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            if not df.empty:
                return df
        except Exception:
            logger.exception("Failed to query s_race for %s", date_str)

        sql = "SELECT * FROM n_race WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            return df
        except Exception:
            logger.exception("Failed to query n_race for %s", date_str)
            return pd.DataFrame()

    def get_entries(self, date_str: str) -> pd.DataFrame:
        """当日の出走馬を取得。s_uma_race → n_uma_race フォールバック。

        戻り値は EveryDB2 生データ (全列 character varying)。型変換は呼び出し側で行う。
        """
        sql = "SELECT * FROM s_uma_race WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            if not df.empty:
                return df
        except Exception:
            logger.exception("Failed to query s_uma_race for %s", date_str)

        sql = "SELECT * FROM n_uma_race WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            return df
        except Exception:
            logger.exception("Failed to query n_uma_race for %s", date_str)
            return pd.DataFrame()
```

- [ ] **Step 4: テストを実行してパスを確認**

Run: `python -m pytest tests/test_everydb2_queries.py::TestEveryDB2Queries -k "get_races or get_entries" -v`
Expected: PASS (7 tests)

- [ ] **Step 5: コミット**

```bash
git add src/db/everydb2_queries.py tests/test_everydb2_queries.py
git commit -m "feat: EveryDB2Queries に get_races(), get_entries() を追加"
```

---

### Task 2: EveryDB2Queries に get_odds_snapshots() と get_odds_time_series() を追加

**Files:**
- Modify: `src/db/everydb2_queries.py`
- Test: `tests/test_everydb2_queries.py`

- [ ] **Step 1: テストを書く**

`tests/test_everydb2_queries.py` の `TestEveryDB2Queries` クラス末尾に追加:

```python
# --- get_odds_snapshots / get_odds_time_series テスト ---

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_odds_snapshots_returns_raw_dataframe(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.return_value = pd.DataFrame(
        {
            "year": ["2026"], "monthday": ["0405"],
            "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
            "umaban": ["1", "2", "3"],
            "tanodds": ["32", "51", "124"],
            "fukuoddslow": ["13", "21", "45"],
        }
    )

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_snapshots("20260405")

    assert not result.empty
    assert len(result) == 3
    assert result["umaban"].dtype == object
    assert result.iloc[0]["tanodds"] == "32"
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

    mock_read_sql.side_effect = [
        pd.DataFrame(),
        pd.DataFrame({"year": ["2026"], "umaban": ["1"], "tanodds": ["50"]}),
    ]

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_snapshots("20260405")

    assert not result.empty
    assert mock_read_sql.call_count == 2
    second_sql = mock_read_sql.call_args_list[1][0][0]
    assert "n_odds_tanpuku" in second_sql

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
            "year": ["2026", "2026"],
            "monthday": ["0405", "0405"],
            "jyocd": ["05", "05"],
            "kaiji": ["01", "01"], "nichiji": ["01", "01"], "racenum": ["01", "01"],
            "umaban": ["1", "1"],
            "happyotime": ["03101500", "03101530"],
            "tanodds": ["35", "33"],
            "fukuoddslow": ["14", "12"],
            "tanninki": ["1", "1"],
        }
    )

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_time_series("20260405")

    assert not result.empty
    assert len(result) == 2
    assert result["happyotime"].dtype == object
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
        pd.DataFrame(),
        pd.DataFrame(
            {
                "year": ["2026"], "umaban": ["1"], "happyotime": ["03101500"],
                "tanodds": ["35"], "tanninki": ["1"],
            }
        ),
    ]

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_time_series("20260405")

    assert not result.empty
    assert mock_read_sql.call_count == 2
    second_sql = mock_read_sql.call_args_list[1][0][0]
    assert "n_jodds_tanpuku" in second_sql

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_odds_snapshots_returns_empty_on_db_error(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.side_effect = Exception("Connection refused")

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_snapshots("20260405")
    assert result.empty

@patch("db.everydb2_queries.pd.read_sql_query")
@patch("db.everydb2_queries.psycopg2.connect")
def test_get_odds_time_series_returns_empty_on_db_error(
    self, mock_connect: MagicMock, mock_read_sql: MagicMock
) -> None:
    from db.everydb2_queries import EveryDB2Queries

    mock_conn = MagicMock()
    mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
    mock_connect.return_value.__exit__ = MagicMock(return_value=False)

    mock_read_sql.side_effect = Exception("Connection refused")

    queries = EveryDB2Queries(connection_string="postgresql://localhost/everydb2")
    result = queries.get_odds_time_series("20260405")
    assert result.empty
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_everydb2_queries.py -k "get_odds" -v`
Expected: FAIL

- [ ] **Step 3: 実装を書く**

`src/db/everydb2_queries.py` の `get_entries()` メソッドの後に追加:

```python
    def get_odds_snapshots(self, date_str: str) -> pd.DataFrame:
        """当日の単勝・複勝オッズスナップショットを取得。s_odds_tanpuku → n_odds_tanpuku フォールバック。

        戻り値は EveryDB2 生データ (全列 character varying)。型変換は呼び出し側で行う。
        """
        sql = "SELECT * FROM s_odds_tanpuku WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            if not df.empty:
                return df
        except Exception:
            logger.exception("Failed to query s_odds_tanpuku for %s", date_str)

        sql = "SELECT * FROM n_odds_tanpuku WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            return df
        except Exception:
            logger.exception("Failed to query n_odds_tanpuku for %s", date_str)
            return pd.DataFrame()

    def get_odds_time_series(self, date_str: str) -> pd.DataFrame:
        """当日の時系列オッズを取得。s_jodds_tanpuku → n_jodds_tanpuku フォールバック。

        戻り値は EveryDB2 生データ (全列 character varying)。型変換は呼び出し側で行う。
        """
        sql = "SELECT * FROM s_jodds_tanpuku WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            if not df.empty:
                return df
        except Exception:
            logger.exception("Failed to query s_jodds_tanpuku for %s", date_str)

        sql = "SELECT * FROM n_jodds_tanpuku WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            return df
        except Exception:
            logger.exception("Failed to query n_jodds_tanpuku for %s", date_str)
            return pd.DataFrame()
```

- [ ] **Step 4: テストを実行してパスを確認**

Run: `python -m pytest tests/test_everydb2_queries.py -k "get_odds" -v`
Expected: PASS (6 tests)

- [ ] **Step 5: コミット**

```bash
git add src/db/everydb2_queries.py tests/test_everydb2_queries.py
git commit -m "feat: EveryDB2Queries に get_odds_snapshots(), get_odds_time_series() を追加"
```

---

### Task 3: readers.py に 4つの DB 版ローダーを追加

**Files:**
- Modify: `src/db/readers.py`
- Test: `tests/test_readers_db.py` (新規)

- [ ] **Step 1: テストを書く**

`tests/test_readers_db.py` を新規作成:

```python
"""readers.py の DB 版ローダーのテスト"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from db.readers import (
    load_entries_from_db,
    load_odds_snapshots_from_db,
    load_odds_time_series_from_db,
    load_races_from_db,
)


class TestLoadRacesFromDb:
    """load_races_from_db のテスト"""

    def test_applies_type_conversions_and_derives_race_id(self) -> None:
        """型変換 → race_date → race_id → _coerce_types → _exclude_steeple が適用される"""
        mock_db = MagicMock()
        mock_db.get_races.return_value = pd.DataFrame(
            {
                "year": ["2026"], "monthday": ["0405"],
                "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
                "trackcd": ["11"], "kyori": ["1200"], "tenkocd": ["2"],
                "syussotosu": ["18"], "honsyokin": ["10000000"],
            }
        )

        result = load_races_from_db(mock_db, "20260405")

        assert not result.empty
        assert "race_date" in result.columns
        assert "race_id" in result.columns
        assert result["race_id"].iloc[0] == "2026040505010101"
        # trackcd が Int64 に変換される
        assert result["trackcd"].dtype.name == "Int64"
        assert result["trackcd"].iloc[0] == 11

    def test_excludes_steeple_races(self) -> None:
        """障害レース (trackcd 51-59) が除外される"""
        mock_db = MagicMock()
        mock_db.get_races.return_value = pd.DataFrame(
            {
                "year": ["2026", "2026"],
                "monthday": ["0405", "0405"],
                "jyocd": ["05", "05"],
                "kaiji": ["01", "01"], "nichiji": ["01", "01"], "racenum": ["01", "02"],
                "trackcd": ["11", "55"],  # 芝 + 障害
            }
        )

        result = load_races_from_db(mock_db, "20260405")

        assert len(result) == 1
        assert result["trackcd"].iloc[0] == 11

    def test_empty_result_returns_empty_dataframe(self) -> None:
        mock_db = MagicMock()
        mock_db.get_races.return_value = pd.DataFrame()

        result = load_races_from_db(mock_db, "20260405")
        assert result.empty


class TestLoadEntriesFromDb:
    """load_entries_from_db のテスト"""

    def test_applies_type_conversions_and_derives_race_id(self) -> None:
        mock_db = MagicMock()
        mock_db.get_entries.return_value = pd.DataFrame(
            {
                "year": ["2026"], "monthday": ["0405"],
                "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
                "umaban": ["1", "2"], "kettonum": ["0012345678", "0012345679"],
                "kakuteijyuni": ["1", ""], "ninki": ["1", "3"],
            }
        )

        result = load_entries_from_db(mock_db, "20260405")

        assert not result.empty
        assert result["race_id"].iloc[0] == "2026040505010101"
        assert result["umaban"].dtype.name == "Int64"
        assert result["umaban"].iloc[0] == 1
        # 空文字は NA になる
        assert pd.isna(result["kakuteijyuni"].iloc[1])

    def test_excludes_steeple_entries(self) -> None:
        mock_db = MagicMock()
        mock_db.get_entries.return_value = pd.DataFrame(
            {
                "year": ["2026", "2026"],
                "monthday": ["0405", "0405"],
                "jyocd": ["05", "05"],
                "kaiji": ["01", "01"], "nichiji": ["01", "01"], "racenum": ["01", "01"],
                "umaban": ["1", "2"], "trackcd": ["11", "55"],
            }
        )

        result = load_entries_from_db(mock_db, "20260405")
        assert len(result) == 1


class TestLoadOddsSnapshotsFromDb:
    """load_odds_snapshots_from_db のテスト"""

    def test_converts_odds_with_divisor_10(self) -> None:
        """tanodds, fukuoddslow を /10 で変換: "150" -> 15.0"""
        mock_db = MagicMock()
        mock_db.get_odds_snapshots.return_value = pd.DataFrame(
            {
                "year": ["2026"], "monthday": ["0405"],
                "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
                "umaban": ["1"], "tanodds": ["150"], "fukuoddslow": ["80"],
            }
        )

        result = load_odds_snapshots_from_db(mock_db, "20260405")

        assert result["tanodds"].iloc[0] == 15.0
        assert result["fukuoddslow"].iloc[0] == 8.0
        assert result["umaban"].dtype.name == "Int64"
        assert result["umaban"].iloc[0] == 1

    def test_empty_result_returns_empty_dataframe(self) -> None:
        mock_db = MagicMock()
        mock_db.get_odds_snapshots.return_value = pd.DataFrame()

        result = load_odds_snapshots_from_db(mock_db, "20260405")
        assert result.empty


class TestLoadOddsTimeSeriesFromDb:
    """load_odds_time_series_from_db のテスト"""

    def test_happyotime_preserved_as_string(self) -> None:
        """happyotime が _coerce_types で数値変換されない"""
        mock_db = MagicMock()
        mock_db.get_odds_time_series.return_value = pd.DataFrame(
            {
                "year": ["2026"], "monthday": ["0405"],
                "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
                "umaban": ["1"], "tanodds": ["150"], "fukuoddslow": ["80"],
                "tanninki": ["1"], "happyotime": ["03101500"],
            }
        )

        result = load_odds_time_series_from_db(mock_db, "20260405")

        assert result["happyotime"].dtype == object
        assert result["happyotime"].iloc[0] == "03101500"
        assert result["tanninki"].dtype.name == "Int64"
        assert result["tanninki"].iloc[0] == 1

    def test_converts_odds_and_derives_race_id(self) -> None:
        mock_db = MagicMock()
        mock_db.get_odds_time_series.return_value = pd.DataFrame(
            {
                "year": ["2026"], "monthday": ["0405"],
                "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
                "umaban": ["1"], "tanodds": ["55"], "fukuoddslow": ["22"],
                "tanninki": ["2"], "happyotime": ["03101500"],
            }
        )

        result = load_odds_time_series_from_db(mock_db, "20260405")

        assert result["race_id"].iloc[0] == "2026040505010101"
        assert result["tanodds"].iloc[0] == 5.5
        assert result["tanninki"].iloc[0] == 2
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `python -m pytest tests/test_readers_db.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_races_from_db'`

- [ ] **Step 3: 実装を書く**

`src/db/readers.py` の import ブロックに `TYPE_CHECKING` を追加:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from db.everydb2_queries import EveryDB2Queries
```

`src/db/readers.py` の `_to_dt()` 関数の前に、import ブロックの後に追加:

```python
from db.etl import _apply_type_conversions, _compute_race_date, _compute_race_id


def load_races_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame:
    """EveryDB2 からレース情報を読み込む。"""
    raw = db.get_races(ymd)
    if raw.empty:
        return raw
    df = _apply_type_conversions(raw, "races")
    df = _compute_race_date(df)
    df = _compute_race_id(df)
    df = _coerce_types(df)
    return _exclude_steeple(df)


def load_entries_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame:
    """EveryDB2 から出走馬を読み込む。"""
    raw = db.get_entries(ymd)
    if raw.empty:
        return raw
    df = _apply_type_conversions(raw, "entries")
    df = _compute_race_date(df)
    df = _compute_race_id(df)
    df = _coerce_types(df)
    return _exclude_steeple(df)


def load_odds_snapshots_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame:
    """EveryDB2 から単勝・複勝オッズスナップショットを読み込む。"""
    raw = db.get_odds_snapshots(ymd)
    if raw.empty:
        return raw
    df = _apply_type_conversions(raw, "odds_tanpuku")
    df = _compute_race_date(df)
    df = _compute_race_id(df)
    return _coerce_types(df)


def load_odds_time_series_from_db(db: EveryDB2Queries, ymd: str) -> pd.DataFrame:
    """EveryDB2 から時系列オッズを読み込む。happyotime を _coerce_types から保護。"""
    raw = db.get_odds_time_series(ymd)
    if raw.empty:
        return raw
    df = _apply_type_conversions(raw, "jodds_tanpuku")
    df = _compute_race_date(df)
    df = _compute_race_id(df)

    # happyotime 保護: _STRING_COLUMNS に一時追加してから _coerce_types を呼ぶ
    # 注意: _STRING_COLUMNS はモジュールレベルの set であるためスレッドセーフでないが、
    # 現状の実行パスはシングルスレッドなので問題なし
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
Expected: PASS (9 tests)

- [ ] **Step 5: コミット**

```bash
git add src/db/readers.py tests/test_readers_db.py
git commit -m "feat: readers.py に 4つの DB 版ローダーを追加"
```

---

### Task 4: _run_setup() を DB 化

**Files:**
- Modify: `scripts/run_paper_trading.py:143-155`

- [ ] **Step 1: import とデータ取得を変更**

`scripts/run_paper_trading.py` の `_run_setup()` 内で:

1. import 行を変更 (古い import を削除して新しい import に置き換える):
```python
# 変更前
from db.readers import load_entries, load_races

# 変更後
from db.everydb2_queries import EveryDB2Queries
from db.readers import load_entries_from_db, load_races_from_db
```

2. データ取得を変更:
```python
# 変更前
race_df = load_races(store, ymd, ymd)
entry_df = load_entries(store, ymd, ymd)

# 変更後
db = EveryDB2Queries(config.everydb2_connection_string)
race_df = load_races_from_db(db, ymd)
entry_df = load_entries_from_db(db, ymd)
```

- [ ] **Step 2: 全テストを実行して回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: PASS (既存テストは Parquet 版のままなので影響なし)

- [ ] **Step 3: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: _run_setup() でレース・出走馬を EveryDB2 から直接取得"
```

---

### Task 5: _run_predict() を DB 化

**Files:**
- Modify: `scripts/run_paper_trading.py:207-230`

- [ ] **Step 1: import とデータ取得を変更**

`scripts/run_paper_trading.py` の `_run_predict()` 内で:

1. import 行を変更 (古い import を削除して新しい import に置き換える):
```python
# 変更前
from db.readers import load_entries, load_odds_snapshots, load_odds_time_series_range, load_races

# 変更後
from db.everydb2_queries import EveryDB2Queries
from db.readers import (
    load_entries_from_db,
    load_odds_snapshots_from_db,
    load_odds_time_series_from_db,
    load_races_from_db,
)
```

2. データ取得を変更:
```python
# 変更前
race_df = load_races(store, ymd, ymd)
entry_df = load_entries(store, ymd, ymd)
odds_df = load_odds_snapshots(store, ymd, ymd)
odds_ts_df = load_odds_time_series_range(store, ymd, ymd)

if race_df.empty or entry_df.empty:
    logger.error("No race/entry data for %s", args.date)
    return

# 変更後
db = EveryDB2Queries(config.everydb2_connection_string)
race_df = load_races_from_db(db, ymd)
entry_df = load_entries_from_db(db, ymd)
odds_df = load_odds_snapshots_from_db(db, ymd)
odds_ts_df = load_odds_time_series_from_db(db, ymd)

if race_df.empty or entry_df.empty or odds_df.empty or odds_ts_df.empty:
    logger.error("EveryDB2 からデータ取得失敗: %s", ymd)
    return
```

- [ ] **Step 2: 全テストを実行して回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 3: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: _run_predict() で全データを EveryDB2 から直接取得"
```

---

### Task 6: _run_dry_run() を DB 化

**Files:**
- Modify: `scripts/run_paper_trading.py:483-523`

- [ ] **Step 1: import とデータ取得を変更**

`scripts/run_paper_trading.py` の `_run_dry_run()` 内で:

1. import 行を変更 (古い import を削除して新しい import に置き換える):
```python
# 変更前
from db.readers import load_entries, load_odds_snapshots, load_odds_time_series_range, load_races

# 変更後
from db.everydb2_queries import EveryDB2Queries
from db.readers import (
    load_entries_from_db,
    load_odds_snapshots_from_db,
    load_odds_time_series_from_db,
    load_races_from_db,
)
```

2. データ取得を変更:
```python
# 変更前
race_df = load_races(store, all_start, all_end)
entry_df = load_entries(store, all_start, all_end)
odds_df = load_odds_snapshots(store, all_start, all_end)
odds_ts_df = load_odds_time_series_range(store, all_start, all_end)

if race_df.empty:
    logger.error("No race data found")
    sys.exit(1)

# 変更後
db = EveryDB2Queries(config.everydb2_connection_string)
race_frames, entry_frames, odds_frames, odds_ts_frames = [], [], [], []
for d in dates:
    ymd = d.strftime("%Y%m%d")
    race_frames.append(load_races_from_db(db, ymd))
    entry_frames.append(load_entries_from_db(db, ymd))
    odds_frames.append(load_odds_snapshots_from_db(db, ymd))
    odds_ts_frames.append(load_odds_time_series_from_db(db, ymd))

race_df = pd.concat(race_frames, ignore_index=True) if race_frames else pd.DataFrame()
entry_df = pd.concat(entry_frames, ignore_index=True) if entry_frames else pd.DataFrame()
odds_df = pd.concat(odds_frames, ignore_index=True) if odds_frames else pd.DataFrame()
odds_ts_df = pd.concat(odds_ts_frames, ignore_index=True) if odds_ts_frames else pd.DataFrame()

if race_df.empty or entry_df.empty or odds_df.empty or odds_ts_df.empty:
    logger.error("EveryDB2 からデータ取得失敗: %s ~ %s", all_start, all_end)
    return
```

- [ ] **Step 2: 全テストを実行して回帰がないことを確認**

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 3: コミット**

```bash
git add scripts/run_paper_trading.py
git commit -m "feat: _run_dry_run() で全データを EveryDB2 から直接取得"
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
