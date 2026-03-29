# Generic ETL: EveryDB2 全テーブル Parquet 出力 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EveryDB2 の103テーブルをYAML設定駆動の汎用ETLで全件Parquet出力し、s_ テーブルによる差分更新を可能にする。

**Architecture:** YAML設定ファイル (`config/etl_tables.yaml`) に103テーブルの定義を書き、汎用ETLエンジン (`src/db/etl.py`) がループで各テーブルを処理する。フルロード (`--mode full`) はn_テーブルからSELECT *で全カラムをそのまま出力。差分更新 (`--mode delta`) はs_テーブルを読み込み、datakubun列に基づいて既存Parquetにupsert/deleteをマージする。

**Tech Stack:** Python 3.11, pandas, pyarrow, SQLAlchemy, PyYAML, tqdm

**Spec:** `docs/superpowers/specs/2026-03-29-generic-etl-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `config/etl_tables.yaml` | Create | 103テーブル定義 (db_table, parquet_key, category, type, pk, partition_cols) |
| `src/db/etl.py` | Replace | 汎用ETLエンジン (load_config, run_full_load, run_delta_update, _merge_delta, _read_db_table, _add_race_date) |
| `scripts/run_etl.py` | Modify | `--mode full\|delta` 対応 |
| `src/db/connection.py` | Modify | `_compute_race_date` を残す, `etl_to_parquet` シグネチャ更新 |
| `src/db/repository.py` | Modify | 生カラム名対応 (race_id計算のみ追加) |
| `tests/test_etl.py` | Replace | 汎用ETLのテスト (config load, full load, delta merge) |
| `tests/test_etl_new_tables.py` | Delete | 不要 (test_etl.py に統合) |
| `data/etl_state.json` | Auto-generated | 差分管理状態 |

---

### Task 1: YAML設定ファイルの作成

**Files:**
- Create: `config/etl_tables.yaml`

- [ ] **Step 1: Create etl_tables.yaml**

spec の `### 設定ファイル (config/etl_tables.yaml)` セクションにあるYAMLをそのままコピーして作成する。103テーブル（n_ 53 + s_ 50）の定義を含む。

- [ ] **Step 2: Verify YAML is valid**

Run: `mise exec -- python -c "import yaml; data = yaml.safe_load(open('config/etl_tables.yaml')); print(f'{len(data[\"tables\"])} tables loaded')"`
Expected: `103 tables loaded`

- [ ] **Step 3: Commit**

```bash
git add config/etl_tables.yaml
git commit -m "feat: EveryDB2全103テーブルのETL設定YAMLを追加"
```

---

### Task 2: 汎用ETLエンジン — 設定ローダーとDB読み込み

**Files:**
- Replace: `src/db/etl.py`

- [ ] **Step 1: Write failing tests for config loader and DB reader**

```python
# tests/test_etl.py (新規書き換え)
"""Generic ETL engine tests (mock-based, no DB required)"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml


class TestLoadTableConfig:
    def test_loads_yaml_and_returns_list(self, tmp_path: Path):
        config_file = tmp_path / "etl_tables.yaml"
        config_file.write_text(yaml.dump({"tables": [
            {"db_table": "n_race", "parquet_key": "races", "category": "raw",
             "type": "raced", "pk": ["year", "monthday", "jyocd"]},
        ]}))
        from db.etl import load_table_config
        result = load_table_config(str(config_file))
        assert len(result) == 1
        assert result[0]["db_table"] == "n_race"
        assert result[0]["type"] == "raced"

    def test_raises_on_missing_file(self):
        from db.etl import load_table_config
        with pytest.raises(FileNotFoundError):
            load_table_config("/nonexistent/path.yaml")


class TestReadDbTable:
    @patch("db.etl.pd.read_sql")
    def test_raced_type_adds_date_filter(self, mock_read_sql):
        """type=raced のテーブルは WHERE (year||monthday)::int BETWEEN :start AND :end を付ける"""
        from db.etl import _read_db_table
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame({"year": ["2024"], "monthday": ["0101"]})

        cfg = {"db_table": "n_race", "type": "raced"}
        _read_db_table(mock_engine, cfg, start="20240101", end="20241231")

        sql_text = mock_read_sql.call_args[0][0].text
        assert "BETWEEN" in sql_text
        assert ":start" in sql_text
        assert ":end" in sql_text
        params = mock_read_sql.call_args[1].get("params", {})
        assert params.get("start") == "20240101"
        assert params.get("end") == "20241231"

    @patch("db.etl.pd.read_sql")
    def test_master_type_no_date_filter(self, mock_read_sql):
        """type=master のテーブルは SELECT * FROM table のみ (日付フィルタなし)"""
        from db.etl import _read_db_table
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame({"kettonum": ["123"]})

        cfg = {"db_table": "n_uma", "type": "master"}
        _read_db_table(mock_engine, cfg)

        sql_text = mock_read_sql.call_args[0][0].text
        assert "BETWEEN" not in sql_text
        assert "FROM n_uma" in sql_text


class TestAddRaceDate:
    def test_adds_race_date_column(self):
        from db.etl import _add_race_date
        df = pd.DataFrame({"year": ["2024"], "monthday": ["0324"]})
        result = _add_race_date(df)
        assert "race_date" in result.columns
        assert result["race_date"].iloc[0] == pd.Timestamp("2024-03-24")

    def test_preserves_existing_columns(self):
        from db.etl import _add_race_date
        df = pd.DataFrame({"year": ["2024"], "monthday": ["0101"], "jyocd": ["05"]})
        result = _add_race_date(df)
        assert "jyocd" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- python -m pytest tests/test_etl.py -v`
Expected: FAIL (ImportError — `load_table_config`, `_read_db_table`, `_add_race_date` not defined)

- [ ] **Step 3: Write the minimal implementation**

Replace `src/db/etl.py` entirely with the generic engine. Key functions:

```python
"""Generic ETL engine: EveryDB2 → Parquet (YAML-driven)"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "etl_tables.yaml"
_STATE_PATH = _PROJECT_ROOT / "data" / "etl_state.json"


def load_table_config(path: str = str(_DEFAULT_CONFIG_PATH)) -> list[dict]:
    """Load table definitions from YAML config."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["tables"]


def _read_db_table(
    engine: Engine,
    cfg: dict,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Read a table from PostgreSQL. Adds date filter for type=raced."""
    table = cfg["db_table"]
    table_type = cfg.get("type", "master")

    if table_type == "raced" and start and end:
        sql = text(
            f"SELECT * FROM {table} "
            f"WHERE (year || monthday)::int BETWEEN :start AND :end"
        )
        return pd.read_sql(sql, engine, params={"start": int(start), "end": int(end)})

    return pd.read_sql(text(f"SELECT * FROM {table}"), engine)


def _add_race_date(df: pd.DataFrame) -> pd.DataFrame:
    """Compute race_date from year + monthday columns."""
    if "year" in df.columns and "monthday" in df.columns:
        year_str = df["year"].astype(str).str.zfill(4)
        monthday_str = df["monthday"].astype(str).str.zfill(4)
        df["race_date"] = pd.to_datetime(year_str + monthday_str, format="%Y%m%d")
    return df


def _load_state() -> dict:
    """Load ETL state from JSON file."""
    if _STATE_PATH.exists():
        with open(_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"tables": {}}


def _save_state(state: dict) -> None:
    """Save ETL state to JSON file."""
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def run_full_load(
    store: ParquetStore,
    engine: Engine,
    config: list[dict],
    start: str,
    end: str,
    tables: list[str] | None = None,
) -> dict[str, int]:
    """Full load: read all n_ tables and write to Parquet."""
    state = _load_state()
    counts: dict[str, int] = {}

    # Filter to non-delta tables only
    full_configs = [c for c in config if c.get("type") != "delta"]

    # Optional table filter
    if tables:
        full_configs = [c for c in full_configs if c["parquet_key"] in tables]

    for cfg in tqdm(full_configs, desc="Full ETL"):
        key = cfg["parquet_key"]
        category = cfg["category"]
        table_type = cfg.get("type", "master")
        partition_cols = cfg.get("partition_cols")

        try:
            if partition_cols and table_type == "raced":
                # Year-by-year chunked loading for large partitioned tables
                start_year = int(start) // 10000
                end_year = int(end) // 10000
                frames = []
                for year in range(start_year, end_year + 1):
                    year_start = f"{year}0101"
                    year_end = f"{year}1231"
                    df = _read_db_table(engine, cfg, start=year_start, end=year_end)
                    if not df.empty:
                        if table_type == "raced":
                            _add_race_date(df)
                        frames.append(df)
                if frames:
                    combined = pd.concat(frames, ignore_index=True)
                    # Add partition columns from race_date
                    if "race_date" in combined.columns:
                        combined["year"] = combined["race_date"].dt.year
                        combined["month"] = combined["race_date"].dt.month
                    store.write(category, key, combined, partition_cols=partition_cols)
                    counts[key] = len(combined)
                else:
                    counts[key] = 0
            else:
                df = _read_db_table(
                    engine, cfg,
                    start=start if table_type == "raced" else None,
                    end=end if table_type == "raced" else None,
                )
                if not df.empty:
                    if table_type == "raced":
                        _add_race_date(df)
                    store.write(category, key, df)
                    counts[key] = len(df)
                else:
                    counts[key] = 0

            logger.info("Full load %s: %d rows", key, counts.get(key, 0))

        except Exception as e:
            logger.error("Full load failed for %s: %s", key, e)
            counts[key] = -1

        # Update state
        state["tables"][key] = {
            "rows": counts.get(key, 0),
            "last_full": pd.Timestamp.now().isoformat(),
        }

    _save_state(state)
    return counts


def _merge_delta(
    existing: pd.DataFrame, delta: pd.DataFrame, pk: list[str]
) -> pd.DataFrame:
    """Merge delta records into existing DataFrame using PK-based upsert/delete.

    datakubun='0' → delete row (remove from existing)
    datakubun!='0' → upsert row (replace existing or insert new)
    """
    deletes = delta[delta["datakubun"] == "0"]
    upserts = delta[delta["datakubun"] != "0"].drop(columns=["datakubun"], errors="ignore")

    # Start with existing data
    result = existing.copy()

    # Remove rows matching delete PKs
    if not deletes.empty:
        delete_keys = deletes[pk].drop_duplicates()
        merge = result.merge(
            delete_keys.assign(_delete=True), on=pk, how="left", indicator=False
        )
        result = result[merge["_delete"] != True].copy()  # noqa: E712

    # Remove rows matching upsert PKs (old versions)
    if not upserts.empty:
        upsert_keys = upserts[pk].drop_duplicates()
        merge = result.merge(
            upsert_keys.assign(_upsert=True), on=pk, how="left", indicator=False
        )
        result = result[merge["_upsert"] != True].copy()  # noqa: E712

    # Append upsert rows
    if not upserts.empty:
        result = pd.concat([result, upserts], ignore_index=True)

    return result


def run_delta_update(
    store: ParquetStore,
    engine: Engine,
    config: list[dict],
) -> dict[str, int]:
    """Delta update: read s_ tables and merge into existing Parquet files."""
    state = _load_state()
    counts: dict[str, int] = {}

    # Filter to delta tables only
    delta_configs = [c for c in config if c.get("type") == "delta"]

    for cfg in tqdm(delta_configs, desc="Delta ETL"):
        key = cfg["parquet_key"]
        category = cfg["category"]
        pk = cfg["pk"]

        try:
            # Read delta data
            delta_df = _read_db_table(engine, cfg)
            if delta_df.empty:
                counts[key] = 0
                continue

            # Check existing Parquet exists
            if not store.exists(category, key):
                logger.warning(
                    "Delta skipped for %s: no existing Parquet. Run --mode full first.", key
                )
                counts[key] = -1
                continue

            # Read existing data
            existing_df = store.read(category, key)

            # Merge
            merged = _merge_delta(existing_df, delta_df, pk)

            # Re-add race_date if needed
            table_type_raced = any(
                c["parquet_key"] == key and c.get("type") == "raced"
                for c in config if c.get("type") != "delta"
            )
            if table_type_raced and "race_date" not in merged.columns:
                _add_race_date(merged)

            # Write back
            store.write(category, key, merged)
            counts[key] = len(delta_df)

            logger.info("Delta merge %s: %d delta rows → %d total rows", key, len(delta_df), len(merged))

        except Exception as e:
            logger.error("Delta merge failed for %s: %s", key, e)
            counts[key] = -1

        # Update state
        state["tables"][key] = {
            "rows": counts.get(key, 0),
            "last_delta": pd.Timestamp.now().isoformat(),
        }

    state["last_delta_applied"] = pd.Timestamp.now().isoformat()
    _save_state(state)
    return counts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `mise exec -- python -m pytest tests/test_etl.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/etl.py tests/test_etl.py
git commit -m "feat: 汎用ETLエンジン — 設定ローダー・DB読み込み・フルロード"
```

---

### Task 3: 差分マージロジックとテスト

**Files:**
- Modify: `tests/test_etl.py` (add merge tests)
- Modify: `src/db/etl.py` (already has `_merge_delta`)

- [ ] **Step 1: Write failing tests for _merge_delta**

追加するテスト:

```python
class TestMergeDelta:
    def test_upsert_replaces_existing_row(self):
        from db.etl import _merge_delta
        existing = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
        delta = pd.DataFrame({"id": [2], "val": ["B"], "datakubun": ["1"]})
        result = _merge_delta(existing, delta, pk=["id"])
        assert len(result) == 2
        assert result[result["id"] == 2]["val"].iloc[0] == "B"

    def test_upsert_inserts_new_row(self):
        from db.etl import _merge_delta
        existing = pd.DataFrame({"id": [1], "val": ["a"]})
        delta = pd.DataFrame({"id": [2], "val": ["b"], "datakubun": ["1"]})
        result = _merge_delta(existing, delta, pk=["id"])
        assert len(result) == 2

    def test_delete_removes_row(self):
        from db.etl import _merge_delta
        existing = pd.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
        delta = pd.DataFrame({"id": [2], "val": ["x"], "datakubun": ["0"]})
        result = _merge_delta(existing, delta, pk=["id"])
        assert len(result) == 2
        assert 2 not in result["id"].values

    def test_composite_pk(self):
        from db.etl import _merge_delta
        existing = pd.DataFrame({
            "year": ["2024", "2024"], "monthday": ["0101", "0102"],
            "val": ["a", "b"]
        })
        delta = pd.DataFrame({
            "year": ["2024"], "monthday": ["0101"],
            "val": ["A"], "datakubun": ["1"]
        })
        result = _merge_delta(existing, delta, pk=["year", "monthday"])
        assert len(result) == 2
        assert result[(result["year"] == "2024") & (result["monthday"] == "0101")]["val"].iloc[0] == "A"

    def test_empty_delta_returns_existing(self):
        from db.etl import _merge_delta
        existing = pd.DataFrame({"id": [1], "val": ["a"]})
        delta = pd.DataFrame({"id": [], "val": [], "datakubun": []})
        result = _merge_delta(existing, delta, pk=["id"])
        assert len(result) == 1

    def test_datakubun_stripped_from_upserts(self):
        from db.etl import _merge_delta
        existing = pd.DataFrame({"id": [1], "val": ["a"]})
        delta = pd.DataFrame({"id": [2], "val": ["b"], "datakubun": ["1"]})
        result = _merge_delta(existing, delta, pk=["id"])
        assert "datakubun" not in result.columns
```

- [ ] **Step 2: Run tests**

Run: `mise exec -- python -m pytest tests/test_etl.py::TestMergeDelta -v`
Expected: PASS (`_merge_delta` is already implemented in Task 2)

- [ ] **Step 3: Commit**

```bash
git add tests/test_etl.py
git commit -m "test: 差分マージロジックのテストを追加"
```

---

### Task 4: run_etl.py の --mode 対応

**Files:**
- Modify: `scripts/run_etl.py`

- [ ] **Step 1: Write the updated script**

```python
"""Generic ETL: EveryDB2 → Parquet

Usage:
  python scripts/run_etl.py --mode full --start 20140101 --end 20231231
  python scripts/run_etl.py --mode delta
  python scripts/run_etl.py --mode full --tables races entries --start 20140101 --end 20231231
"""

import argparse
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic ETL: EveryDB2 → Parquet")
    parser.add_argument("--mode", choices=["full", "delta"], required=True,
                        help="full: n_テーブル全件出力, delta: s_テーブル差分マージ")
    parser.add_argument("--start", help="開始日 YYYYMMDD (full mode)")
    parser.add_argument("--end", help="終了日 YYYYMMDD (full mode)")
    parser.add_argument("--tables", nargs="*", help="対象テーブル (parquet_key指定)")
    args = parser.parse_args()

    if args.mode == "full" and (not args.start or not args.end):
        parser.error("--mode full requires --start and --end")

    from db.connection import DatabaseConnection
    from db.etl import load_table_config, run_full_load, run_delta_update
    from db.parquet_store import ParquetStore

    config = load_table_config()
    store = ParquetStore()
    db = DatabaseConnection()
    engine = db.get_engine()

    logger.info("ETL開始: mode=%s", args.mode)
    t0 = time.time()

    try:
        if args.mode == "full":
            counts = run_full_load(store, engine, config, args.start, args.end, args.tables)
        else:
            counts = run_delta_update(store, engine, config)
    except KeyboardInterrupt:
        logger.warning("ETLが中断されました")
        sys.exit(1)
    except Exception as e:
        if "could not connect" in str(e).lower() or "connection refused" in str(e).lower():
            logger.error("PostgreSQLに接続できません。localhost:5432 が実行中か確認してください。")
        else:
            logger.error("ETL失敗: %s", e)
        sys.exit(1)

    elapsed = time.time() - t0
    logger.info("ETL完了 (%.0f秒)", elapsed)

    for table, n in counts.items():
        logger.info("  %s: %d行", table, n)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Update connection.py**

`connection.py` の `etl_to_parquet` メソッドを新しい `run_full_load` に委譲するよう更新:

```python
# In connection.py, update etl_to_parquet:
def etl_to_parquet(self, store: "ParquetStore", start: str, end: str) -> dict[str, int]:
    """EveryDB2外部テーブル → Parquet にETL。"""
    from db.etl import load_table_config, run_full_load
    config = load_table_config()
    return run_full_load(store, self.get_engine(), config, start, end)
```

- [ ] **Step 3: Run existing tests to check nothing broke**

Run: `mise exec -- python -m pytest tests/test_etl.py tests/test_repository.py tests/test_db.py -v`
Expected: Some tests may fail due to import changes. Fix as needed.

- [ ] **Step 4: Delete obsolete test file**

```bash
rm tests/test_etl_new_tables.py
```

- [ ] **Step 5: Commit**

```bash
git add scripts/run_etl.py src/db/connection.py tests/test_etl.py
git rm tests/test_etl_new_tables.py
git commit -m "feat: run_etl.py --mode full|delta 対応、connection.py を新ETLに委譲"
```

---

### Task 5: DataRepository の更新 — race_id 計算追加

**Files:**
- Modify: `src/db/repository.py`

- [ ] **Step 1: Add race_id computation to repository**

repository.py の `load_races()` と `load_entries()` に `race_id` 計算を追加する。カラム名は生のまま (`year`, `monthday`, `jyocd`, `kaiji`, `nichiji`, `racenum`)。

`_exclude_steeple` の `track_cd` → `trackcd` に修正。

```python
# 変更点:
# 1. _compute_race_id ヘルパーを追加
# 2. _exclude_steeple のカラム名を trackcd に修正
# 3. load_races, load_entries で race_id を計算

def _compute_race_id(df: pd.DataFrame) -> pd.DataFrame:
    """year + monthday + jyocd + kaiji + nichiji + racenum → 16桁 race_id"""
    df["race_id"] = (
        df["year"].astype(str).str.zfill(4)
        + df["monthday"].astype(str).str.zfill(4)
        + df["jyocd"].astype(str).str.zfill(2)
        + df["kaiji"].astype(str).str.zfill(2)
        + df["nichiji"].astype(str).str.zfill(2)
        + df["racenum"].astype(str).str.zfill(2)
    )
    return df


def _exclude_steeple(df: pd.DataFrame) -> pd.DataFrame:
    """障害レース除外 (trackcd 51-59)"""
    if "trackcd" not in df.columns:
        return df
    return df[~df["trackcd"].astype(int, errors="ignore").between(51, 59)].copy()
```

- [ ] **Step 2: Run repository tests**

Run: `mise exec -- python -m pytest tests/test_repository.py -v`
Expected: May need fixes for column name changes. Update test mocks accordingly.

- [ ] **Step 3: Commit**

```bash
git add src/db/repository.py tests/test_repository.py
git commit -m "feat: DataRepositoryにrace_id計算を追加、生カラム名に対応"
```

---

### Task 6: MLパイプラインのカラム名修正

**Files:**
- Modify: `src/features/feature_engine.py` and all files referencing renamed columns
- Modify: All test files that use renamed column names

**注意:** このタスクは影響範囲が広い。`grep` で `track_cd`, `distance`, `jyo_cd`, `month_day`, `race_num`, `baba_cd`, `tenko_cd` 等の参照箇所をすべて洗い出し、生カラム名に修正する。

- [ ] **Step 1: Find all renamed column references**

```bash
grep -rn "track_cd\|distance\|jyo_cd\|month_day\|race_num\|baba_cd\|tenko_cd\|syubetu_cd\|jyoken_cd\|grade_cd\|field_size\|finish_pos\|finish_time\|win_odds\|ba_taijyu\|zogen_fugo\|zogen_sa\|kisyu_code\|chokyosi_code\|haron_time_l3\|time_diff\|corner_1c\|corner_4c\|ketto_num\|tan_umaban\|tan_pay\|fuku_umaban\|fuku_pay\|tan_odds\|fuku_odds\|odds_low\|odds_high\|race_date" src/ tests/ --include="*.py" | head -100
```

- [ ] **Step 2: Update each file**

Map of renamed → raw column names (partial list):

| Renamed | Raw |
|---------|-----|
| `track_cd` | `trackcd` |
| `distance` | `kyori` |
| `jyo_cd` | `jyocd` |
| `month_day` | `monthday` |
| `race_num` | `racenum` |
| `baba_cd` | computed from `sibababacd`/`dirtbabacd` + `trackcd` |
| `tenko_cd` | `tenkocd` |
| `finish_pos` | `kakuteijyuni` |
| `finish_time` | `time` |
| `win_odds` | `odds` (needs /10 conversion) |
| `haron_time_l3` | `harontimel3` |
| `ketto_num` | `kettonum` |
| `race_id` | computed (not in raw data) |
| `race_date` | computed (in raw data from ETL) |

- [ ] **Step 3: Run all tests**

Run: `mise exec -- python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: MLパイプラインのカラム名をEveryDB2生名に統一"
```

---

### Task 7: 既存テストのクリーンアップと統合テスト

**Files:**
- Modify: `tests/test_etl.py` (add run_full_load integration test)

- [ ] **Step 1: Add integration test for run_full_load**

```python
class TestRunFullLoad:
    @patch("db.etl._read_db_table")
    def test_processes_all_raced_and_master_tables(self, mock_read):
        """Full load processes raced + master tables, skips delta tables"""
        from db.etl import run_full_load

        mock_store = MagicMock()
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame({"year": ["2024"], "monthday": ["0101"], "col1": ["val"]})

        config = [
            {"db_table": "n_race", "parquet_key": "races", "category": "raw",
             "type": "raced", "pk": ["year", "monthday"]},
            {"db_table": "n_uma", "parquet_key": "horses", "category": "raw",
             "type": "master", "pk": ["kettonum"]},
            {"db_table": "s_race", "parquet_key": "races", "category": "raw",
             "type": "delta", "pk": ["year", "monthday"]},
        ]

        result = run_full_load(mock_store, mock_engine, config, "20240101", "20241231")

        # Should have processed 2 tables (races + horses), skipped 1 (delta)
        assert "races" in result
        assert "horses" in result
        assert mock_store.write.call_count == 2

    @patch("db.etl._read_db_table")
    def test_table_filter_limits_scope(self, mock_read):
        """--tables filter limits which tables are processed"""
        from db.etl import run_full_load

        mock_store = MagicMock()
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame({"year": ["2024"], "monthday": ["0101"]})

        config = [
            {"db_table": "n_race", "parquet_key": "races", "category": "raw",
             "type": "raced", "pk": ["year"]},
            {"db_table": "n_uma", "parquet_key": "horses", "category": "raw",
             "type": "master", "pk": ["kettonum"]},
        ]

        result = run_full_load(mock_store, mock_engine, config, "20240101", "20241231",
                               tables=["races"])

        assert "races" in result
        assert "horses" not in result


class TestRunDeltaUpdate:
    @patch("db.etl._read_db_table")
    def test_skips_when_no_existing_parquet(self, mock_read):
        """Delta skips tables with no existing Parquet file"""
        from db.etl import run_delta_update

        mock_store = MagicMock()
        mock_store.exists.return_value = False
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame({"id": [1], "datakubun": ["1"]})

        config = [
            {"db_table": "s_race", "parquet_key": "races", "category": "raw",
             "type": "delta", "pk": ["id"]},
        ]

        result = run_delta_update(mock_store, mock_engine, config)
        assert result["races"] == -1  # skipped

    @patch("db.etl._read_db_table")
    def test_merges_delta_into_existing(self, mock_read):
        """Delta merges s_ data into existing Parquet"""
        from db.etl import run_delta_update

        mock_store = MagicMock()
        mock_store.exists.return_value = True
        mock_store.read.return_value = pd.DataFrame(
            {"id": [1, 2], "val": ["a", "b"]}
        )
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame(
            {"id": [2], "val": ["B"], "datakubun": ["1"]}
        )

        config = [
            {"db_table": "s_race", "parquet_key": "races", "category": "raw",
             "type": "delta", "pk": ["id"]},
        ]

        result = run_delta_update(mock_store, mock_engine, config)
        assert result["races"] == 1  # 1 delta row processed
        mock_store.write.assert_called_once()
```

- [ ] **Step 2: Run all tests**

Run: `mise exec -- python -m pytest tests/test_etl.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_etl.py
git commit -m "test: 汎用ETLの統合テストを追加"
```

---

### Task 8: 最終確認 — 全テストパス + 型チェック

- [ ] **Step 1: Run full test suite**

Run: `mise exec -- python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Run type check**

Run: `mise exec -- python -m mypy src/db/etl.py`
Expected: No errors (or only pre-existing ones)

- [ ] **Step 3: Run lint**

Run: `mise exec -- python -m ruff check src/db/etl.py scripts/run_etl.py`
Expected: No errors

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: 汎用ETL実装完了 — 全テストパス・型チェックOK"
```
