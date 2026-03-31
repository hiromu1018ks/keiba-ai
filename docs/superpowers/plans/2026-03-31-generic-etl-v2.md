# Generic ETL v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EveryDB2全テーブルをYAML設定駆動の汎用ETLでParquet出力し、DataRepositoryの一時リネームレイヤーでMLパイプラインの互換性を維持する。

**Architecture:** YAML設定 (`config/etl_tables.yaml`) に全テーブル定義を書き、汎用ETLエンジン (`src/db/etl.py`) がループで各テーブルを処理。フルロード (`--mode full`) はn_テーブルからSELECT *で全カラムをそのまま出力 + race_date/race_id計算。差分更新 (`--mode delta`) はs_テーブルをdatakubunに基づいてupsert/delete。DataRepositoryが生カラム名→ML既存名にリネーム+型変換して下流への影響をゼロにする。

**Tech Stack:** Python 3.11, pandas, pyarrow, SQLAlchemy, PyYAML, tqdm

**Spec:** `docs/superpowers/specs/2026-03-31-generic-etl-v2-design.md`
**EveryDB2テーブル定義:** `docs/everydb2/*.md` (YAML作成時に照合)
**元YAML (参考):** `docs/superpowers/specs/2026-03-29-generic-etl-design.md` の `### 設定ファイル` セクション

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `config/etl_tables.yaml` | Create | 全テーブル定義 (db_table, parquet_key, category, type, pk, partition_cols) |
| `src/db/etl.py` | Replace | 汎用ETLエンジン (load_config, run_full_load, run_delta_update, _merge_delta, _read_db_table, _compute_race_date, _compute_race_id) |
| `scripts/run_etl.py` | Modify | `--mode full\|delta` 対応 |
| `src/db/connection.py` | Modify | `_compute_race_id/date` の旧定義を削除し、etl.pyからimportして再export。`etl_to_parquet` を新ETLに委譲 |
| `src/db/repository.py` | Modify | `_transform_raw_columns()` 変換レイヤー (リネーム+型変換) を各load_*メソッドに追加 |
| `tests/test_etl.py` | Replace | 汎用ETLのテスト (config load, db reader, race_date, race_id, full load, delta merge) |
| `tests/test_etl_new_tables.py` | Delete | 不要 (旧PostgreSQL書き込み系テスト) |
| `tests/test_db.py` | Modify | `_compute_race_id/date` のテストをrawカラム名に更新 |
| `tests/test_repository.py` | Modify | リネームレイヤーのテストを追加、モックデータをrawカラム名に変更 |
| `data/etl_state.json` | Auto-generated | 差分管理状態 |

---

## 元YAMLからの修正点

元のspec (`2026-03-29-generic-etl-design.md`) のYAMLセクションをベースに、`docs/everydb2/*.md` のテーブル定義と照合して以下を修正:

1. **kisyu_seiseki PK**: `[kisyucode, setyear]` → `[kisyucode, num]` (EveryDB2仕様: Numが連番、SetYearはデータ列)
2. **chokyo_seiseki PK**: `[chokyosicode, setyear]` → `[chokyosicode, num]` (同上)
3. **s_テーブルのPK**: 対応するn_テーブルと同じPK (元のspec通り)

---

### Task 1: YAML設定ファイルの作成

**Files:**
- Create: `config/etl_tables.yaml`

- [ ] **Step 1: Create etl_tables.yaml**

元のspec (`docs/superpowers/specs/2026-03-29-generic-etl-design.md`) の `### 設定ファイル` セクションのYAMLをベースに、以下の修正を加えて `config/etl_tables.yaml` に作成:

修正点:
- `n_kisyu_seiseki` の pk: `[kisyucode, num]` に変更
- `n_chokyo_seiseki` の pk: `[chokyosicode, num]` に変更
- `s_kisyu_seiseki` の pk: `[kisyucode, num]` に変更
- `s_chokyo_seiseki` の pk: `[chokyosicode, num]` に変更

テーブル数は元のspec通り103テーブル (n_ 53 + s_ 50)。

- [ ] **Step 2: Verify YAML is valid**

Run: `mise exec -- python -c "import yaml; data = yaml.safe_load(open('config/etl_tables.yaml')); print(f'{len(data[\"tables\"])} tables loaded')"`
Expected: `103 tables loaded`

- [ ] **Step 3: Verify PK corrections**

Run: `mise exec -- python -c "import yaml; data = yaml.safe_load(open('config/etl_tables.yaml')); ks = [t for t in data['tables'] if t['db_table'] == 'n_kisyu_seiseki'][0]; print(ks['pk'])"`
Expected: `['kisyucode', 'num']`

- [ ] **Step 4: Commit**

```bash
git add config/etl_tables.yaml
git commit -m "feat: EveryDB2全103テーブルのETL設定YAMLを追加"
```

---

### Task 2: 汎用ETLエンジン — 設定ローダー・DB読み込み・計算ヘルパー

**Files:**
- Create: `tests/test_etl.py` (new version, replaces old)
- Replace: `src/db/etl.py`

- [ ] **Step 1: Write failing tests for core functions**

`tests/test_etl.py` を新規作成 (旧内容は全て削除):

```python
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
        assert params.get("start") == 20240101
        assert params.get("end") == 20241231

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

    @patch("db.etl.pd.read_sql")
    def test_delta_type_no_date_filter(self, mock_read_sql):
        """type=delta のテーブルも SELECT * FROM table のみ"""
        from db.etl import _read_db_table
        mock_engine = MagicMock()
        mock_read_sql.return_value = pd.DataFrame({"id": ["1"], "datakubun": ["1"]})

        cfg = {"db_table": "s_race", "type": "delta"}
        _read_db_table(mock_engine, cfg)

        sql_text = mock_read_sql.call_args[0][0].text
        assert "BETWEEN" not in sql_text


class TestComputeRaceDate:
    def test_adds_race_date_column(self):
        from db.etl import _compute_race_date
        df = pd.DataFrame({"year": ["2024"], "monthday": ["0324"]})
        result = _compute_race_date(df)
        assert "race_date" in result.columns
        assert result["race_date"].iloc[0] == pd.Timestamp("2024-03-24")

    def test_preserves_existing_columns(self):
        from db.etl import _compute_race_date
        df = pd.DataFrame({"year": ["2024"], "monthday": ["0101"], "jyocd": ["05"]})
        result = _compute_race_date(df)
        assert "jyocd" in result.columns

    def test_skips_when_no_year_monthday(self):
        from db.etl import _compute_race_date
        df = pd.DataFrame({"kettonum": ["123"]})
        result = _compute_race_date(df)
        assert "race_date" not in result.columns


class TestComputeRaceId:
    def test_computes_16_digit_race_id(self):
        from db.etl import _compute_race_id
        df = pd.DataFrame({
            "year": ["2024"], "monthday": ["0324"], "jyocd": ["05"],
            "kaiji": ["03"], "nichiji": ["02"], "racenum": ["08"],
        })
        result = _compute_race_id(df)
        assert result["race_id"].iloc[0] == "2024032405030208"

    def test_zfill_padding(self):
        from db.etl import _compute_race_id
        df = pd.DataFrame({
            "year": ["2024"], "monthday": ["101"], "jyocd": ["1"],
            "kaiji": ["1"], "nichiji": ["1"], "racenum": ["1"],
        })
        result = _compute_race_id(df)
        assert result["race_id"].iloc[0] == "2024010101010101"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- python -m pytest tests/test_etl.py -v`
Expected: FAIL (ImportError — `load_table_config`, `_read_db_table`, `_compute_race_date`, `_compute_race_id` not defined)

- [ ] **Step 3: Write the minimal implementation**

`src/db/etl.py` を完全にリプレイス。PostgreSQL書き込み系関数は全て削除し、汎用ETLエンジンのみを含む:

```python
"""Generic ETL engine: EveryDB2 → Parquet (YAML-driven)"""

from __future__ import annotations

import json
import logging
import shutil
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


def _compute_race_date(df: pd.DataFrame) -> pd.DataFrame:
    """Compute race_date from year + monthday columns."""
    if "year" in df.columns and "monthday" in df.columns:
        year_str = df["year"].astype(str).str.zfill(4)
        monthday_str = df["monthday"].astype(str).str.zfill(4)
        df["race_date"] = pd.to_datetime(year_str + monthday_str, format="%Y%m%d")
    return df


def _compute_race_id(df: pd.DataFrame) -> pd.DataFrame:
    """year + monthday + jyocd + kaiji + nichiji + racenum → 16桁 race_id"""
    required = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]
    if all(c in df.columns for c in required):
        df["race_id"] = (
            df["year"].astype(str).str.zfill(4)
            + df["monthday"].astype(str).str.zfill(4)
            + df["jyocd"].astype(str).str.zfill(2)
            + df["kaiji"].astype(str).str.zfill(2)
            + df["nichiji"].astype(str).str.zfill(2)
            + df["racenum"].astype(str).str.zfill(2)
        )
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
                        _compute_race_date(df)
                        _compute_race_id(df)
                        frames.append(df)
                if frames:
                    combined = pd.concat(frames, ignore_index=True)
                    # Add partition columns from race_date
                    if "race_date" in combined.columns:
                        combined["year"] = combined["race_date"].dt.year
                        combined["month"] = combined["race_date"].dt.month
                    # Clear existing partitioned data to avoid stale year partitions
                    partition_path = store.data_dir / category / key
                    if partition_path.is_dir():
                        shutil.rmtree(partition_path)
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
                        _compute_race_date(df)
                        _compute_race_id(df)
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
            is_raced = any(
                c["parquet_key"] == key and c.get("type") == "raced"
                for c in config if c.get("type") != "delta"
            )
            if is_raced:
                if "race_date" not in merged.columns:
                    _compute_race_date(merged)
                if "race_id" not in merged.columns:
                    _compute_race_id(merged)

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
Expected: PASS (TestLoadTableConfig, TestReadDbTable, TestAddRaceDate, TestComputeRaceId)

- [ ] **Step 5: Commit**

```bash
git add src/db/etl.py tests/test_etl.py
git commit -m "feat: 汎用ETLエンジン — 設定ローダー・DB読み込み・フルロード・差分マージ"
```

---

### Task 3: 差分マージとフルロードのテスト追加

**Files:**
- Modify: `tests/test_etl.py`

- [ ] **Step 1: Add delta merge tests**

`tests/test_etl.py` の末尾に追加:

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

- [ ] **Step 2: Add run_full_load and run_delta_update tests**

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

    @patch("db.etl._read_db_table")
    def test_raced_table_gets_race_date_and_race_id(self, mock_read):
        """type=raced テーブルは race_date と race_id を付与"""
        from db.etl import run_full_load

        mock_store = MagicMock()
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame({
            "year": ["2024"], "monthday": ["0101"], "jyocd": ["05"],
            "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
        })

        config = [
            {"db_table": "n_race", "parquet_key": "races", "category": "raw",
             "type": "raced", "pk": ["year", "monthday"]},
        ]

        result = run_full_load(mock_store, mock_engine, config, "20240101", "20241231")

        written_df = mock_store.write.call_args[0][2]
        assert "race_date" in written_df.columns
        assert "race_id" in written_df.columns
        assert written_df["race_id"].iloc[0] == "2024010105010101"

    @patch("db.etl._read_db_table")
    def test_master_table_no_race_date(self, mock_read):
        """type=master テーブルは race_date / race_id を付与しない"""
        from db.etl import run_full_load

        mock_store = MagicMock()
        mock_engine = MagicMock()
        mock_read.return_value = pd.DataFrame({"kettonum": ["123"]})

        config = [
            {"db_table": "n_uma", "parquet_key": "horses", "category": "raw",
             "type": "master", "pk": ["kettonum"]},
        ]

        run_full_load(mock_store, mock_engine, config, "20240101", "20241231")

        written_df = mock_store.write.call_args[0][2]
        assert "race_date" not in written_df.columns
        assert "race_id" not in written_df.columns


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

- [ ] **Step 3: Run all ETL tests**

Run: `mise exec -- python -m pytest tests/test_etl.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_etl.py
git commit -m "test: 差分マージ・フルロード・デルタ更新のテストを追加"
```

---

### Task 4: connection.py の更新

**Files:**
- Modify: `src/db/connection.py`
- Modify: `tests/test_db.py`

- [ ] **Step 1: Update connection.py**

`connection.py` の変更点:

1. **`_compute_race_id` と `_compute_race_date` の旧定義を削除** (現在の33-57行)
2. `etl.py` から import して再export (下位互換のため)
3. `etl_to_parquet` を新ETLに委譲

```python
# connection.py の変更:

# 旧定義 (_compute_race_id, _compute_race_date) を削除し、etl.pyからimport
from db.etl import _compute_race_date, _compute_race_id  # noqa: F401

# etl_to_parquet メソッドを更新:
def etl_to_parquet(self, store: "ParquetStore", start: str, end: str) -> dict[str, int]:
    """EveryDB2外部テーブル → Parquet にETL。"""
    from db.etl import load_table_config, run_full_load
    config = load_table_config()
    return run_full_load(store, self.get_engine(), config, start, end)
```

**注意:** `_compute_race_id` と `_compute_race_date` の実体は `etl.py` にある。`connection.py` は `etl.py` から import して再exportするだけ。これにより `from db.connection import _compute_race_id` を使っているコードが壊れない。

- [ ] **Step 2: Update test_db.py**

`tests/test_db.py` の `TestComputeHelpers` を更新:

```python
class TestComputeHelpers:
    def test_compute_race_id(self) -> None:
        # rawカラム名を使用 (month_day → monthday, jyo_cd → jyocd, race_num → racenum)
        df = pd.DataFrame(
            {
                "year": ["2020"],
                "monthday": ["0101"],
                "jyocd": ["5"],
                "kaiji": ["1"],
                "nichiji": ["1"],
                "racenum": ["11"],
            }
        )
        result = _compute_race_id(df)
        assert result["race_id"].iloc[0] == "2020010105010111"

    def test_compute_race_date(self) -> None:
        # rawカラム名を使用 (month_day → monthday)
        df = pd.DataFrame({"year": ["2020"], "monthday": ["0315"]})
        result = _compute_race_date(df)
        assert result["race_date"].iloc[0] == pd.Timestamp("2020-03-15")
```

import行は変更なし (`from db.connection import ...` のまま動作する)。

- [ ] **Step 3: Run db tests**

Run: `mise exec -- python -m pytest tests/test_db.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/db/connection.py tests/test_db.py
git commit -m "refactor: connection.py — compute helpersをetl.pyに移動・再export"
```

---

### Task 5: DataRepository 変換レイヤーの追加 (リネーム+型変換)

**Files:**
- Modify: `src/db/repository.py`
- Modify: `tests/test_repository.py`

**重要:** ETLは全カラムをvarcharで出力するため、RepositoryはMLパイプラインが期待する型に変換する必要がある。単なるリネームだけでなく、`_to_int`/`_to_float`/`_to_odds` 型変換もここで行う。

- [ ] **Step 1: Write failing tests for rename layer**

`tests/test_repository.py` に以下を追加:

```python
class TestTransformLayer:
    def test_load_races_renames_raw_columns(self, repo: DataRepository, mock_store: MagicMock):
        """Repository が生カラム名をML既存名にリネームする"""
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "year": [2020], "monthday": ["0601"],
            "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
            "trackcd": ["11"], "kyori": ["1600"], "tenkocd": ["1"],
            "syubetucd": ["13"], "jyokencd1": ["999"],
            "gradecd": [""], "syussotosu": ["18"],
        })
        result = repo.load_races("20200101", "20201231")
        # リネーム後のカラム名が存在すること
        assert "month_day" in result.columns
        assert "jyo_cd" in result.columns
        assert "race_num" in result.columns
        assert "track_cd" in result.columns
        assert "distance" in result.columns
        # 生カラム名は残らないこと
        assert "monthday" not in result.columns
        assert "jyocd" not in result.columns
        assert "trackcd" not in result.columns

    def test_load_entries_renames_raw_columns(self, repo: DataRepository, mock_store: MagicMock):
        """entries の生カラム名がリネームされる"""
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "year": [2020], "monthday": ["0601"],
            "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
            "umaban": ["1"], "kettonum": ["0001234567"],
            "kakuteijyuni": ["3"], "time": ["95.3"],
            "odds": ["0054"], "ninki": ["3"],
            "bataijyu": ["480"], "zogenfugo": [""], "zogensa": [""],
            "kisyucode": ["01056"], "chokyosicode": ["01023"],
            "harontimel3": ["33.5"], "timediff": ["0.3"],
            "jyuni1c": ["2"], "jyuni4c": ["3"],
            "honsyokin": ["0"], "kyakusitukubun": ["0"],
        })
        result = repo.load_entries("20200101", "20201231")
        assert "ketto_num" in result.columns
        assert "finish_pos" in result.columns
        assert "win_odds" in result.columns
        assert "kisyu_code" in result.columns
        assert "haron_time_l3" in result.columns
        assert "kettonum" not in result.columns
        assert "kakuteijyuni" not in result.columns

    def test_load_races_computes_race_id(self, repo: DataRepository, mock_store: MagicMock):
        """race_id が生カラムから計算される"""
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)],
            "year": [2020], "monthday": ["0601"],
            "jyocd": ["05"], "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"],
            "trackcd": ["11"], "kyori": ["1600"],
        })
        result = repo.load_races("20200101", "20201231")
        assert "race_id" in result.columns
        assert result["race_id"].iloc[0] == "2020060105010101"

    def test_load_races_steeple_exclusion_still_works(self, repo: DataRepository, mock_store: MagicMock):
        """障害除外が track_cd (変換後int) で動作"""
        mock_store.read.return_value = pd.DataFrame({
            "race_date": [datetime(2020, 6, 1)] * 3,
            "trackcd": ["11", "51", "55"],
            "year": [2020] * 3, "monthday": ["0601"] * 3,
            "jyocd": ["05"] * 3, "kaiji": ["01"] * 3,
            "nichiji": ["01"] * 3, "racenum": ["01"] * 3,
            "kyori": ["1600"] * 3,
        })
        result = repo.load_races("20200101", "20201231")
        assert len(result) == 1
        assert result["track_cd"].iloc[0] == 11  # int after transform
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `mise exec -- python -m pytest tests/test_repository.py::TestTransformLayer -v`
Expected: FAIL (columns like `month_day`, `ketto_num` not found)

- [ ] **Step 3: Implement transform layer in repository.py**

`repository.py` に変換ロジックを追加。リネームだけでなく型変換も行う:

```python
def _to_int(val: str | None) -> int | None:
    """空文字・非数値 → None、それ以外は int に変換"""
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _to_float(val: str | None) -> float | None:
    """空文字・非数値 → None、それ以外は float に変換"""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _to_odds(val: str | None, divisor: int = 10) -> float | None:
    """EveryDB2 オッズ文字列 → float (÷ divisor). "0054" → 5.4"""
    if val is None or val == "":
        return None
    try:
        return float(val) / divisor
    except (ValueError, TypeError):
        return None


def _transform_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """EveryDB2生カラム名をMLパイプライン互換に変換 (リネーム+型変換)。

    ETLは全カラムをvarcharで出力するため、MLパイプラインが期待する
    int/float型に変換する。Phase 2でMLパイプライン側を生カラム名に
    移行したらこの関数は削除する。
    """
    # --- リネーム ---
    rename_map = {
        # races
        "monthday": "month_day",
        "jyocd": "jyo_cd",
        "racenum": "race_num",
        "trackcd": "track_cd",
        "kyori": "distance",
        "tenkocd": "tenko_cd",
        "syubetucd": "syubetu_cd",
        "jyokencd1": "jyoken_cd",
        "gradecd": "grade_cd",
        "syussotosu": "field_size",
        # entries
        "kettonum": "ketto_num",
        "kakuteijyuni": "finish_pos",
        "time": "finish_time",
        "odds": "win_odds",
        "bataijyu": "ba_taijyu",
        "zogenfugo": "zogen_fugo",
        "zogensa": "zogen_sa",
        "kisyucode": "kisyu_code",
        "chokyosicode": "chokyosi_code",
        "harontimel3": "haron_time_l3",
        "timediff": "time_diff",
        "jyuni1c": "corner_1c",
        "jyuni4c": "corner_4c",
        "kyakusitukubun": "kyakusitu",
    }
    existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_renames:
        df = df.rename(columns=existing_renames)

    # --- 型変換 (リネーム後のカラム名で処理) ---
    int_cols = ["track_cd", "distance", "tenko_cd", "field_size",
                "umaban", "finish_pos", "ninki",
                "corner_1c", "corner_4c", "honsyokin", "kyakusitu"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_int)

    float_cols = ["finish_time", "ba_taijyu", "zogen_sa",
                  "haron_time_l3", "time_diff"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_float)

    # win_odds は ÷10 変換が必要
    if "win_odds" in df.columns:
        df["win_odds"] = df["win_odds"].apply(_to_odds)

    return df


def _compute_race_id_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """race_id を生カラム名から計算する (変換前に呼ぶこと)"""
    required = ["year", "monthday", "jyocd", "kaiji", "nichiji", "racenum"]
    if all(c in df.columns for c in required):
        df["race_id"] = (
            df["year"].astype(str).str.zfill(4)
            + df["monthday"].astype(str).str.zfill(4)
            + df["jyocd"].astype(str).str.zfill(2)
            + df["kaiji"].astype(str).str.zfill(2)
            + df["nichiji"].astype(str).str.zfill(2)
            + df["racenum"].astype(str).str.zfill(2)
        )
    return df
```

そして各 `load_*` メソッドで `_transform_raw_columns` を呼ぶ:

```python
def load_races(self, start: str, end: str) -> pd.DataFrame:
    df = self.store.read("raw", "races", filters=_date_filters(start, end))
    df = _compute_race_id_from_raw(df)
    df = _transform_raw_columns(df)
    return _exclude_steeple(df)

def load_entries(self, start: str, end: str) -> pd.DataFrame:
    df = self.store.read("raw", "entries", filters=_date_filters(start, end))
    df = _transform_raw_columns(df)
    return _exclude_steeple(df)

# load_payouts, load_odds_snapshots, load_wide_odds も同様に変換を追加
# (payouts, odds系はrace_idをParquetに持っているので_compute_race_id_from_rawは不要)
```

- [ ] **Step 4: Run repository tests**

Run: `mise exec -- python -m pytest tests/test_repository.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/repository.py tests/test_repository.py
git commit -m "feat: DataRepositoryに一時変換レイヤーを追加 (生カラム名→ML既存名+型変換)"
```

---

### Task 6: run_etl.py の --mode 対応

**Files:**
- Modify: `scripts/run_etl.py`

- [ ] **Step 1: Update run_etl.py**

```python
"""Generic ETL: EveryDB2 → Parquet

使い方:
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

- [ ] **Step 2: Commit**

```bash
git add scripts/run_etl.py
git commit -m "feat: run_etl.py --mode full|delta 対応"
```

---

### Task 7: 不要ファイルの削除とクリーンアップ

**Files:**
- Delete: `tests/test_etl_new_tables.py`

- [ ] **Step 1: Delete obsolete test file**

```bash
rm tests/test_etl_new_tables.py
```

- [ ] **Step 2: Update existing repository tests for raw column names**

`tests/test_repository.py` の既存テストのモックデータを更新。
`load_races`, `load_entries` 等のモックが `track_cd` 等のML名ではなく `trackcd` 等の生カラム名を返すように変更:

各テストの `mock_store.read.return_value` で:
- `track_cd` → `trackcd`
- `race_date` は残す (ETLが計算済み)

例:
```python
# Before:
mock_store.read.return_value = pd.DataFrame(
    {"race_date": [datetime(2020, 6, 1)], "track_cd": [10]}
)
# After:
mock_store.read.return_value = pd.DataFrame(
    {"race_date": [datetime(2020, 6, 1)], "trackcd": ["10"],
     "year": [2020], "monthday": ["0601"], "jyocd": ["05"],
     "kaiji": ["01"], "nichiji": ["01"], "racenum": ["01"], "kyori": ["1600"]}
)
```

全ての `mock_store.read.return_value` に `year`, `monthday`, `jyocd`, `kaiji`, `nichiji`, `racenum` を追加し、カラム名を生名に変更。

- [ ] **Step 3: Run full test suite**

Run: `mise exec -- python -m pytest tests/test_etl.py tests/test_repository.py tests/test_db.py tests/test_parquet_store.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_repository.py
git rm tests/test_etl_new_tables.py
git commit -m "chore: 旧テストファイル削除、リポジトリテストを生カラム名に更新"

**注意:** `s_hanro` と `s_hansyoku` はEveryDB2に存在しないため、YAMLに含めない。元のspecのs_テーブルリスト (50テーブル) が正。
```

---

### Task 8: 最終確認 — 全テストパス + 型チェック

- [ ] **Step 1: Run full test suite**

Run: `mise exec -- python -m pytest tests/ -v`
Expected: All tests pass (MLパイプラインのテスト含む)

- [ ] **Step 2: Run type check**

Run: `mise exec -- python -m mypy src/db/etl.py src/db/repository.py src/db/connection.py`
Expected: No errors (or only pre-existing ones)

- [ ] **Step 3: Run lint**

Run: `mise exec -- python -m ruff check src/db/etl.py src/db/repository.py src/db/connection.py scripts/run_etl.py`
Expected: No errors

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: 汎用ETL v1実装完了 — 全テストパス・型チェックOK"
```
