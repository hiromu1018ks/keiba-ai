# ETL + 実データ検証 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EveryDB2外部テーブルからプロジェクトスキーマへETLを実装し、既存コードのデータフロー問題を修正して、ノートブック駆動で実データ検証を可能にする。

**Architecture:** `src/db/etl.py` を新規作成してEveryDB2→プロジェクトスキーマのETLを集約。既存の `connection.py`, `feature_engine.py`, `training_pipeline.py` のデータフロー問題を修正し、ノートブックを実データ対応に改修する。

**Tech Stack:** Python 3.11, SQLAlchemy Core, psycopg2, pandas, LightGBM, Jupyter

**Spec:** `docs/superpowers/specs/2026-03-27-etl-realdata-validation-design.md`

---

## Phase 1: 既存コードのデータフロー修正

既存コードにはETL未完成による想定外のバグが7つある。ETLの前にこれらを修正する。

### Task 1: connection.py に race_date、ローダー追加、schema.py に ninki 追加

**Files:**
- Modify: `src/db/connection.py`
- Modify: `src/db/schema.py`
- Test: `tests/test_db.py`

**問題:** (a) `raw.races` に `race_date` カラムがなく、時系列特徴量がスキップされる。(b) 期間指定の時系列オッズローダーとワイドオッズローダーがない。(c) `odds_history.odds_time_series` に `ninki` カラムがなく、`popularity_change_30_10` が常にNaN。

- [ ] **Step 1: schema.py に ninki カラムを追加**

`src/db/schema.py` の `odds_history.odds_time_series` テーブルに `ninki INTEGER` を追加:

```sql
CREATE TABLE IF NOT EXISTS odds_history.odds_time_series (
    race_id     VARCHAR(16) NOT NULL,
    happyo_time VARCHAR(8) NOT NULL,
    umaban      INTEGER NOT NULL,
    tan_odds    FLOAT,
    fuku_odds   FLOAT,
    ninki       INTEGER,
    PRIMARY KEY (race_id, happyo_time, umaban)
);
```

- [ ] **Step 2: `load_races()` の戻り値に race_date を追加**

`src/db/connection.py` の `load_races()` メソッド末尾に、`year` と `month_day` から `race_date` を計算する処理を追加:

```python
def load_races(self, start_date: str, end_date: str) -> "pd.DataFrame":
    import pandas as pd
    engine = self.get_engine()
    sql = text("""
        SELECT * FROM raw.races
        WHERE (year || month_day)::int BETWEEN :start AND :end
        AND track_cd NOT BETWEEN 51 AND 59
        ORDER BY year, month_day, jyo_cd, kaiji, nichiji, race_num
    """)
    df = pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})
    # race_date を文字列 YYYYMMDD から日付型に変換
    df["race_date"] = pd.to_datetime(df["year"].astype(str) + df["month_day"], format="%Y%m%d")
    return df
```

同様に `load_entries_with_results()` にも `race_date` を追加。join で `raw.races` の `year`, `month_day` を取得できるようクエリを修正:

```python
def load_entries_with_results(self, start_date: str, end_date: str) -> "pd.DataFrame":
    import pandas as pd
    engine = self.get_engine()
    sql = text("""
        SELECT e.*, r.year, r.month_day
        FROM raw.entries e
        JOIN raw.races r ON e.race_id = r.race_id
        WHERE (r.year || r.month_day)::int BETWEEN :start AND :end
        AND r.track_cd NOT BETWEEN 51 AND 59
        AND e.finish_pos > 0
        ORDER BY e.race_id, e.umaban
    """)
    df = pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})
    df["race_date"] = pd.to_datetime(df["year"].astype(str) + df["month_day"], format="%Y%m%d")
    return df
```

- [ ] **Step 3: `load_odds_time_series_range()` を追加**

`connection.py` に期間指定の時系列オッズローダーを追加:

```python
def load_odds_time_series_range(self, start_date: str, end_date: str) -> "pd.DataFrame":
    """指定期間の時系列オッズをDataFrameで取得"""
    import pandas as pd
    engine = self.get_engine()
    sql = text("""
        SELECT o.* FROM odds_history.odds_time_series o
        JOIN raw.races r ON o.race_id = r.race_id
        WHERE (r.year || r.month_day)::int BETWEEN :start AND :end
        AND r.track_cd NOT BETWEEN 51 AND 59
        ORDER BY o.race_id, o.happyo_time, o.umaban
    """)
    return pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})
```

- [ ] **Step 4: `load_wide_odds()` を追加**

```python
def load_wide_odds(self, start_date: str, end_date: str) -> "pd.DataFrame":
    """指定期間のワイドオッズをDataFrameで取得"""
    import pandas as pd
    engine = self.get_engine()
    sql = text("""
        SELECT w.* FROM odds_history.wide_odds w
        JOIN raw.races r ON w.race_id = r.race_id
        WHERE (r.year || r.month_day)::int BETWEEN :start AND :end
        AND r.track_cd NOT BETWEEN 51 AND 59
        ORDER BY w.race_id, w.kumi
    """)
    return pd.read_sql(sql, engine, params={"start": start_date, "end": end_date})
```

- [ ] **Step 5: 既存テストを実行して回帰なしを確認**

Run: `python -m pytest tests/test_db.py -v`
Expected: 全て PASS

- [ ] **Step 6: Commit**

```bash
git add src/db/connection.py src/db/schema.py
git commit -m "fix: schemaにninki追加、connectionにrace_date/wide_oddsローダーを追加"
```

---

### Task 2: feature_engine.py のカラム名マッピング修正

**Files:**
- Modify: `src/features/feature_engine.py`
- Test: `tests/test_feature_engine.py`

**問題:** `_map_basic_features()` が `kyakusitu` を `running_style` に rename していない。また、`win_odds` を `win_odds_actual` に、`fuku_odds` を `place_odds_actual` に rename していない。これらは `training_pipeline.py` や `backtest/engine.py` が参照するカラム名。

- [ ] **Step 1: `_map_basic_features()` に rename を追加**

`src/features/feature_engine.py` の `_map_basic_features()` メソッド内に以下を追加:

```python
# running_style (kyakusitu = 脚質)
if "kyakusitu" in df.columns:
    df["running_style"] = df["kyakusitu"].fillna(0).astype(int)

# actual odds (DBのオッズと予測オッズを区別)
if "win_odds" in df.columns:
    df["win_odds_actual"] = df["win_odds"]
if "fuku_odds" in df.columns:
    df["place_odds_actual"] = df["fuku_odds"]
```

- [ ] **Step 2: テストを実行**

Run: `python -m pytest tests/test_feature_engine.py -v`
Expected: 全て PASS

- [ ] **Step 3: Commit**

```bash
git add src/features/feature_engine.py
git commit -m "fix: feature_engineにrunning_style/win_odds_actual/place_odds_actualのrenameを追加"
```

---

### Task 3: training_pipeline.py の修正

**Files:**
- Modify: `src/pipelines/training_pipeline.py`
- Test: `tests/test_training_pipeline.py`

**問題:** (a) `odds_ts_df` を `build_all()` に渡していない → オッズ動態特徴量が全てNaN。(b) 日付フォーマットが `YYYY-MM-DD` だが `connection.py` は `YYYYMMDD` を期待。(c) ワイドオッズの pivot と merge が未実装。

- [ ] **Step 1: 日付フォーマットを YYYYMMDD に変換するヘルパーを追加**

`training_pipeline.py` の `run()` メソッド冒頭にフォーマット変換を追加:

```python
def _to_yyyymmdd(date_str: str) -> str:
    """YYYY-MM-DD → YYYYMMDD"""
    return date_str.replace("-", "")
```

`run()` 内の `load_*()` 呼び出しを修正:
```python
start = self._to_yyyymmdd(train_start)
end = self._to_yyyymmdd(train_end)
race_df = self.db.load_races(start, end)
entry_df = self.db.load_entries_with_results(start, end)
odds_df = self.db.load_odds_snapshots(start, end)
```

- [ ] **Step 2: `odds_ts_df` を `build_all()` に渡す**

```python
odds_ts_df = self.db.load_odds_time_series_range(start, end)
feat_df = self.feature_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

- [ ] **Step 3: ワイドオッズの pivot と merge を追加**

`build_all()` 呼び出しの後に、ワイドオッズをロードして pivot し、feat_df に merge:

```python
wide_odds_df = self.db.load_wide_odds(start, end)
# pivot: long format → wide columns (wide_odds_1_2, wide_odds_1_3, ...)
# kumiは "1-2" 形式で保存されているため "-" を "_" に置換
# values="odds_low" のみ使用（WideJointPairBuilderは単一のオッズ値を期待）
if wide_odds_df is not None and not wide_odds_df.empty:
    wide_pivot = wide_odds_df.pivot_table(
        index="race_id", columns="kumi", values="odds_low"
    )
    wide_pivot.columns = [f"wide_odds_{kumi.replace('-', '_')}" for kumi in wide_pivot.columns]
    wide_pivot = wide_pivot.reset_index()
    feat_df = feat_df.merge(wide_pivot, on="race_id", how="left")
```

**注意:** `load_wide_odds()` は Task 1 で `connection.py` に追加済み。

- [ ] **Step 4: テストを実行**

Run: `python -m pytest tests/test_training_pipeline.py -v`
Expected: 全て PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/connection.py src/pipelines/training_pipeline.py
git commit -m "fix: training_pipelineに時系列オッズ渡し・日付変換・ワイドオッズmergeを追加"
```

---

### Task 4: backtest/engine.py の日付フォーマット修正

**Files:**
- Modify: `src/backtest/engine.py`
- Test: `tests/test_backtest_engine.py`

**問題:** (a) `run()` メソッドが `YYYY-MM-DD` 形式の日付をそのまま `connection.py` に渡している。(b) `odds_ts_df` を `build_all()` に渡していない → オッズ動態特徴量が全てNaN。

- [ ] **Step 1: 日付フォーマット変換と odds_ts_df 渡しを追加**

`run()` メソッド内の修正:

```python
start = test_start.replace("-", "")
end = test_end.replace("-", "")
# ... 既存の load_*() 呼び出し ...

# 時系列オッズをロードして build_all() に渡す
odds_ts_df = self.db.load_odds_time_series_range(start, end)
feat_df = feat_engine.build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)
```

- [ ] **Step 2: テストを実行**

Run: `python -m pytest tests/test_backtest_engine.py -v`
Expected: 全て PASS

- [ ] **Step 3: Commit**

```bash
git add src/backtest/engine.py
git commit -m "fix: backtest engineの日付フォーマットをYYYYMMDDに変換"
```

---

## Phase 2: ETL実装

### Task 5: src/db/etl.py の実装

**Files:**
- Create: `src/db/etl.py`
- Test: `tests/test_etl.py`

- [ ] **Step 1: テストを先に書く**

`tests/test_etl.py` を作成。モックを使用してETL関数の入出力を検証:

```python
"""ETLモジュールのテスト"""
from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import text


@pytest.fixture
def mock_engine():
    return MagicMock()


@pytest.fixture
def sample_n_race_df():
    """n_race のモックデータ（全カラム character varying）"""
    return pd.DataFrame({
        "year": ["2025", "2025"],
        "monthday": ["0101", "0102"],
        "jyocd": ["01", "05"],
        "kaiji": ["01", "01"],
        "nichiji": ["01", "01"],
        "racenum": ["01", "02"],
        "trackcd": ["11", "23"],
        "kyori": ["1200", "1600"],
        "tenkocd": ["1", "2"],
        "sibababacd": ["2", ""],
        "dirtbabacd": ["", "3"],
        "syubetucd": ["11", "11"],
        "jyokencd1": ["01", "02"],
        "gradecd": ["", "A"],
        "syussotosu": ["18", "16"],
    })


class TestEtlRaces:
    """etl_races のテスト"""

    def test_maps_columns_correctly(self, mock_engine, sample_n_race_df):
        """カラムマッピングが正しいこと"""
        from db.etl import etl_races
        # mock read_sql to return sample data
        with patch("db.etl.pd.read_sql", return_value=sample_n_race_df):
            etl_races(mock_engine, "20250101", "20261231")
        # verify execute was called
        assert mock_engine.begin.called or mock_engine.connect.called

    def test_baba_cd_selection(self):
        """track_cdに応じてbaba_cdを選択すること"""
        from db.etl import _select_baba_cd
        # turf → sibababacd
        assert _select_baba_cd(track_cd=11, siba="2", dirt="3") == 2
        # dirt → dirtbabacd
        assert _select_baba_cd(track_cd=23, siba="2", dirt="3") == 3
        # empty string → None
        assert _select_baba_cd(track_cd=11, siba="", dirt="3") is None

    def test_type_conversion_empty_to_null(self):
        """空文字がNULLに変換されること"""
        from db.etl import _to_int, _to_float
        assert _to_int("") is None
        assert _to_int("123") == 123
        assert _to_float("") is None
        assert _to_float("12.5") == 12.5


class TestEtlEntries:
    """etl_entries のテスト"""

    def test_race_id_generation(self):
        """race_idが正しく生成されること"""
        from db.etl import _make_race_id
        assert _make_race_id("2025", "0101", "01", "01", "01", "01") == "2025010101010101"


class TestRunFullEtl:
    """run_full_etl のテスト"""

    def test_calls_all_etl_functions(self, mock_engine):
        """全ETL関数が呼ばれること"""
        from db.etl import run_full_etl
        with patch("db.etl.etl_races"), \
             patch("db.etl.etl_entries"), \
             patch("db.etl.etl_payouts"), \
             patch("db.etl.etl_odds_snapshots"), \
             patch("db.etl.etl_wide_odds"), \
             patch("db.etl.etl_odds_timeseries"), \
             patch("db.etl.create_project_schemas"):
            run_full_etl(mock_engine, "20250101", "20261231")
```

- [ ] **Step 2: テストを実行して FAIL を確認**

Run: `python -m pytest tests/test_etl.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'db.etl')

- [ ] **Step 3: `src/db/etl.py` を実装**

```python
"""EveryDB2外部テーブル → プロジェクトスキーマ のETLモジュール"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from db.schema import ALL_CREATE_STATEMENTS

logger = logging.getLogger(__name__)


def _to_int(val: str | None) -> Optional[int]:
    """空文字→NULL、それ以外→INT変換"""
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        logger.warning("INT変換失敗: %r", val)
        return None


def _to_float(val: str | None) -> Optional[float]:
    """空文字→NULL、それ以外→FLOAT変換"""
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        logger.warning("FLOAT変換失敗: %r", val)
        return None


def _make_race_id(year: str, monthday: str, jyocd: str, kaiji: str, nichiji: str, racenum: str) -> str:
    return f"{year}{monthday}{jyocd}{kaiji}{nichiji}{racenum}"


def _select_baba_cd(track_cd: int, siba: str, dirt: str) -> Optional[int]:
    """track_cdに応じて芝/ダートの馬場状態を選択"""
    if 10 <= track_cd <= 22:
        return _to_int(siba)
    elif 23 <= track_cd <= 29:
        return _to_int(dirt)
    return None


def _insert_on_conflict(engine: Engine, df: pd.DataFrame, table: str, schema: str, pk_columns: list[str]) -> int:
    """INSERT ... ON CONFLICT DO NOTHING で冪等な挿入（psycopg2.extras.execute_values 使用）"""
    if df.empty:
        return 0
    from psycopg2.extras import execute_values
    cols = list(df.columns)
    tuples = [tuple(x) for x in df[cols].itertuples(index=False, name=None)]
    # 型変換: None を psycopg2 适配
    tuples = [tuple(None if pd.isna(v) else v for v in row) for row in tuples]
    sql = f'INSERT INTO {schema}.{table} ({", ".join(cols)}) VALUES %s ON CONFLICT DO NOTHING'
    pk_str = ", ".join(pk_columns)
    sql = f'INSERT INTO {schema}.{table} ({", ".join(cols)}) VALUES %s ON CONFLICT ({pk_str}) DO NOTHING'
    with engine.begin() as conn:
        result = conn.connection.cursor()
        execute_values(result, sql, tuples, page_size=50000)
        inserted = result.rowcount
    logger.info("%s.%s: %d 件挿入（重複除外済）", schema, table, inserted)
    return inserted


def create_project_schemas(engine: Engine) -> None:
    """全スキーマとテーブルを作成（冪等）"""
    for ddl in ALL_CREATE_STATEMENTS:
        for statement in ddl.split(";"):
            stmt = statement.strip()
            if stmt:
                with engine.begin() as conn:
                    conn.execute(text(stmt))
    logger.info("プロジェクトスキーマ作成完了")


def etl_races(engine: Engine, start: str, end: str) -> int:
    """n_race → raw.races"""
    sql = text("""
        SELECT year, monthday, jyocd, kaiji, nichiji, racenum,
               trackcd, kyori, tenkocd, sibababacd, dirtbabacd,
               syubetucd, jyokencd1, gradecd, syussotosu
        FROM n_race
        WHERE (year || monthday)::int BETWEEN :start AND :end
        AND trackcd NOT BETWEEN 51 AND 59
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.warning("etl_races: 該当データなし")
        return 0

    track_cd = df["trackcd"].apply(_to_int)
    df = df.assign(
        year=df["year"].apply(_to_int),
        month_day=df["monthday"],
        jyo_cd=df["jyocd"],
        kaiji=df["kaiji"],
        nichiji=df["nichiji"],
        race_num=df["racenum"],
        track_cd=track_cd,
        distance=df["kyori"].apply(_to_int),
        tenko_cd=df["tenkocd"].apply(_to_int),
        baba_cd=df.apply(lambda r: _select_baba_cd(
            _to_int(r["trackcd"]), r["sibababacd"], r["dirtbabacd"]
        ), axis=1),
        syubetu_cd=df["syubetucd"],
        jyoken_cd=df["jyokencd1"],
        grade_cd=df["gradecd"].apply(lambda x: x if x and x.strip() else "_"),
        field_size=df["syussotosu"].apply(_to_int),
    )

    out = df[["year", "month_day", "jyo_cd", "kaiji", "nichiji", "race_num",
              "track_cd", "distance", "tenko_cd", "baba_cd",
              "syubetu_cd", "jyoken_cd", "grade_cd", "field_size"]]

    return _insert_on_conflict(engine, out, "races", "raw",
                               ["year", "month_day", "jyo_cd", "kaiji", "nichiji", "race_num"])


def etl_entries(engine: Engine, start: str, end: str) -> int:
    """n_uma_race → raw.entries（SQL JOINでFK整合性を保証）"""
    sql = text("""
        SELECT u.year, u.monthday, u.jyocd, u.kaiji, u.nichiji, u.racenum,
               u.umaban, u.kettonum, u.kakuteijyuni, u.time, u.odds, u.ninki,
               u.bataijyu, u.zogenfugo, u.zogensa, u.kisyucode, u.chokyosicode,
               u.harontimel3, u.honsyokin, u.kyakusitukubun
        FROM n_uma_race u
        JOIN raw.races r
            ON u.year = r.year AND u.monthday = r.month_day
            AND u.jyocd = r.jyo_cd AND u.kaiji = r.kaiji
            AND u.nichiji = r.nichiji AND u.racenum = r.race_num
        WHERE (u.year || u.monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.warning("etl_entries: 該当データなし")
        return 0

    df = df.assign(
        race_id=df.apply(lambda r: _make_race_id(
            r["year"], r["monthday"], r["jyocd"], r["kaiji"], r["nichiji"], r["racenum"]
        ), axis=1),
        umaban=df["umaban"].apply(_to_int),
        ketto_num=df["kettonum"],
        finish_pos=df["kakuteijyuni"].apply(_to_int),
        finish_time=df["time"].apply(_to_float),
        win_odds=df["odds"].apply(_to_float),
        ninki=df["ninki"].apply(_to_int),
        ba_taijyu=df["bataijyu"].apply(_to_float),
        zogen_fugo=df["zogenfugo"].apply(_to_int),
        zogen_sa=df["zogensa"].apply(_to_float),
        kisyu_code=df["kisyucode"],
        chokyosi_code=df["chokyosicode"],
        haron_time_l3=df["harontimel3"].apply(_to_float),
        honsyokin=df["honsyokin"].apply(_to_int),
        kyakusitu=df["kyakusitukubun"].apply(_to_int),
    )

    out = df[["race_id", "umaban", "ketto_num", "finish_pos", "finish_time",
              "haron_time_l3", "ninki", "win_odds", "ba_taijyu",
              "zogen_fugo", "zogen_sa", "kisyu_code", "chokyosi_code",
              "kyakusitu", "honsyokin"]]

    return _insert_on_conflict(engine, out, "entries", "raw", ["race_id", "umaban"])


def etl_payouts(engine: Engine, start: str, end: str) -> int:
    """n_harai → raw.payouts（SQL JOINでFK整合性を保証）"""
    sql = text("""
        SELECT r.race_id,
               h.paytansyoumaban1, h.paytansyopay1,
               h.payfukusyoumaban1, h.payfukusyopay1,
               h.payfukusyoumaban2, h.payfukusyopay2,
               h.payfukusyoumaban3, h.payfukusyopay3,
               h.payfukusyoumaban4, h.payfukusyopay4,
               h.payfukusyoumaban5, h.payfukusyopay5
        FROM n_harai h
        JOIN raw.races r
            ON h.year = r.year AND h.monthday = r.month_day
            AND h.jyocd = r.jyo_cd AND h.kaiji = r.kaiji
            AND h.nichiji = r.nichiji AND h.racenum = r.race_num
        WHERE (h.year || h.monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.warning("etl_payouts: 該当データなし")
        return 0

    out = df.rename(columns={
        "paytansyoumaban1": "tan_umaban", "paytansyopay1": "tan_pay",
        "payfukusyoumaban1": "fuku_umaban1", "payfukusyopay1": "fuku_pay1",
        "payfukusyoumaban2": "fuku_umaban2", "payfukusyopay2": "fuku_pay2",
        "payfukusyoumaban3": "fuku_umaban3", "payfukusyopay3": "fuku_pay3",
        "payfukusyoumaban4": "fuku_umaban4", "payfukusyopay4": "fuku_pay4",
        "payfukusyoumaban5": "fuku_umaban5", "payfukusyopay5": "fuku_pay5",
    })

    return _insert_on_conflict(engine, out, "payouts", "raw", ["race_id"])


def etl_odds_snapshots(engine: Engine, start: str, end: str) -> int:
    """n_odds_tanpuku → odds_history.odds_snapshots（SQL JOINでFK整合性を保証）"""
    sql = text("""
        SELECT r.race_id, o.umaban, o.tanodds, o.fukuoddslow
        FROM n_odds_tanpuku o
        JOIN raw.races r
            ON o.year = r.year AND o.monthday = r.month_day
            AND o.jyocd = r.jyo_cd AND o.kaiji = r.kaiji
            AND o.nichiji = r.nichiji AND o.racenum = r.race_num
        WHERE (o.year || o.monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.warning("etl_odds_snapshots: 該当データなし")
        return 0

    out = df.assign(
        umaban=df["umaban"].apply(_to_int),
        tan_odds=df["tanodds"].apply(_to_float),
        fuku_odds=df["fukuoddslow"].apply(_to_float),
    )[["race_id", "umaban", "tan_odds", "fuku_odds"]]

    return _insert_on_conflict(engine, out, "odds_snapshots", "odds_history", ["race_id", "umaban"])


def etl_wide_odds(engine: Engine, start: str, end: str) -> int:
    """n_odds_wide → odds_history.wide_odds（SQL JOINでFK整合性を保証）"""
    sql = text("""
        SELECT r.race_id, w.kumi, w.oddslow, w.oddshigh
        FROM n_odds_wide w
        JOIN raw.races r
            ON w.year = r.year AND w.monthday = r.month_day
            AND w.jyocd = r.jyo_cd AND w.kaiji = r.kaiji
            AND w.nichiji = r.nichiji AND w.racenum = r.race_num
        WHERE (w.year || w.monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.warning("etl_wide_odds: 該当データなし")
        return 0

    out = df.assign(
        odds_low=df["oddslow"].apply(_to_float),
        odds_high=df["oddshigh"].apply(_to_float),
    )[["race_id", "kumi", "odds_low", "odds_high"]]

    return _insert_on_conflict(engine, out, "wide_odds", "odds_history", ["race_id", "kumi"])


def etl_odds_timeseries(engine: Engine, start: str, end: str) -> int:
    """n_jodds_tanpuku → odds_history.odds_time_series（年次分割 + SQL JOIN + 冪等）"""
    total = 0
    years = pd.read_sql(
        text("SELECT DISTINCT year::int FROM n_jodds_tanpuku WHERE (year || monthday)::int BETWEEN :start AND :end ORDER BY year"),
        engine, params={"start": start, "end": end}
    )["year"].tolist()

    for year in years:
        sql = text("""
            SELECT r.race_id, j.happyo_time, j.umaban, j.tanodds, j.fukuoddslow, j.tanninki
            FROM n_jodds_tanpuku j
            JOIN raw.races r
                ON j.year = r.year AND j.monthday = r.month_day
                AND j.jyocd = r.jyo_cd AND j.kaiji = r.kaiji
                AND j.nichiji = r.nichiji AND j.racenum = r.race_num
            WHERE j.year = :year
        """)
        df = pd.read_sql(sql, engine, params={"year": str(year)})

        if df.empty:
            continue

        out = df.assign(
            umaban=df["umaban"].apply(_to_int),
            tan_odds=df["tanodds"].apply(_to_float),
            fuku_odds=df["fukuoddslow"].apply(_to_float),
            ninki=df["tanninki"].apply(_to_int),
        )[["race_id", "happyo_time", "umaban", "tan_odds", "fuku_odds", "ninki"]]

        n = _insert_on_conflict(engine, out, "odds_time_series", "odds_history",
                                ["race_id", "happyo_time", "umaban"])
        total += n
        logger.info("etl_odds_timeseries: %d年 %d 件挿入", year, n)

    logger.info("etl_odds_timeseries: 合計 %d 件挿入完了", total)
    return total


def run_full_etl(engine: Engine, start: str, end: str) -> dict[str, int]:
    """全ETLを一括実行（冪等）"""
    create_project_schemas(engine)
    counts: dict[str, int] = {}
    counts["races"] = etl_races(engine, start, end)
    counts["entries"] = etl_entries(engine, start, end)
    counts["payouts"] = etl_payouts(engine, start, end)
    counts["odds_snapshots"] = etl_odds_snapshots(engine, start, end)
    counts["wide_odds"] = etl_wide_odds(engine, start, end)
    counts["odds_time_series"] = etl_odds_timeseries(engine, start, end)
    logger.info("ETL完了: %s", counts)
    return counts
```

- [ ] **Step 4: テストを実行して PASS を確認**

Run: `python -m pytest tests/test_etl.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/db/etl.py tests/test_etl.py
git commit -m "feat: EveryDB2→プロジェクトスキーマのETLモジュールを実装"
```

---

## Phase 3: ノートブック改修

### Task 6: notebooks/00_setup.ipynb の修正

**Files:**
- Modify: `notebooks/00_setup.ipynb`

**問題:** `load_races()` が `race_date` カラムを参照しているが `raw.races` に存在しない。ETL実行セルがない。

- [ ] **Step 1: `load_races()` を修正**

`raw.races` の `year||month_day` を使うクエリに変更:

```python
def load_races(engine, start, end):
    sql = text("""
        SELECT * FROM raw.races
        WHERE (year || month_day)::int BETWEEN :start AND :end
        AND track_cd NOT BETWEEN 51 AND 59
        ORDER BY year, month_day, jyo_cd, kaiji, nichiji, race_num
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})
    df["race_date"] = pd.to_datetime(df["year"].astype(str) + df["month_day"], format="%Y%m%d")
    return df
```

- [ ] **Step 2: ETL実行セルを追加**

```python
from db.etl import run_full_etl
counts = run_full_etl(get_engine(), "20150101", "20261231")
print("ETL完了:", counts)
```

- [ ] **Step 3: 検証セルを追加**

```python
for schema, table in [("raw", "races"), ("raw", "entries"), ("raw", "payouts"),
                       ("odds_history", "odds_snapshots"), ("odds_history", "wide_odds"),
                       ("odds_history", "odds_time_series")]:
    cnt = pd.read_sql(text(f"SELECT count(*) FROM {schema}.{table}"), get_engine()).iloc[0, 0]
    print(f"{schema}.{table}: {cnt:,} 件")
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/00_setup.ipynb
git commit -m "fix: 00_setupのDBクエリ修正とETL実行セルを追加"
```

---

### Task 7: notebooks/01_eda.ipynb の修正

**Files:**
- Modify: `notebooks/01_eda.ipynb`

**問題:** `race_date`, `popularity_rank`, `win_odds` の参照が不正。

- [ ] **Step 1: データロードを `connection.py` のメソッドに変更**

`load_races()` と `load_entries_with_results()` を使うように変更。`race_date` は `load_races()` が生成する。`popularity_rank` は `ninki` を参照。`win_odds` は entries 側に join する。

- [ ] **Step 2: グラフ生成コードのカラム参照を修正**

- `df_races["race_date"].dt.year` → OK（修正済みload_racesが生成）
- `df_races.groupby("popularity_rank")` → `df_entries.groupby("ninki")` に変更
- `df_races["win_odds"]` → `df_entries["win_odds"]` に変更
- 日付フォーマット `"2018-01-01"` → `"20180101"` に変更

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_eda.ipynb
git commit -m "fix: 01_edaのカラム参照と日付フォーマットを修正"
```

---

### Task 8: notebooks/02_odds_dynamics.ipynb の修正

**Files:**
- Modify: `notebooks/02_odds_dynamics.ipynb`

**問題:** テーブル名 `odds.odds_timeseries` が間違っている。`happyo_time` フィルターが不正。

- [ ] **Step 1: テーブル名とクエリを修正**

- `odds.odds_timeseries` → `odds_history.odds_time_series`
- `WHERE happyo_time >= '2020-01-01'` → `WHERE race_id IN (SELECT race_id FROM raw.races WHERE (year || month_day)::int >= 20200101)`
- Mock data の `happyo_time` を `MMDDHHmm` 形式に修正

- [ ] **Step 2: Commit**

```bash
git add notebooks/02_odds_dynamics.ipynb
git commit -m "fix: 02_odds_dynamicsのテーブル名とクエリを修正"
```

---

### Task 9: notebooks/01b_feature_engineering.ipynb を新規作成

**Files:**
- Create: `notebooks/01b_feature_engineering.ipynb`

- [ ] **Step 1: ノートブックを作成**

内容:
1. `%run "./00_setup.ipynb"` でセットアップ
2. `connection.py` から race_df, entry_df, odds_df, odds_ts_df をロード
3. `FeatureEngine().build_all(race_df, entry_df, odds_df, odds_ts_df=odds_ts_df)` を実行
4. 各特徴量の `describe()` で分布確認
5. `df.isnull().sum()` で欠損値確認
6. `validate_no_future_leakage()` を実行

- [ ] **Step 2: Commit**

```bash
git add notebooks/01b_feature_engineering.ipynb
git commit -m "feat: 特徴量エンジン検証ノートブックを作成"
```

---

### Task 10: notebooks/11_holdout_final_evaluation.ipynb の修正

**Files:**
- Modify: `notebooks/11_holdout_final_evaluation.ipynb`

**問題:** バックテストコードがコメントアウトされている。日付フォーマットが不正。

- [ ] **Step 1: バックテストコードを有効化**

日付フォーマットを `YYYYMMDD` に修正:
```python
from pipelines.training_pipeline import TrainingPipelineV5
from backtest.engine import BacktestEngine

pipeline = TrainingPipelineV5()
models = pipeline.run("20150101", "20211231")

engine = BacktestEngine(models, initial_bankroll=100000)
result = engine.run("20220101", "20241231")
print(f"Total bets: {result.total_bets}")
print(f"ROI: {result.total_roi:.1%}")
print(f"Max DD: {result.max_drawdown:.1%}")
print(f"Final bankroll: {result.final_bankroll:,.0f}")
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/11_holdout_final_evaluation.ipynb
git commit -m "fix: 11_holdoutのバックテストコードを有効化"
```

---

## Phase 4: 全体テストとリント

### Task 11: 全テスト + リント + 型チェック

- [ ] **Step 1: 全テスト実行**

Run: `python -m pytest tests/ -v`
Expected: 全て PASS

- [ ] **Step 2: リント**

Run: `ruff check src/ tests/`
Expected: エラーなし

- [ ] **Step 3: 型チェック**

Run: `mypy src/`
Expected: エラーなし（または既存の警告のみ）

- [ ] **Step 4: 最終コミット（必要に応じて）**
