"""EveryDB2 → プロジェクトスキーマ ETL モジュール

EveryDB2外部テーブル(n_race, n_uma_race, n_harai, n_odds_tanpuku,
n_odds_wide, n_jodds_tanpuku) からプロジェクトスキーマ(raw.*, odds_history.*)
へデータをコピーする。
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from db.schema import ALL_CREATE_STATEMENTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _to_int(val: str | None) -> Optional[int]:
    """空文字 → None、それ以外は int に変換"""
    if val is None or val == "":
        return None
    return int(val)


def _to_float(val: str | None) -> Optional[float]:
    """空文字 → None、それ以外は float に変換"""
    if val is None or val == "":
        return None
    return float(val)


def _make_race_id(
    year: str, monthday: str, jyocd: str, kaiji: str, nichiji: str, racenum: str
) -> str:
    """複合PKから race_id を生成"""
    return f"{year}{monthday}{jyocd}{kaiji}{nichiji}{racenum}"


def _select_baba_cd(track_cd: int, siba: str, dirt: str) -> Optional[int]:
    """トラックコードに応じて芝/ダートの馬場状態を選択

    turf (10-22) → siba, dirt (23-29) → dirt
    """
    if 10 <= track_cd <= 22:
        return _to_int(siba)
    elif 23 <= track_cd <= 29:
        return _to_int(dirt)
    return None


def _insert_on_conflict(
    engine: Engine,
    df: pd.DataFrame,
    table: str,
    schema: str,
    pk_columns: list[str],
) -> int:
    """psycopg2.extras.execute_values で ON CONFLICT DO NOTHING 挿入（冪等）

    Parameters
    ----------
    engine : SQLAlchemy Engine
    df : 挿入するDataFrame
    table : テーブル名
    schema : スキーマ名
    pk_columns : 主キーカラムのリスト

    Returns
    -------
    int : 挿入件数
    """
    if df.empty:
        return 0

    from psycopg2.extras import execute_values

    cols = list(df.columns)
    tuples = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df[cols].itertuples(index=False, name=None)
    ]
    pk_str = ", ".join(pk_columns)
    sql = (
        f'INSERT INTO {schema}.{table} ({", ".join(cols)}) '
        f"VALUES %s ON CONFLICT ({pk_str}) DO NOTHING"
    )
    with engine.begin() as conn:
        result = conn.connection.cursor()
        execute_values(result, sql, tuples, page_size=50000)
        inserted = result.rowcount
    logger.info("%s.%s: %d 件挿入", schema, table, inserted)
    return inserted


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


def create_project_schemas(engine: Engine) -> None:
    """ALL_CREATE_STATEMENTS DDL を実行して全スキーマ・テーブルを作成"""
    for ddl in ALL_CREATE_STATEMENTS:
        for statement in ddl.split(";"):
            stmt = statement.strip()
            if stmt:
                with engine.begin() as conn:
                    conn.execute(text(stmt))
    logger.info("全スキーマ作成完了")


# ---------------------------------------------------------------------------
# ETL functions
# ---------------------------------------------------------------------------


def etl_races(engine: Engine, start: str, end: str) -> int:
    """n_race → raw.races

    Parameters
    ----------
    engine : SQLAlchemy Engine
    start : 開始日 (YYYYMMDD形式 int文字列)
    end : 終了日 (YYYYMMDD形式 int文字列)

    Returns
    -------
    int : 挿入件数
    """
    sql = text("""
        SELECT
            year,
            monthday,
            jyocd,
            kaiji,
            nichiji,
            racenum,
            trackcd,
            kyori,
            tenkocd,
            sibababacd,
            dirtbabacd,
            syubetucd,
            jyokencd1,
            gradecd,
            syussotosu
        FROM n_race
        WHERE (year || monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.info("etl_races: 該当データなし")
        return 0

    # 型変換
    df["year"] = df["year"].apply(_to_int)
    df["month_day"] = df["monthday"]
    df["jyo_cd"] = df["jyocd"]
    df["race_num"] = df["racenum"]
    df["track_cd"] = df["trackcd"].apply(_to_int)
    df["distance"] = df["kyori"].apply(_to_int)
    df["tenko_cd"] = df["tenkocd"].apply(_to_int)
    df["baba_cd"] = df.apply(
        lambda row: _select_baba_cd(
            row["track_cd"], row["sibababacd"], row["dirtbabacd"]
        ),
        axis=1,
    )
    df["syubetu_cd"] = df["syubetucd"]
    df["jyoken_cd"] = df["jyokencd1"]
    df["grade_cd"] = df["gradecd"].apply(lambda x: x if x and x != "" else "_")
    df["field_size"] = df["syussotosu"].apply(_to_int)

    # 不要カラムを削除
    out = df[
        [
            "year",
            "month_day",
            "jyo_cd",
            "kaiji",
            "nichiji",
            "race_num",
            "track_cd",
            "distance",
            "tenko_cd",
            "baba_cd",
            "syubetu_cd",
            "jyoken_cd",
            "grade_cd",
            "field_size",
        ]
    ]

    return _insert_on_conflict(
        engine,
        out,
        "races",
        "raw",
        ["year", "month_day", "jyo_cd", "kaiji", "nichiji", "race_num"],
    )


def etl_entries(engine: Engine, start: str, end: str) -> int:
    """n_uma_race → raw.entries (SQL JOIN with raw.races for FK integrity)

    Parameters
    ----------
    engine : SQLAlchemy Engine
    start : 開始日 (YYYYMMDD形式 int文字列)
    end : 終了日 (YYYYMMDD形式 int文字列)

    Returns
    -------
    int : 挿入件数
    """
    sql = text("""
        SELECT
            s.umaban,
            s.kettonum,
            s.kakuteijyuni,
            s.time,
            s.odds,
            s.ninki,
            s.bataijyu,
            s.zogenfugo,
            s.zogensa,
            s.kisyucode,
            s.chokyosicode,
            s.harontimel3,
            s.honsyokin,
            s.kyakusitukubun,
            r.race_id
        FROM n_uma_race s
        JOIN raw.races r
            ON s.year = r.year AND s.monthday = r.month_day
            AND s.jyocd = r.jyo_cd AND s.kaiji = r.kaiji
            AND s.nichiji = r.nichiji AND s.racenum = r.race_num
        WHERE (s.year || s.monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.info("etl_entries: 該当データなし")
        return 0

    # 型変換・カラムリネーム
    df["umaban"] = df["umaban"].apply(_to_int)
    df["ketto_num"] = df["kettonum"]
    df["finish_pos"] = df["kakuteijyuni"].apply(_to_int)
    df["finish_time"] = df["time"].apply(_to_float)
    df["win_odds"] = df["odds"].apply(_to_float)
    df["ninki"] = df["ninki"].apply(_to_int)
    df["ba_taijyu"] = df["bataijyu"].apply(_to_float)
    df["zogen_fugo"] = df["zogenfugo"].apply(_to_int)
    df["zogen_sa"] = df["zogensa"].apply(_to_float)
    df["kisyu_code"] = df["kisyucode"]
    df["chokyosi_code"] = df["chokyosicode"]
    df["haron_time_l3"] = df["harontimel3"].apply(_to_float)
    df["honsyokin"] = df["honsyokin"].apply(_to_int)
    df["kyakusitu"] = df["kyakusitukubun"].apply(_to_int)

    out = df[
        [
            "race_id",
            "umaban",
            "ketto_num",
            "finish_pos",
            "finish_time",
            "haron_time_l3",
            "ninki",
            "win_odds",
            "ba_taijyu",
            "zogen_fugo",
            "zogen_sa",
            "kisyu_code",
            "chokyosi_code",
            "kyakusitu",
            "honsyokin",
        ]
    ]

    return _insert_on_conflict(engine, out, "entries", "raw", ["race_id", "umaban"])


def etl_payouts(engine: Engine, start: str, end: str) -> int:
    """n_harai → raw.payouts (SQL JOIN with raw.races)

    Parameters
    ----------
    engine : SQLAlchemy Engine
    start : 開始日 (YYYYMMDD形式 int文字列)
    end : 終了日 (YYYYMMDD形式 int文字列)

    Returns
    -------
    int : 挿入件数
    """
    sql = text("""
        SELECT
            s.paytansyoumaban1,
            s.paytansyopay1,
            s.payfukusyoumaban1,
            s.payfukusyopay1,
            s.payfukusyoumaban2,
            s.payfukusyopay2,
            s.payfukusyoumaban3,
            s.payfukusyopay3,
            s.payfukusyoumaban4,
            s.payfukusyopay4,
            s.payfukusyoumaban5,
            s.payfukusyopay5,
            r.race_id
        FROM n_harai s
        JOIN raw.races r
            ON s.year = r.year AND s.monthday = r.month_day
            AND s.jyocd = r.jyo_cd AND s.kaiji = r.kaiji
            AND s.nichiji = r.nichiji AND s.racenum = r.race_num
        WHERE (s.year || s.monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.info("etl_payouts: 該当データなし")
        return 0

    # カラムリネーム・型変換
    df["tan_umaban"] = df["paytansyoumaban1"].apply(_to_int)
    df["tan_pay"] = df["paytansyopay1"].apply(_to_float)
    for i in range(1, 6):
        df[f"fuku_umaban{i}"] = df[f"payfukusyoumaban{i}"].apply(_to_int)
        df[f"fuku_pay{i}"] = df[f"payfukusyopay{i}"].apply(_to_float)

    out = df[
        [
            "race_id",
            "tan_umaban",
            "tan_pay",
            "fuku_umaban1",
            "fuku_pay1",
            "fuku_umaban2",
            "fuku_pay2",
            "fuku_umaban3",
            "fuku_pay3",
            "fuku_umaban4",
            "fuku_pay4",
            "fuku_umaban5",
            "fuku_pay5",
        ]
    ]

    return _insert_on_conflict(engine, out, "payouts", "raw", ["race_id"])


def etl_odds_snapshots(engine: Engine, start: str, end: str) -> int:
    """n_odds_tanpuku → odds_history.odds_snapshots (SQL JOIN with raw.races)

    Parameters
    ----------
    engine : SQLAlchemy Engine
    start : 開始日 (YYYYMMDD形式 int文字列)
    end : 終了日 (YYYYMMDD形式 int文字列)

    Returns
    -------
    int : 挿入件数
    """
    sql = text("""
        SELECT
            s.umaban,
            s.tanodds,
            s.fukuoddslow,
            r.race_id
        FROM n_odds_tanpuku s
        JOIN raw.races r
            ON s.year = r.year AND s.monthday = r.month_day
            AND s.jyocd = r.jyo_cd AND s.kaiji = r.kaiji
            AND s.nichiji = r.nichiji AND s.racenum = r.race_num
        WHERE (s.year || s.monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.info("etl_odds_snapshots: 該当データなし")
        return 0

    # 型変換・リネーム
    df["umaban"] = df["umaban"].apply(_to_int)
    df["tan_odds"] = df["tanodds"].apply(_to_float)
    df["fuku_odds"] = df["fukuoddslow"].apply(_to_float)

    out = df[["race_id", "umaban", "tan_odds", "fuku_odds"]]

    return _insert_on_conflict(
        engine, out, "odds_snapshots", "odds_history", ["race_id", "umaban"]
    )


def etl_wide_odds(engine: Engine, start: str, end: str) -> int:
    """n_odds_wide → odds_history.wide_odds (SQL JOIN with raw.races)

    Parameters
    ----------
    engine : SQLAlchemy Engine
    start : 開始日 (YYYYMMDD形式 int文字列)
    end : 終了日 (YYYYMMDD形式 int文字列)

    Returns
    -------
    int : 挿入件数
    """
    sql = text("""
        SELECT
            s.kumi,
            s.oddslow,
            s.oddshigh,
            r.race_id
        FROM n_odds_wide s
        JOIN raw.races r
            ON s.year = r.year AND s.monthday = r.month_day
            AND s.jyocd = r.jyo_cd AND s.kaiji = r.kaiji
            AND s.nichiji = r.nichiji AND s.racenum = r.race_num
        WHERE (s.year || s.monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.info("etl_wide_odds: 該当データなし")
        return 0

    # 型変換・リネーム
    df["odds_low"] = df["oddslow"].apply(_to_float)
    df["odds_high"] = df["oddshigh"].apply(_to_float)

    out = df[["race_id", "kumi", "odds_low", "odds_high"]]

    return _insert_on_conflict(
        engine, out, "wide_odds", "odds_history", ["race_id", "kumi"]
    )


def etl_odds_timeseries(engine: Engine, start: str, end: str) -> int:
    """n_jodds_tanpuku → odds_history.odds_time_series

    83M行の大規模テーブルのため、yearごとに分割ロード。
    SQL JOIN with raw.races for FK integrity.

    Parameters
    ----------
    engine : SQLAlchemy Engine
    start : 開始日 (YYYYMMDD形式 int文字列)
    end : 終了日 (YYYYMMDD形式 int文字列)

    Returns
    -------
    int : 総挿入件数
    """
    start_int = int(start)
    end_int = int(end)
    start_year = start_int // 10000
    end_year = end_int // 10000

    total_inserted = 0

    for year in range(start_year, end_year + 1):
        sql = text("""
            SELECT
                s.happyo_time,
                s.umaban,
                s.tanodds,
                s.fukuoddslow,
                s.tanninki,
                r.race_id
            FROM n_jodds_tanpuku s
            JOIN raw.races r
                ON s.year = r.year AND s.monthday = r.month_day
                AND s.jyocd = r.jyo_cd AND s.kaiji = r.kaiji
                AND s.nichiji = r.nichiji AND s.racenum = r.race_num
            WHERE s.year = :year
              AND (s.year || s.monthday)::int BETWEEN :start AND :end
        """)
        df = pd.read_sql(
            sql, engine, params={"year": year, "start": start, "end": end}
        )

        if df.empty:
            logger.info("etl_odds_timeseries: year=%d 該当データなし", year)
            continue

        # 型変換・リネーム
        df["umaban"] = df["umaban"].apply(_to_int)
        df["tan_odds"] = df["tanodds"].apply(_to_float)
        df["fuku_odds"] = df["fukuoddslow"].apply(_to_float)
        df["ninki"] = df["tanninki"].apply(_to_int)

        out = df[["race_id", "happyo_time", "umaban", "tan_odds", "fuku_odds", "ninki"]]

        inserted = _insert_on_conflict(
            engine,
            out,
            "odds_time_series",
            "odds_history",
            ["race_id", "happyo_time", "umaban"],
        )
        total_inserted += inserted

    return total_inserted


def run_full_etl(engine: Engine, start: str, end: str) -> dict[str, int]:
    """全ETL関数を順次実行し、各テーブルの挿入件数を返す

    Parameters
    ----------
    engine : SQLAlchemy Engine
    start : 開始日 (YYYYMMDD形式 int文字列)
    end : 終了日 (YYYYMMDD形式 int文字列)

    Returns
    -------
    dict[str, int] : テーブル名 → 挿入件数
    """
    create_project_schemas(engine)

    counts: dict[str, int] = {}
    counts["raw.races"] = etl_races(engine, start, end)
    counts["raw.entries"] = etl_entries(engine, start, end)
    counts["raw.payouts"] = etl_payouts(engine, start, end)
    counts["odds_history.odds_snapshots"] = etl_odds_snapshots(engine, start, end)
    counts["odds_history.wide_odds"] = etl_wide_odds(engine, start, end)
    counts["odds_history.odds_time_series"] = etl_odds_timeseries(engine, start, end)

    logger.info("ETL完了: %s", counts)
    return counts
