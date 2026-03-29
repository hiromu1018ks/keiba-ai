"""EveryDB2 → プロジェクトスキーマ ETL モジュール

EveryDB2外部テーブル(n_race, n_uma_race, n_harai, n_odds_tanpuku,
n_odds_wide, n_jodds_tanpuku) からプロジェクトスキーマ(raw.*, odds_history.*)
へデータをコピーする。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tqdm.auto import tqdm

from db.schema import ALL_CREATE_STATEMENTS

if TYPE_CHECKING:
    from db.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _to_int(val: str | None) -> Optional[int]:
    """空文字・非数値 → None、それ以外は int に変換"""
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _to_float(val: str | None) -> Optional[float]:
    """空文字・非数値 → None、それ以外は float に変換"""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _to_odds(val: str | None, divisor: int = 10) -> Optional[float]:
    """EveryDB2 オッズ文字列 → float (÷ divisor)

    EveryDB2はオッズをゼロ埋め整数で保存:
    - tan/fuku: "0014" → 1.4 (÷10)
    - wide:     "03783" → 37.83 (÷100)
    """
    if val is None or val == "":
        return None
    try:
        return float(val) / divisor
    except (ValueError, TypeError):
        return None


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
    """ステージングテーブル + ON CONFLICT DO NOTHING で冪等挿入

    pandas to_sql で一時テーブルに書き込み後、
    ON CONFLICT DO NOTHING で本テーブルにコピーする。
    to_sql が型変換を適切に処理するため、varchar桁数エラーを回避。

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

    staging = f"_etl_staging_{table}"

    # ステージングテーブルに書き込み（to_sql が型変換を処理）
    df.to_sql(staging, engine, if_exists="replace", index=False)

    # ON CONFLICT DO NOTHING で本テーブルに挿入
    cols = ", ".join(f'"{c}"' for c in df.columns)
    pk_conflict = ", ".join(f'"{c}"' for c in pk_columns)

    insert_sql = f"""
        INSERT INTO {schema}.{table} ({cols})
        SELECT {cols} FROM {staging}
        ON CONFLICT ({pk_conflict}) DO NOTHING
    """
    try:
        with engine.begin() as conn:
            result = conn.execute(text(insert_sql))
            inserted = result.rowcount
    except Exception as e:
        # ステージングテーブルを削除（エラー時も掃除）
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))
        logger.error(
            "%s.%s 挿入失敗: %s — カラム型の不整合可能性",
            schema,
            table,
            e,
        )
        raise

    # ステージングテーブルを削除
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {staging}"))

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
          AND trackcd::int NOT BETWEEN 51 AND 59
          AND jyocd BETWEEN '01' AND '10'
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
        lambda row: _select_baba_cd(row["track_cd"], row["sibababacd"], row["dirtbabacd"]),
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
            s.timedifn,
            s.jyuni1c,
            s.jyuni4c,
            s.honsyokin,
            s.kyakusitukubun,
            r.race_id
        FROM n_uma_race s
        JOIN raw.races r
            ON s.year::int = r.year AND s.monthday = r.month_day
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
    df["win_odds"] = df["odds"].apply(_to_odds)
    df["ninki"] = df["ninki"].apply(_to_int)
    df["ba_taijyu"] = df["bataijyu"].apply(_to_float)
    df["zogen_fugo"] = df["zogenfugo"]
    df["zogen_sa"] = df["zogensa"].apply(_to_float)
    df["kisyu_code"] = df["kisyucode"]
    df["chokyosi_code"] = df["chokyosicode"]
    df["haron_time_l3"] = df["harontimel3"].apply(_to_float)
    df["time_diff"] = df["timedifn"].apply(_to_float)
    df["corner_1c"] = df["jyuni1c"].apply(_to_int)
    df["corner_4c"] = df["jyuni4c"].apply(_to_int)
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
            "time_diff",
            "corner_1c",
            "corner_4c",
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
            ON s.year::int = r.year AND s.monthday = r.month_day
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
            ON s.year::int = r.year AND s.monthday = r.month_day
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
    df["tan_odds"] = df["tanodds"].apply(_to_odds)
    df["fuku_odds"] = df["fukuoddslow"].apply(_to_odds)

    out = df[["race_id", "umaban", "tan_odds", "fuku_odds"]]

    return _insert_on_conflict(engine, out, "odds_snapshots", "odds_history", ["race_id", "umaban"])


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
            ON s.year::int = r.year AND s.monthday = r.month_day
            AND s.jyocd = r.jyo_cd AND s.kaiji = r.kaiji
            AND s.nichiji = r.nichiji AND s.racenum = r.race_num
        WHERE (s.year || s.monthday)::int BETWEEN :start AND :end
    """)
    df = pd.read_sql(sql, engine, params={"start": start, "end": end})

    if df.empty:
        logger.info("etl_wide_odds: 該当データなし")
        return 0

    # 型変換・リネーム
    df["odds_low"] = df["oddslow"].apply(lambda v: _to_odds(v, divisor=100))
    df["odds_high"] = df["oddshigh"].apply(lambda v: _to_odds(v, divisor=100))

    out = df[["race_id", "kumi", "odds_low", "odds_high"]]

    return _insert_on_conflict(engine, out, "wide_odds", "odds_history", ["race_id", "kumi"])


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

    for year in tqdm(range(start_year, end_year + 1), desc="odds_timeseries (年)"):
        sql = text("""
            SELECT
                s.happyotime,
                s.umaban,
                s.tanodds,
                s.fukuoddslow,
                s.tanninki,
                r.race_id
            FROM n_jodds_tanpuku s
            JOIN raw.races r
                ON s.year::int = r.year AND s.monthday = r.month_day
                AND s.jyocd = r.jyo_cd AND s.kaiji = r.kaiji
                AND s.nichiji = r.nichiji AND s.racenum = r.race_num
            WHERE s.year::int = :year
              AND (s.year || s.monthday)::int BETWEEN :start AND :end
        """)
        df = pd.read_sql(sql, engine, params={"year": year, "start": start, "end": end})

        if df.empty:
            logger.info("etl_odds_timeseries: year=%d 該当データなし", year)
            continue

        # 型変換・リネーム
        df["happyo_time"] = df["happyotime"]
        df["umaban"] = df["umaban"].apply(_to_int)
        df["tan_odds"] = df["tanodds"].apply(_to_odds)
        df["fuku_odds"] = df["fukuoddslow"].apply(_to_odds)
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

    etl_steps = [
        ("raw.races", etl_races),
        ("raw.entries", etl_entries),
        ("raw.payouts", etl_payouts),
        ("odds_history.odds_snapshots", etl_odds_snapshots),
        ("odds_history.wide_odds", etl_wide_odds),
        ("odds_history.odds_time_series", etl_odds_timeseries),
    ]

    counts: dict[str, int] = {}
    for name, func in tqdm(etl_steps, desc="ETL テーブル"):
        counts[name] = func(engine, start, end)

    logger.info("ETL完了: %s", counts)
    return counts


# ---------------------------------------------------------------------------
# Parquet ETL (EveryDB2 → Parquet via ParquetStore)
# ---------------------------------------------------------------------------


def _etl_horses_to_parquet(engine: Engine, store: "ParquetStore") -> int:
    """x_UMA → data/raw/horses.parquet — 馬マスターデータ (血統・産駒成績)"""
    sql = text("""
        SELECT
            kettonum,
            ketto3infohansyokunum1, ketto3infohansyokunum2, ketto3infohansyokunum3, ketto3infohansyokunum4,
            ketto3infohansyokunum5, ketto3infohansyokunum6, ketto3infohansyokunum7, ketto3infohansyokunum8,
            ketto3infohansyokunum9, ketto3infohansyokunum10, ketto3infohansyokunum11, ketto3infohansyokunum12,
            ketto3infohansyokunum13, ketto3infohansyokunum14,
            ba1chakukaisu1, ba1chakukaisu2, ba1chakukaisu3, ba1chakukaisu4, ba1chakukaisu5, ba1chakukaisu6,
            ba2chakukaisu1, ba2chakukaisu2, ba2chakukaisu3, ba2chakukaisu4, ba2chakukaisu5, ba2chakukaisu6,
            ba3chakukaisu1, ba3chakukaisu2, ba3chakukaisu3, ba3chakukaisu4, ba3chakukaisu5, ba3chakukaisu6,
            ba4chakukaisu1, ba4chakukaisu2, ba4chakukaisu3, ba4chakukaisu4, ba4chakukaisu5, ba4chakukaisu6,
            ba5chakukaisu1, ba5chakukaisu2, ba5chakukaisu3, ba5chakukaisu4, ba5chakukaisu5, ba5chakukaisu6,
            ba6chakukaisu1, ba6chakukaisu2, ba6chakukaisu3, ba6chakukaisu4, ba6chakukaisu5, ba6chakukaisu6,
            kyori1chakukaisu1, kyori1chakukaisu2, kyori1chakukaisu3, kyori1chakukaisu4,
            kyori1chakukaisu5, kyori1chakukaisu6,
            kyori2chakukaisu1, kyori2chakukaisu2, kyori2chakukaisu3, kyori2chakukaisu4,
            kyori2chakukaisu5, kyori2chakukaisu6,
            kyori3chakukaisu1, kyori3chakukaisu2, kyori3chakukaisu3, kyori3chakukaisu4,
            kyori3chakukaisu5, kyori3chakukaisu6,
            kyori4chakukaisu1, kyori4chakukaisu2, kyori4chakukaisu3, kyori4chakukaisu4,
            kyori4chakukaisu5, kyori4chakukaisu6,
            kyori5chakukaisu1, kyori5chakukaisu2, kyori5chakukaisu3, kyori5chakukaisu4,
            kyori5chakukaisu5, kyori5chakukaisu6,
            kyori6chakukaisu1, kyori6chakukaisu2, kyori6chakukaisu3, kyori6chakukaisu4,
            kyori6chakukaisu5, kyori6chakukaisu6,
            chuochakukaisu1, chuochakukaisu2, chuochakukaisu3, chuochakukaisu4, chuochakukaisu5, chuochakukaisu6,
            ruikeihonsyoheichi,
            kyakusitu1, kyakusitu2, kyakusitu3, kyakusitu4
        FROM x_uma
        WHERE kettonum IS NOT NULL
    """)
    df = pd.read_sql(sql, engine)

    if df.empty:
        return 0

    # Convert numeric columns: all chakukaisu columns to int, ruikeihonsyoheichi to float
    chakukaisu_cols = [c for c in df.columns if "chakukaisu" in c]
    for col in chakukaisu_cols:
        df[col] = df[col].apply(_to_int)

    df["ruikeihonsyoheichi"] = df["ruikeihonsyoheichi"].apply(_to_float)

    # Convert kyakusitu columns to int
    for col in ["kyakusitu1", "kyakusitu2", "kyakusitu3", "kyakusitu4"]:
        if col in df.columns:
            df[col] = df[col].apply(_to_int)

    # Convert bloodline columns to string (ketto3infohansyokunum*)
    blood_cols = [c for c in df.columns if c.startswith("ketto3infohansyokunum")]
    for col in blood_cols:
        df[col] = df[col].astype(str)

    store.write("raw", "horses", df)
    return len(df)


def _etl_jockey_stats_to_parquet(engine: Engine, store: "ParquetStore") -> int:
    """x_KISYU_SEISEKI → data/raw/jockey_stats.parquet — 騎手年度別成績"""
    sql = text("""
        SELECT
            setyear, kisyucode,
            heichichakukaisu1, heichichakukaisu2, heichichakukaisu3,
            heichichakukaisu4, heichichakukaisu5, heichichakukaisu6,
            jyo1chakukaisu1, jyo1chakukaisu2, jyo1chakukaisu3,
            jyo1chakukaisu4, jyo1chakukaisu5, jyo1chakukaisu6,
            jyo2chakukaisu1, jyo2chakukaisu2, jyo2chakukaisu3,
            jyo2chakukaisu4, jyo2chakukaisu5, jyo2chakukaisu6,
            jyo3chakukaisu1, jyo3chakukaisu2, jyo3chakukaisu3,
            jyo3chakukaisu4, jyo3chakukaisu5, jyo3chakukaisu6,
            jyo4chakukaisu1, jyo4chakukaisu2, jyo4chakukaisu3,
            jyo4chakukaisu4, jyo4chakukaisu5, jyo4chakukaisu6,
            jyo5chakukaisu1, jyo5chakukaisu2, jyo5chakukaisu3,
            jyo5chakukaisu4, jyo5chakukaisu5, jyo5chakukaisu6,
            kyori1chakukaisu1, kyori1chakukaisu2, kyori1chakukaisu3,
            kyori1chakukaisu4, kyori1chakukaisu5, kyori1chakukaisu6,
            kyori2chakukaisu1, kyori2chakukaisu2, kyori2chakukaisu3,
            kyori2chakukaisu4, kyori2chakukaisu5, kyori2chakukaisu6,
            kyori3chakukaisu1, kyori3chakukaisu2, kyori3chakukaisu3,
            kyori3chakukaisu4, kyori3chakukaisu5, kyori3chakukaisu6,
            kyori4chakukaisu1, kyori4chakukaisu2, kyori4chakukaisu3,
            kyori4chakukaisu4, kyori4chakukaisu5, kyori4chakukaisu6,
            kyori5chakukaisu1, kyori5chakukaisu2, kyori5chakukaisu3,
            kyori5chakukaisu4, kyori5chakukaisu5, kyori5chakukaisu6,
            kyori6chakukaisu1, kyori6chakukaisu2, kyori6chakukaisu3,
            kyori6chakukaisu4, kyori6chakukaisu5, kyori6chakukaisu6,
            honsyokinheichi
        FROM x_kisyu_seiseki
        WHERE setyear IS NOT NULL
    """)
    df = pd.read_sql(sql, engine)

    if df.empty:
        return 0

    # Type conversions
    chakukaisu_cols = [c for c in df.columns if "chakukaisu" in c]
    for col in chakukaisu_cols:
        df[col] = df[col].apply(_to_int)
    df["setyear"] = df["setyear"].apply(_to_int)
    df["honsyokinheichi"] = df["honsyokinheichi"].apply(_to_float)

    store.write("raw", "jockey_stats", df)
    return len(df)


def _etl_trainer_stats_to_parquet(engine: Engine, store: "ParquetStore") -> int:
    """x_CHOKYO_SEISEKI → data/raw/trainer_stats.parquet — 調教師年度別成績"""
    sql = text("""
        SELECT
            setyear, chokyosicode,
            heichichakukaisu1, heichichakukaisu2, heichichakukaisu3,
            heichichakukaisu4, heichichakukaisu5, heichichakukaisu6,
            jyo1chakukaisu1, jyo1chakukaisu2, jyo1chakukaisu3,
            jyo1chakukaisu4, jyo1chakukaisu5, jyo1chakukaisu6,
            jyo2chakukaisu1, jyo2chakukaisu2, jyo2chakukaisu3,
            jyo2chakukaisu4, jyo2chakukaisu5, jyo2chakukaisu6,
            jyo3chakukaisu1, jyo3chakukaisu2, jyo3chakukaisu3,
            jyo3chakukaisu4, jyo3chakukaisu5, jyo3chakukaisu6,
            jyo4chakukaisu1, jyo4chakukaisu2, jyo4chakukaisu3,
            jyo4chakukaisu4, jyo4chakukaisu5, jyo4chakukaisu6,
            jyo5chakukaisu1, jyo5chakukaisu2, jyo5chakukaisu3,
            jyo5chakukaisu4, jyo5chakukaisu5, jyo5chakukaisu6,
            kyori1chakukaisu1, kyori1chakukaisu2, kyori1chakukaisu3,
            kyori1chakukaisu4, kyori1chakukaisu5, kyori1chakukaisu6,
            kyori2chakukaisu1, kyori2chakukaisu2, kyori2chakukaisu3,
            kyori2chakukaisu4, kyori2chakukaisu5, kyori2chakukaisu6,
            kyori3chakukaisu1, kyori3chakukaisu2, kyori3chakukaisu3,
            kyori3chakukaisu4, kyori3chakukaisu5, kyori3chakukaisu6,
            kyori4chakukaisu1, kyori4chakukaisu2, kyori4chakukaisu3,
            kyori4chakukaisu4, kyori4chakukaisu5, kyori4chakukaisu6,
            kyori5chakukaisu1, kyori5chakukaisu2, kyori5chakukaisu3,
            kyori5chakukaisu4, kyori5chakukaisu5, kyori5chakukaisu6,
            kyori6chakukaisu1, kyori6chakukaisu2, kyori6chakukaisu3,
            kyori6chakukaisu4, kyori6chakukaisu5, kyori6chakukaisu6,
            honsyokinheichi
        FROM x_chokyo_seiseki
        WHERE setyear IS NOT NULL
    """)
    df = pd.read_sql(sql, engine)

    if df.empty:
        return 0

    # Type conversions
    chakukaisu_cols = [c for c in df.columns if "chakukaisu" in c]
    for col in chakukaisu_cols:
        df[col] = df[col].apply(_to_int)
    df["setyear"] = df["setyear"].apply(_to_int)
    df["honsyokinheichi"] = df["honsyokinheichi"].apply(_to_float)

    store.write("raw", "trainer_stats", df)
    return len(df)


def run_full_etl_to_parquet(
    engine: Engine, store: "ParquetStore", start: str, end: str
) -> dict[str, int]:
    """EveryDB2 → Parquet ETL。

    既存のSQL読み取り（EveryDB2外部テーブル）はそのまま使い、
    書き込み先をPostgreSQL内部スキーマ → Parquetに変更。
    """
    from db.connection import _compute_race_date, _compute_race_id

    counts: dict[str, int] = {}

    # 1. races — EveryDB2 n_race から直接読み取り（JOIN不要）
    races_sql = text("""
        SELECT
            year, monthday, jyocd, kaiji, nichiji, racenum,
            trackcd, kyori, tenkocd, sibababacd, dirtbabacd,
            syubetucd, jyokencd1, gradecd, syussotosu
        FROM n_race
        WHERE (year || monthday)::int BETWEEN :start AND :end
          AND trackcd::int NOT BETWEEN 51 AND 59
          AND jyocd BETWEEN '01' AND '10'
    """)
    races_df = pd.read_sql(races_sql, engine, params={"start": int(start), "end": int(end)})

    if not races_df.empty:
        # Apply same transformations as etl_races
        races_df["year"] = races_df["year"].apply(_to_int)
        races_df["month_day"] = races_df["monthday"]
        races_df["jyo_cd"] = races_df["jyocd"]
        races_df["race_num"] = races_df["racenum"]
        races_df["track_cd"] = races_df["trackcd"].apply(_to_int)
        races_df["distance"] = races_df["kyori"].apply(_to_int)
        races_df["tenko_cd"] = races_df["tenkocd"].apply(_to_int)
        races_df["baba_cd"] = races_df.apply(
            lambda row: _select_baba_cd(row["track_cd"], row["sibababacd"], row["dirtbabacd"]),
            axis=1,
        )
        races_df["syubetu_cd"] = races_df["syubetucd"]
        races_df["jyoken_cd"] = races_df["jyokencd1"]
        races_df["grade_cd"] = races_df["gradecd"].apply(lambda x: x if x and x != "" else "_")
        races_df["field_size"] = races_df["syussotosu"].apply(_to_int)

        races_out = races_df[
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
        _compute_race_id(races_out)
        _compute_race_date(races_out)
        store.write("raw", "races", races_out)
        counts["races"] = len(races_out)
    else:
        counts["races"] = 0

    # Build race key map for pandas merge
    # (entries, payouts, etc. need race_id and race_date)
    race_key_cols = ["year", "month_day", "jyo_cd", "kaiji", "nichiji", "race_num"]
    if not races_df.empty:
        race_keys = races_out[race_key_cols + ["race_id", "race_date"]].copy()
    else:
        race_keys = pd.DataFrame(columns=race_key_cols + ["race_id", "race_date"])

    # 2. entries — n_uma_race (no JOIN with raw.races)
    entries_sql = text("""
        SELECT
            year, monthday, jyocd, kaiji, nichiji, racenum,
            umaban, kettonum, kakuteijyuni, time, odds, ninki,
            bataijyu, zogenfugo, zogensa, kisyucode, chokyosicode,
            harontimel3, timedifn, jyuni1c, jyuni4c, honsyokin, kyakusitukubun
        FROM n_uma_race
        WHERE (year || monthday)::int BETWEEN :start AND :end
    """)
    entries_df = pd.read_sql(entries_sql, engine, params={"start": int(start), "end": int(end)})

    if not entries_df.empty:
        entries_df["jyo_cd"] = entries_df["jyocd"]
        entries_df["race_num"] = entries_df["racenum"]
        entries_df["month_day"] = entries_df["monthday"]
        entries_df["year"] = entries_df["year"].apply(_to_int)
        entries_df = entries_df.merge(race_keys, on=race_key_cols, how="inner")

        entries_df["umaban"] = entries_df["umaban"].apply(_to_int)
        entries_df["ketto_num"] = entries_df["kettonum"]
        entries_df["finish_pos"] = entries_df["kakuteijyuni"].apply(_to_int)
        entries_df["finish_time"] = entries_df["time"].apply(_to_float)
        entries_df["win_odds"] = entries_df["odds"].apply(_to_odds)
        entries_df["ninki"] = entries_df["ninki"].apply(_to_int)
        entries_df["ba_taijyu"] = entries_df["bataijyu"].apply(_to_float)
        entries_df["zogen_fugo"] = entries_df["zogenfugo"]
        entries_df["zogen_sa"] = entries_df["zogensa"].apply(_to_float)
        entries_df["kisyu_code"] = entries_df["kisyucode"]
        entries_df["chokyosi_code"] = entries_df["chokyosicode"]
        entries_df["haron_time_l3"] = entries_df["harontimel3"].apply(_to_float)
        entries_df["time_diff"] = entries_df["timedifn"].apply(_to_float)
        entries_df["corner_1c"] = entries_df["jyuni1c"].apply(_to_int)
        entries_df["corner_4c"] = entries_df["jyuni4c"].apply(_to_int)
        entries_df["honsyokin"] = entries_df["honsyokin"].apply(_to_int)
        entries_df["kyakusitu"] = entries_df["kyakusitukubun"].apply(_to_int)

        entries_out = entries_df[
            [
                "race_id",
                "umaban",
                "ketto_num",
                "finish_pos",
                "finish_time",
                "haron_time_l3",
                "time_diff",
                "corner_1c",
                "corner_4c",
                "ninki",
                "win_odds",
                "ba_taijyu",
                "zogen_fugo",
                "zogen_sa",
                "kisyu_code",
                "chokyosi_code",
                "kyakusitu",
                "honsyokin",
                "race_date",
            ]
        ]
        store.write("raw", "entries", entries_out)
        counts["entries"] = len(entries_out)
    else:
        counts["entries"] = 0

    # 3. payouts — n_harai (no JOIN with raw.races)
    payouts_sql = text("""
        SELECT
            year, monthday, jyocd, kaiji, nichiji, racenum,
            paytansyoumaban1, paytansyopay1,
            payfukusyoumaban1, payfukusyopay1,
            payfukusyoumaban2, payfukusyopay2,
            payfukusyoumaban3, payfukusyopay3,
            payfukusyoumaban4, payfukusyopay4,
            payfukusyoumaban5, payfukusyopay5
        FROM n_harai
        WHERE (year || monthday)::int BETWEEN :start AND :end
    """)
    payouts_df = pd.read_sql(payouts_sql, engine, params={"start": int(start), "end": int(end)})

    if not payouts_df.empty:
        payouts_df["jyo_cd"] = payouts_df["jyocd"]
        payouts_df["race_num"] = payouts_df["racenum"]
        payouts_df["month_day"] = payouts_df["monthday"]
        payouts_df["year"] = payouts_df["year"].apply(_to_int)
        payouts_df = payouts_df.merge(race_keys, on=race_key_cols, how="inner")

        payouts_df["tan_umaban"] = payouts_df["paytansyoumaban1"].apply(_to_int)
        payouts_df["tan_pay"] = payouts_df["paytansyopay1"].apply(_to_float)
        for i in range(1, 6):
            payouts_df[f"fuku_umaban{i}"] = payouts_df[f"payfukusyoumaban{i}"].apply(_to_int)
            payouts_df[f"fuku_pay{i}"] = payouts_df[f"payfukusyopay{i}"].apply(_to_float)

        payouts_out = payouts_df[
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
                "race_date",
            ]
        ]
        store.write("raw", "payouts", payouts_out)
        counts["payouts"] = len(payouts_out)
    else:
        counts["payouts"] = 0

    # 4. odds_snapshots — n_odds_tanpuku (no JOIN with raw.races)
    snapshots_sql = text("""
        SELECT
            year, monthday, jyocd, kaiji, nichiji, racenum,
            umaban, tanodds, fukuoddslow
        FROM n_odds_tanpuku
        WHERE (year || monthday)::int BETWEEN :start AND :end
    """)
    snapshots_df = pd.read_sql(snapshots_sql, engine, params={"start": int(start), "end": int(end)})

    if not snapshots_df.empty:
        snapshots_df["jyo_cd"] = snapshots_df["jyocd"]
        snapshots_df["race_num"] = snapshots_df["racenum"]
        snapshots_df["month_day"] = snapshots_df["monthday"]
        snapshots_df["year"] = snapshots_df["year"].apply(_to_int)
        snapshots_df = snapshots_df.merge(race_keys, on=race_key_cols, how="inner")

        snapshots_df["umaban"] = snapshots_df["umaban"].apply(_to_int)
        snapshots_df["tan_odds"] = snapshots_df["tanodds"].apply(_to_odds)
        snapshots_df["fuku_odds"] = snapshots_df["fukuoddslow"].apply(_to_odds)

        snapshots_out = snapshots_df[["race_id", "umaban", "tan_odds", "fuku_odds", "race_date"]]
        store.write("odds", "snapshots", snapshots_out)
        counts["odds_snapshots"] = len(snapshots_out)
    else:
        counts["odds_snapshots"] = 0

    # 5. wide_odds — n_odds_wide (no JOIN with raw.races)
    wide_sql = text("""
        SELECT
            year, monthday, jyocd, kaiji, nichiji, racenum,
            kumi, oddslow, oddshigh
        FROM n_odds_wide
        WHERE (year || monthday)::int BETWEEN :start AND :end
    """)
    wide_df = pd.read_sql(wide_sql, engine, params={"start": int(start), "end": int(end)})

    if not wide_df.empty:
        wide_df["jyo_cd"] = wide_df["jyocd"]
        wide_df["race_num"] = wide_df["racenum"]
        wide_df["month_day"] = wide_df["monthday"]
        wide_df["year"] = wide_df["year"].apply(_to_int)
        wide_df = wide_df.merge(race_keys, on=race_key_cols, how="inner")

        wide_df["odds_low"] = wide_df["oddslow"].apply(lambda v: _to_odds(v, divisor=100))
        wide_df["odds_high"] = wide_df["oddshigh"].apply(lambda v: _to_odds(v, divisor=100))

        wide_out = wide_df[["race_id", "kumi", "odds_low", "odds_high", "race_date"]]
        store.write("odds", "wide", wide_out)
        counts["wide_odds"] = len(wide_out)
    else:
        counts["wide_odds"] = 0

    # 6. odds_time_series — n_jodds_tanpuku (year-by-year, partitioned write)
    start_int = int(start)
    end_int = int(end)
    start_year = start_int // 10000
    end_year = end_int // 10000
    total_ts = 0

    ts_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        ts_sql = text("""
            SELECT
                year, monthday, jyocd, kaiji, nichiji, racenum,
                happyotime, umaban, tanodds, fukuoddslow, tanninki
            FROM n_jodds_tanpuku
            WHERE year::int = :year
              AND (year || monthday)::int BETWEEN :start AND :end
        """)
        ts_df = pd.read_sql(ts_sql, engine, params={"year": year, "start": start, "end": end})
        if not ts_df.empty:
            ts_df["jyo_cd"] = ts_df["jyocd"]
            ts_df["race_num"] = ts_df["racenum"]
            ts_df["month_day"] = ts_df["monthday"]
            ts_df["year_int"] = ts_df["year"].apply(_to_int)
            ts_df = ts_df.merge(
                race_keys,
                left_on=[
                    "year_int",
                    "month_day",
                    "jyo_cd",
                    "kaiji",
                    "nichiji",
                    "race_num",
                ],
                right_on=race_key_cols,
                how="inner",
            )
            ts_df["happyo_time"] = ts_df["happyotime"]
            ts_df["umaban"] = ts_df["umaban"].apply(_to_int)
            ts_df["tan_odds"] = ts_df["tanodds"].apply(_to_odds)
            ts_df["fuku_odds"] = ts_df["fukuoddslow"].apply(_to_odds)
            ts_df["ninki"] = ts_df["tanninki"].apply(_to_int)
            ts_frames.append(
                ts_df[
                    [
                        "race_id",
                        "happyo_time",
                        "umaban",
                        "tan_odds",
                        "fuku_odds",
                        "ninki",
                        "race_date",
                    ]
                ]
            )

    if ts_frames:
        ts_out = pd.concat(ts_frames, ignore_index=True)
        ts_out["year"] = ts_out["race_date"].dt.year
        ts_out["month"] = ts_out["race_date"].dt.month
        store.write("odds", "time_series", ts_out, partition_cols=["year", "month"])
        total_ts = len(ts_out)
    counts["odds_time_series"] = total_ts

    # 7. horses — x_uma (bloodline stats)
    counts["horses"] = _etl_horses_to_parquet(engine, store)

    # 8. jockey_stats — x_kisyu_seiseki
    counts["jockey_stats"] = _etl_jockey_stats_to_parquet(engine, store)

    # 9. trainer_stats — x_chokyo_seiseki
    counts["trainer_stats"] = _etl_trainer_stats_to_parquet(engine, store)

    logger.info("ETL to Parquet完了: %s", counts)
    return counts
