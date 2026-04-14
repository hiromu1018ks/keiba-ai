"""EveryDB2 速報系テーブルへの直接クエリラッパー

実装開始前に EveryDB2 インスタンスで s_ プレフィックスのテーブル名と列構造を確認すること。
テーブル名はハードコードせず、将来の変更に対応できるようにする。
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import psycopg2

logger = logging.getLogger(__name__)


class EveryDB2Queries:
    """EveryDB2 PostgreSQL テーブルへのクエリラッパー

    速報系 (s_) テーブルおよび蓄積系 (n_) テーブルに対する読み取り専用クエリを提供する。
    """

    def __init__(self, connection_string: str, timeout_seconds: int = 30) -> None:
        self._connection_string = connection_string
        self._timeout = timeout_seconds

    def _connect(self) -> Any:
        """PostgreSQL に接続 (UTF8クライアントエンコーディング)"""
        conn = psycopg2.connect(self._connection_string, connect_timeout=self._timeout)
        conn.set_client_encoding("UTF8")
        return conn

    def _query(self, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
        """SQL を実行して DataFrame を返す"""
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
            return df

    def get_race_schedule(self, target_date: date) -> list[dict[str, Any]]:
        """当日のレーススケジュールを取得。

        蓄積系テーブル n_uma_race を使用（木曜以降に利用可能）。
        非開催日は空リストを返す。

        注意: 実際のテーブル名・列名は EveryDB2 インスタンスで確認が必要。
        """
        # TODO: 実際のテーブル名・列名を EveryDB2 で確認後に修正
        ymd = target_date.strftime("%Y%m%d")
        sql = """
            SELECT
                CAST(Year || MonthDay || JyoCD || Kaiji || Nichiji || RaceNum
                     AS VARCHAR) as race_id,
                'venue_name' as venue,
                RaceNum as race_num,
                'HH:MM' as post_time,
                CASE WHEN TrackCD BETWEEN 10 AND 22 THEN 'turf' ELSE 'dirt' END as surface,
                Distance as distance
            FROM n_uma_race
            WHERE Year || MonthDay = %s
              AND TrackCD < 51
            ORDER BY JyoCD, RaceNum
        """
        try:
            df = self._query(sql, (ymd,))
        except Exception:
            logger.exception("Failed to get race schedule for %s", target_date)
            return []

        if df.empty:
            return []

        records: list[dict[str, Any]] = df.to_dict("records")
        return records

    def get_horse_weights(self, race_id: str) -> pd.DataFrame | None:
        """速報馬体重を取得。

        発走約1時間前に EveryDB2 自動更新で反映される。
        テーブル名は要確認 (s_bataijyu 推定)。
        """
        # TODO: 実際のテーブル名を確認
        year = race_id[:4]
        month_day = race_id[4:8]
        sql = """
            SELECT umaban, bataijyu as weight
            FROM s_bataijyu
            WHERE Year || MonthDay = %s
        """
        try:
            df = self._query(sql, (year + month_day,))
            return df if not df.empty else None
        except Exception:
            logger.exception("Failed to get horse weights for %s", race_id)
            return None

    def get_latest_odds(self, race_id: str) -> pd.DataFrame | None:
        """最新の速報オッズを取得。

        テーブル名は要確認 (s_odds_tanpuku 推定)。
        """
        # TODO: 実際のテーブル名を確認
        year = race_id[:4]
        month_day = race_id[4:8]
        sql = """
            SELECT umaban, tan_odds, fuku_odds
            FROM s_odds_tanpuku
            WHERE Year || MonthDay = %s
        """
        try:
            df = self._query(sql, (year + month_day,))
            return df if not df.empty else None
        except Exception:
            logger.exception("Failed to get odds for %s", race_id)
            return None

    def get_race_results(self, target_date: date) -> pd.DataFrame:
        """レース結果・払戻を取得。

        蓄積系テーブル n_uma_race + n_harai を使用。
        reconcile は 18:30 実行のため確定データが利用可能。
        """
        # TODO: 実際のテーブル名・列名を確認
        ymd = target_date.strftime("%Y%m%d")
        sql = """
            SELECT
                CAST(Year || MonthDay || JyoCD || Kaiji || Nichiji || RaceNum
                     AS VARCHAR) as race_id,
                Umaban as umaban,
                KakuteiJyunni as finish_pos,
                0.0 as place_pay,
                0.0 as place_odds,
                '' as horse_name
            FROM n_uma_race
            WHERE Year || MonthDay = %s
              AND TrackCD < 51
        """
        try:
            return self._query(sql, (ymd,))
        except Exception:
            logger.exception("Failed to get race results for %s", target_date)
            return pd.DataFrame()

    def get_track_condition(self, race_id: str) -> str | None:
        """天候馬場状態を取得。"""
        # TODO: 実際のテーブル名を確認
        year = race_id[:4]
        month_day = race_id[4:8]
        sql = """
            SELECT BabaCD as baba_cd, TenkoCD as tenko_cd
            FROM n_race
            WHERE Year || MonthDay = %s
        """
        try:
            df = self._query(sql, (year + month_day,))
            if df.empty:
                return None
            return str(df.iloc[0].get("baba_cd", ""))
        except Exception:
            logger.exception("Failed to get track condition for %s", race_id)
            return None

    def get_races(self, date_str: str) -> pd.DataFrame:
        """当日のレース情報を取得。n_race → s_race フォールバック。

        戻り値は EveryDB2 生データ (全列 character varying)。型変換は呼び出し側で行う。

        優先順位:
          1. n_race (蓄積テーブル): 全レースのメタデータを保持、TCC含む、PIT安全
          2. s_race DK=2 (速報テーブル): 当日データでn_raceに未反映の場合のフォールバック

        レースメタデータ (trackcd, kyori, TCC等) は全DataKubun値で共通のため、
        どのテーブルから取得してもPIT安全（ルックアヘッドなし）。
        """
        sql = "SELECT * FROM n_race WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            if not df.empty:
                return df
        except Exception:
            logger.exception("Failed to query n_race for %s", date_str)

        # n_race が空の場合（当日データ未反映等）は s_race DK=2 にフォールバック
        sql = "SELECT * FROM s_race WHERE year || monthday = %s AND DataKubun = '2'"
        try:
            df = self._query(sql, (date_str,))
            return df
        except Exception:
            logger.exception("Failed to query s_race for %s", date_str)
            return pd.DataFrame()

    def get_entries(self, date_str: str) -> pd.DataFrame:
        """当日の出走馬を取得。n_uma_race → s_uma_race フォールバック。

        戻り値は EveryDB2 生データ (全列 character varying)。型変換は呼び出し側で行う。

        優先順位:
          1. n_uma_race (蓄積テーブル): 全出走馬データを保持、PIT安全
          2. s_uma_race DK=2 (速報テーブル): 当日データのフォールバック

        出走データ (馬番, 騎手, 重量等) はDataKubun値に関わらず共通。
        n_uma_race は確定成績 (kakuteijyuni等) を含むが、
        特徴量エンジニアリングでは着順列は使用しないためPIT安全。
        """
        sql = "SELECT * FROM n_uma_race WHERE year || monthday = %s"
        try:
            df = self._query(sql, (date_str,))
            if not df.empty:
                return df
        except Exception:
            logger.exception("Failed to query n_uma_race for %s", date_str)

        # n_uma_race が空の場合は s_uma_race DK=2 にフォールバック
        sql = "SELECT * FROM s_uma_race WHERE year || monthday = %s AND DataKubun = '2'"
        try:
            df = self._query(sql, (date_str,))
            return df
        except Exception:
            logger.exception("Failed to query s_uma_race for %s", date_str)
            return pd.DataFrame()

    def get_odds_snapshots(self, date_str: str) -> pd.DataFrame:
        """当日の単勝・複勝オッズスナップショットを取得。

        s_odds_tanpuku は初回発売時のまま更新されない場合があるため、
        s_jodds_tanpuku (時系列) から各馬の最新エントリを取得する。
        s_jodds_tanpuku → n_jodds_tanpuku フォールバック。
        戻り値は EveryDB2 生データ (全列 character varying)。型変換は呼び出し側で行う。
        """
        # 時系列テーブルから各馬の最新オッズを取得
        sql = """
            SELECT DISTINCT ON (year, monthday, jyocd, kaiji, nichiji, racenum, umaban)
                *
            FROM s_jodds_tanpuku
            WHERE year || monthday = %s
            ORDER BY year, monthday, jyocd, kaiji, nichiji, racenum, umaban, happyotime DESC
        """
        try:
            df = self._query(sql, (date_str,))
            if not df.empty:
                return df
        except Exception:
            logger.exception("Failed to query s_jodds_tanpuku for %s", date_str)

        sql = """
            SELECT DISTINCT ON (year, monthday, jyocd, kaiji, nichiji, racenum, umaban)
                *
            FROM n_jodds_tanpuku
            WHERE year || monthday = %s
            ORDER BY year, monthday, jyocd, kaiji, nichiji, racenum, umaban, happyotime DESC
        """
        try:
            df = self._query(sql, (date_str,))
            return df
        except Exception:
            logger.exception("Failed to query n_jodds_tanpuku for %s", date_str)
            return pd.DataFrame()

    def get_payouts(self, date_str: str) -> pd.DataFrame:
        """払戻データを取得。s_harai → n_harai フォールバック。

        複勝払戻の馬番・払戻金を含む。払戻金は「100円あたりの円」(整数 varchar 9桁)。
        戻り値は EveryDB2 生データ (全列 character varying)。
        """
        pk = "year, monthday, jyocd, kaiji, nichiji, racenum"
        sql = f"""
            SELECT
                CAST(year || monthday || jyocd || kaiji || nichiji || racenum AS varchar) AS race_id,
                payfukusyoumaban1, payfukusyopay1,
                payfukusyoumaban2, payfukusyopay2,
                payfukusyoumaban3, payfukusyopay3,
                payfukusyoumaban4, payfukusyopay4,
                payfukusyoumaban5, payfukusyopay5
            FROM s_harai
            WHERE year || monthday = %s
              AND datakubun IN ('1', '2')
        """
        try:
            df = self._query(sql, (date_str,))
            if not df.empty:
                return df
        except Exception:
            logger.exception("Failed to query s_harai for %s", date_str)

        sql = f"""
            SELECT
                CAST(year || monthday || jyocd || kaiji || nichiji || racenum AS varchar) AS race_id,
                payfukusyoumaban1, payfukusyopay1,
                payfukusyoumaban2, payfukusyopay2,
                payfukusyoumaban3, payfukusyopay3,
                payfukusyoumaban4, payfukusyopay4,
                payfukusyoumaban5, payfukusyopay5
            FROM n_harai
            WHERE year || monthday = %s
              AND datakubun IN ('1', '2')
        """
        try:
            df = self._query(sql, (date_str,))
            return df
        except Exception:
            logger.exception("Failed to query n_harai for %s", date_str)
            return pd.DataFrame()

    def get_odds_time_series(self, date_str: str) -> pd.DataFrame:
        """当日の時系列オッズを取得。

        s_jodds_tanpuku → n_jodds_tanpuku フォールバック。
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
