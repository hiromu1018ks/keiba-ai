"""MLパイプラインの唯一のデータアクセス窓口。

将来DuckDB/Polarsへの移行を妨げないよう、この層が唯一のアクセス経路。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from db.parquet_store import ParquetStore


def _to_dt(yyyymmdd: str) -> datetime:
    """'YYYYMMDD' 文字列 → datetime"""
    return datetime.strptime(yyyymmdd, "%Y%m%d")


def _date_filters(start: str, end: str) -> list[tuple]:
    """pyarrow述語プッシュダウン用フィルタを生成。"""
    s, e = _to_dt(start), _to_dt(end)
    return [("race_date", ">=", s), ("race_date", "<=", e)]


def _exclude_steeple(df: pd.DataFrame) -> pd.DataFrame:
    """障害レース除外（track_cd 51-59）。track_cd列がなければそのまま返す。"""
    if "track_cd" not in df.columns:
        return df
    return df[~df["track_cd"].between(51, 59)].copy()


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
    """EveryDB2生カラム名をMLパイプライン互換に変換 (リネーム+型変換)。"""
    rename_map = {
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


def _transform_payouts_columns(df: pd.DataFrame) -> pd.DataFrame:
    """n_harai の生カラム名をML既存名に変換 (payoutsテーブル用)"""
    rename_map = {
        "paytansyoumaban1": "tan_umaban",
        "paytansyopay1": "tan_pay",
    }
    for i in range(1, 6):
        rename_map[f"payfukusyoumaban{i}"] = f"fuku_umaban{i}"
        rename_map[f"payfukusyopay{i}"] = f"fuku_pay{i}"
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)
    if "tan_umaban" in df.columns:
        df["tan_umaban"] = df["tan_umaban"].apply(_to_int)
    if "tan_pay" in df.columns:
        df["tan_pay"] = df["tan_pay"].apply(_to_float)
    for i in range(1, 6):
        for prefix in ("fuku_umaban", "fuku_pay"):
            col = f"{prefix}{i}"
            if col in df.columns:
                df[col] = df[col].apply(_to_int if "umaban" in prefix else _to_float)
    return df


def _transform_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """n_odds_tanpuku/n_odds_wide の生カラム名をML既存名に変換"""
    rename_map = {
        "tanodds": "tan_odds",
        "fukuoddslow": "fuku_odds",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)
    if "umaban" in df.columns:
        df["umaban"] = df["umaban"].apply(_to_int)
    if "tan_odds" in df.columns:
        df["tan_odds"] = df["tan_odds"].apply(_to_odds)
    if "fuku_odds" in df.columns:
        df["fuku_odds"] = df["fuku_odds"].apply(_to_odds)
    return df


def _transform_wide_odds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """n_odds_wide の生カラム名をML既存名に変換"""
    rename_map = {
        "oddslow": "odds_low",
        "oddshigh": "odds_high",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)
    if "odds_low" in df.columns:
        df["odds_low"] = df["odds_low"].apply(lambda v: _to_odds(v, divisor=100))
    if "odds_high" in df.columns:
        df["odds_high"] = df["odds_high"].apply(lambda v: _to_odds(v, divisor=100))
    return df


def _transform_time_series_columns(df: pd.DataFrame) -> pd.DataFrame:
    """n_jodds_tanpuku の生カラム名をML既存名に変換"""
    rename_map = {
        "happyotime": "happyo_time",
        "tanodds": "tan_odds",
        "fukuoddslow": "fuku_odds",
        "tanninki": "ninki",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)
    if "umaban" in df.columns:
        df["umaban"] = df["umaban"].apply(_to_int)
    if "tan_odds" in df.columns:
        df["tan_odds"] = df["tan_odds"].apply(_to_odds)
    if "fuku_odds" in df.columns:
        df["fuku_odds"] = df["fuku_odds"].apply(_to_odds)
    if "ninki" in df.columns:
        df["ninki"] = df["ninki"].apply(_to_int)
    return df


class DataRepository:
    """MLパイプラインのデータアクセス窓口。"""

    def __init__(self, store: ParquetStore) -> None:
        self.store = store

    # --- 読み取り（pyarrow filtersでプッシュダウン） ---

    def load_races(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "races", filters=_date_filters(start, end))
        df = _compute_race_id_from_raw(df)
        df = _transform_raw_columns(df)
        return _exclude_steeple(df)

    def load_entries(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "entries", filters=_date_filters(start, end))
        df = _transform_raw_columns(df)
        return _exclude_steeple(df)

    def load_odds_snapshots(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("odds", "snapshots", filters=_date_filters(start, end))
        df = _transform_odds_columns(df)
        return df

    def load_odds_time_series_range(self, start: str, end: str) -> pd.DataFrame:
        """オッズ時系列（日付範囲）— パーティションテーブル

        year/month パーティションに対して述語プッシュダウンを効かせるため、
        race_date フィルタに加えて year フィルタも追加。
        """
        s, e = _to_dt(start), _to_dt(end)
        filters = [
            ("year", ">=", s.year),
            ("year", "<=", e.year),
            ("race_date", ">=", s),
            ("race_date", "<=", e),
        ]
        df = self.store.read("odds", "time_series", filters=filters)
        df = _transform_time_series_columns(df)
        return df

    def load_odds_time_series(self, race_id: str) -> pd.DataFrame:
        """オッズ時系列（単一レース）"""
        return self.store.read("odds", "time_series", filters=[("race_id", "==", race_id)])

    def load_wide_odds(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("odds", "wide", filters=_date_filters(start, end))
        df = _transform_wide_odds_columns(df)
        return df

    def load_payouts(self, start: str, end: str) -> pd.DataFrame:
        df = self.store.read("raw", "payouts", filters=_date_filters(start, end))
        df = _transform_payouts_columns(df)
        return df

    # --- 全履歴参照（HorseHistoryFeatures用） ---

    def load_history_entries(self, lookback_years: int = 5) -> pd.DataFrame:
        """過去N年のentriesをロード。lookback_yearsでメモリ制御。

        注意: 障害レースを含む。HorseHistoryFeaturesが全成績を評価するため。
        """
        cutoff = datetime.now() - timedelta(days=lookback_years * 365)
        return self.store.read("raw", "entries", filters=[("race_date", ">=", cutoff)])

    def load_history_races(self, lookback_years: int = 5) -> pd.DataFrame:
        """過去N年のracesをロード。障害レースを含む（HorseHistoryFeatures用）。"""
        cutoff = datetime.now() - timedelta(days=lookback_years * 365)
        return self.store.read("raw", "races", filters=[("race_date", ">=", cutoff)])

    # --- 静的マスターデータ (horses/jockey/trainer stats) ---

    def load_horses(self) -> pd.DataFrame:
        """x_UMA 馬マスターデータ (血統・産駒成績) — 日付フィルタ不要"""
        return self.store.read("raw", "horses")

    def load_jockey_stats(self) -> pd.DataFrame:
        """x_KISYU_SEISEKI 騎手年度別成績 — 日付フィルタ不要"""
        return self.store.read("raw", "jockey_stats")

    def load_trainer_stats(self) -> pd.DataFrame:
        """x_CHOKYO_SEISEKI 調教師年度別成績 — 日付フィルタ不要"""
        return self.store.read("raw", "trainer_stats")

    # --- 特徴量キャッシュ ---

    def load_features(self, start: str, end: str) -> pd.DataFrame | None:
        """特徴量キャッシュがあれば返す、なければNone。"""
        if not self.store.exists("features", "horse_features"):
            return None
        return self.store.read("features", "horse_features", filters=_date_filters(start, end))

    def save_features(self, df: pd.DataFrame) -> None:
        self.store.write("features", "horse_features", df)

    # --- 予測・馬券 ---

    def save_predictions(self, df: pd.DataFrame) -> None:
        self.store.write("predictions", "predictions", df)

    def save_bets(self, df: pd.DataFrame) -> None:
        self.store.write("bets", "bets", df)
