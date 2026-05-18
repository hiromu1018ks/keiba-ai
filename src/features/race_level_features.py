"""カテゴリF: レースレベル集約特徴量

レース全体の市場構造を表す特徴量を計算する:
- rl_log_odds_entropy: インプライド確率のシャノンエントロピー (拮抗度)
- rl_odds_dispersion: レース内tanoddsの標準偏差 (オッズ散らばり)
- rl_top3_odds_gap: 1番人気と3番人気のtanodds差 (混戦度)
- rl_top1_odds: 1番人気のtanodds値を全馬にブロードキャスト (鉄板度)
- rl_favorite_rank_gap: 1番人気と2番人気の対数オッズ差 (支配度)
- rl_n_horses: 出走頭数

全特徴量は tanodds (pre-race snapshot) のみを使用。
POST_RACE_COLS に含まれる列は一切使用しない (D-03)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RL_COLS: list[str] = [
    "rl_log_odds_entropy",
    "rl_odds_dispersion",
    "rl_top3_odds_gap",
    "rl_top1_odds",
    "rl_favorite_rank_gap",
    "rl_n_horses",
]


def _calc_log_odds_entropy(p_series: pd.Series) -> float:
    """インプライド確率のシャノンエントロピー: H = -sum(p * log(p))

    market_bias_features.py の _calc_entropy パターンを踏襲。
    p > 0 のみを使用して log(0) を回避。

    Args:
        p_series: 正規化済みインプライド確率のSeries

    Returns:
        シャノンエントロピー値
    """
    p = p_series.values.astype(float)
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return float(-np.sum(p * np.log(p)))


def _extract_odds_by_rank(
    tanodds_group: pd.Series,
    rank: int,
) -> float:
    """グループ内のオッズ順位(昇順)から指定順位のオッズを抽出

    Args:
        tanodds_group: 単一レースのtanodds Series (NaN/0除外済み)
        rank: 取得する順位 (1=1番人気, 2=2番人気, ...)

    Returns:
        指定順位のオッズ値。該当順位が存在しない場合はNaN
    """
    sorted_odds = np.sort(tanodds_group.values)
    if rank < 1 or len(sorted_odds) < rank:
        return np.nan
    return float(sorted_odds[rank - 1])


def _compute_for_single_race(df: pd.DataFrame, tanodds: pd.Series) -> pd.DataFrame:
    """単一レースの特徴量計算 (race_idなし、groupbyなし)

    build_features() パリティ用。DataFrame全体を1レースとして処理する。

    Args:
        df: 特徴量追加先のDataFrame
        tanodds: 前処理済みtanodds Series (NaN/0除外)

    Returns:
        rl_* 列が追加されたDataFrame
    """
    valid_odds = tanodds.dropna()
    n_valid = len(valid_odds)

    if n_valid == 0:
        for col in RL_COLS:
            df[col] = np.nan
        return df

    # インプライド確率の計算と正規化
    inv_odds = 1.0 / valid_odds
    total = inv_odds.sum()
    if total == 0 or np.isnan(total):
        for col in RL_COLS:
            df[col] = np.nan
        return df

    p_norm = inv_odds / total

    # RLF-01: シャノンエントロピー
    entropy = _calc_log_odds_entropy(p_norm)
    df["rl_log_odds_entropy"] = entropy

    # RLF-02: オッズ標準偏差
    if n_valid >= 2:
        df["rl_odds_dispersion"] = float(valid_odds.std(ddof=1))
    else:
        df["rl_odds_dispersion"] = np.nan

    # RLF-03: 1番人気と3番人気のオッズ差
    fav1 = _extract_odds_by_rank(valid_odds, 1)
    fav3 = _extract_odds_by_rank(valid_odds, 3)
    if np.isnan(fav1) or np.isnan(fav3):
        df["rl_top3_odds_gap"] = np.nan
    else:
        df["rl_top3_odds_gap"] = fav3 - fav1

    # RLF-04: 1番人気オッズ
    df["rl_top1_odds"] = fav1

    # RLF-05: 1番人気と2番人気の対数オッズ差
    fav2 = _extract_odds_by_rank(valid_odds, 2)
    if np.isnan(fav1) or np.isnan(fav2) or fav1 <= 0 or fav2 <= 0:
        df["rl_favorite_rank_gap"] = np.nan
    else:
        df["rl_favorite_rank_gap"] = np.log(fav2 / fav1)

    # RLF-06: 出走頭数
    df = _assign_n_horses(df, n_valid)

    return df


def _compute_for_multi_race(df: pd.DataFrame, tanodds: pd.Series) -> pd.DataFrame:
    """複数レースの特徴量計算 (groupby("race_id")版)

    build_all() パス用。race_idでグループ化してレースごとに計算する。

    Args:
        df: 特徴量追加先のDataFrame
        tanodds: 前処理済みtanodds Series (NaN/0除外)

    Returns:
        rl_* 列が追加されたDataFrame
    """
    valid_mask = tanodds.notna()
    has_any_valid = valid_mask.any()

    if not has_any_valid:
        for col in RL_COLS:
            df[col] = np.nan
        return df

    # 有効オッズのみで作業用Series作成
    tanodds_valid = tanodds.where(valid_mask)
    race_ids = df["race_id"]

    # インプライド確率
    inv_odds = 1.0 / tanodds_valid
    # レースごとの正規化
    race_sum = inv_odds.groupby(race_ids, observed=True).transform("sum")
    p_norm = inv_odds / race_sum.replace(0, np.nan)

    # RLF-01: シャノンエントロピー per race
    def _entropy_per_race(group: pd.Series) -> float:
        return _calc_log_odds_entropy(group)

    entropy_values = p_norm.groupby(race_ids, observed=True).apply(
        _entropy_per_race, include_groups=False
    )
    df["rl_log_odds_entropy"] = race_ids.map(entropy_values)

    # RLF-02: オッズ標準偏差
    df["rl_odds_dispersion"] = tanodds_valid.groupby(race_ids, observed=True).transform(
        "std"
    )

    # RLF-03, RLF-04, RLF-05: 順位ベースの特徴量
    # compute_flb_slope() の _race_shape パターンを踏襲 (tuple返却 + map)
    def _rank_features(group: pd.Series) -> tuple[float, float, float]:
        """レース内のオッズ順位から複数の特徴量を同時計算

        Returns:
            (fav1, top3_gap, rank_gap) のタプル
        """
        sorted_odds = np.sort(group.dropna().values)
        n = len(sorted_odds)
        fav1 = float(sorted_odds[0]) if n >= 1 else np.nan
        fav2 = float(sorted_odds[1]) if n >= 2 else np.nan
        fav3 = float(sorted_odds[2]) if n >= 3 else np.nan

        top3_gap = (fav3 - fav1) if not (np.isnan(fav1) or np.isnan(fav3)) else np.nan

        if not (np.isnan(fav1) or np.isnan(fav2)) and fav1 > 0 and fav2 > 0:
            rank_gap = np.log(fav2 / fav1)
        else:
            rank_gap = np.nan

        return fav1, top3_gap, rank_gap

    rank_results = tanodds_valid.groupby(race_ids, observed=True).apply(
        _rank_features, include_groups=False
    )

    df["rl_top1_odds"] = race_ids.map(
        rank_results.map(lambda x: x[0])
    )
    df["rl_top3_odds_gap"] = race_ids.map(
        rank_results.map(lambda x: x[1])
    )
    df["rl_favorite_rank_gap"] = race_ids.map(
        rank_results.map(lambda x: x[2])
    )

    # RLF-06: 出走頭数
    df = _assign_n_horses_grouped(df, race_ids, tanodds_valid)

    return df


def _assign_n_horses(df: pd.DataFrame, n_valid: int) -> pd.DataFrame:
    """出走頭数を割り当て (単一レース用)

    field_size列が存在し、かつ有効な値(>0)の場合はそのまま使用。
    そうでない場合は有効オッズ数(n_valid)を使用。

    Args:
        df: DataFrame
        n_valid: 有効なオッズ値の数

    Returns:
        rl_n_horses列が追加されたDataFrame
    """
    if "field_size" in df.columns:
        field_size_val = df["field_size"].iloc[0]
        if pd.notna(field_size_val) and field_size_val > 0:
            df["rl_n_horses"] = int(field_size_val)
            return df
    df["rl_n_horses"] = n_valid
    return df


def _assign_n_horses_grouped(
    df: pd.DataFrame,
    race_ids: pd.Series,
    tanodds_valid: pd.Series,
) -> pd.DataFrame:
    """出走頭数を割り当て (groupby版)

    field_size列が存在し、かつ有効な値(>0)の場合はそのまま使用。
    そうでない場合はレース内の有効オッズ数を使用。

    Args:
        df: DataFrame
        race_ids: race_id Series
        tanodds_valid: 前処理済みtanodds Series

    Returns:
        rl_n_horses列が追加されたDataFrame
    """
    if "field_size" in df.columns:
        fs = df["field_size"].copy()
        fs = pd.to_numeric(fs, errors="coerce").fillna(0)
        # field_size > 0 のレースはそのまま、0のレースはgroupby.size()で補完
        valid_count = tanodds_valid.groupby(race_ids, observed=True).count()
        # broadcast用にmap
        fallback = race_ids.map(valid_count)
        df["rl_n_horses"] = np.where(fs > 0, fs, fallback).astype(float)
    else:
        valid_count = tanodds_valid.groupby(race_ids, observed=True).count()
        df["rl_n_horses"] = race_ids.map(valid_count).astype(float)

    return df


def compute_race_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """レース構造特徴量を計算 (RLF-01~06)

    レース全体の市場構造を表す6特徴量を計算し、全馬にブロードキャストする。
    全特徴量は tanodds (pre-race snapshot) のみを使用し、POST_RACE列は一切使用しない。

    build_all() と build_features() の両パスから呼び出される:
    - build_all(): 複数レース → groupby("race_id") で計算
    - build_features(): 単一レース → groupbyなしで全体を1レースとして計算

    Args:
        df: race_id, tanodds, field_size を含むDataFrame
            (race_idはオプショナル — なしの場合は単一レースとして処理)

    Returns:
        rl_* 列が追加されたDataFrame (入力は変更されない)
    """
    df = df.copy()

    # tanodds列なし → 全rl_*をNaNで初期化
    if "tanodds" not in df.columns:
        for col in RL_COLS:
            df[col] = np.nan
        return df

    # tanoddsの前処理: 数値化 → 0をNaNに変換
    tanodds = pd.to_numeric(df["tanodds"], errors="coerce").replace(0, np.nan)

    # race_idの有無で分岐
    if "race_id" in df.columns:
        return _compute_for_multi_race(df, tanodds)
    else:
        return _compute_for_single_race(df, tanodds)
