"""カテゴリG: 市場クロス整合性特徴量 (Market Cross-Consistency Features)

単勝×ワイド×三連複のクロスコンシステンシー特徴量を計算する:
- rl_favorite_in_wide_top1: 1番人気がワイドninki=1組合せに含まれるか (0/1)
- rl_trio_overlap: 三連複ninki=1構成馬と単勝上位3頭のオーバーラップ数 (0-3)
- rl_market_consistency: 1番人気が三連複ninki=1組合せに含まれるか (0/1)
- rl_trio_odds_ratio: 実三連複ninki=1オッズ / Harville理論三連複オッズ
- rl_wide_harville_ratio: 実ワイドninki=1中間オッズ / Harville理論ワイドオッズ

全特徴量は tanodds (pre-race snapshot) のみを使用。
POST_RACE_COLS に含まれる列は一切使用しない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MCF_COLS: list[str] = [
    "rl_favorite_in_wide_top1",
    "rl_trio_overlap",
    "rl_market_consistency",
    "rl_trio_odds_ratio",
    "rl_wide_harville_ratio",
]


def _parse_wide_kumi(kumi: str) -> tuple[int, int]:
    """4桁ワイドkumi文字列を馬番ペアに変換.

    Args:
        kumi: 4桁ゼロ埋め文字列 (例: "0102" → umaban 1, 2)

    Returns:
        (馬番1, 馬番2) のタプル
    """
    try:
        return int(kumi[0:2]), int(kumi[2:4])
    except (ValueError, IndexError):
        return (-1, -1)


def _parse_trio_kumi(kumi: str) -> tuple[int, int, int]:
    """6桁三連複kumi文字列を馬番トリプルに変換.

    Args:
        kumi: 6桁ゼロ埋め文字列 (例: "010203" → umaban 1, 2, 3)

    Returns:
        (馬番1, 馬番2, 馬番3) のタプル
    """
    try:
        return int(kumi[0:2]), int(kumi[2:4]), int(kumi[4:6])
    except (ValueError, IndexError):
        return (-1, -1, -1)


def _harville_wide_prob(p_i: float, p_j: float) -> float:
    """Harville理論ワイド確率 (非順序ペア) (D-08).

    P(i,j) = P(i)*P(j)/(1-P(i)) + P(j)*P(i)/(1-P(j))
           = P(i)*P(j) * (1/(1-P(i)) + 1/(1-P(j)))

    Args:
        p_i: 馬iの正規化インプライド確率
        p_j: 馬jの正規化インプライド確率

    Returns:
        理論ワイド確率 (両順序の和)
    """
    eps = 1e-10
    denom_i = max(1.0 - p_i, eps)
    denom_j = max(1.0 - p_j, eps)
    return p_i * p_j * (1.0 / denom_i + 1.0 / denom_j)


def _harville_trio_prob(p_i: float, p_j: float, p_k: float) -> float:
    """Harville理論三連複確率 (非順序トリプル) (D-08).

    P(i,j,k) = sum over all 6 permutations of:
        P(first) * P(second)/(1-P(first)) * P(third)/(1-P(first)-P(second))

    Args:
        p_i: 馬iの正規化インプライド確率
        p_j: 馬jの正規化インプライド確率
        p_k: 馬kの正規化インプライド確率

    Returns:
        理論三連複確率 (6順列の和)
    """
    eps = 1e-10
    perms: list[tuple[float, float, float]] = [
        (p_i, p_j, p_k), (p_i, p_k, p_j),
        (p_j, p_i, p_k), (p_j, p_k, p_i),
        (p_k, p_i, p_j), (p_k, p_j, p_i),
    ]
    total = 0.0
    for a, b, c in perms:
        denom1 = max(1.0 - a, eps)
        denom2 = max(1.0 - a - b, eps)
        total += a * (b / denom1) * (c / denom2)
    return total


def _get_favorite_umaban(
    tanodds_group: pd.Series,
    umaban_group: pd.Series,
    rank: int = 1,
) -> int | float:
    """tanodds最小(=1番人気)のumabanを取得.

    Args:
        tanodds_group: 単一レースのtanodds Series
        umaban_group: 対応するumaban Series
        rank: 取得する順位 (1=1番人気)

    Returns:
        指定順位のumaban。該当なしの場合はNaN
    """
    valid = tanodds_group.dropna()
    if len(valid) < rank:
        return np.nan
    sorted_idx = valid.sort_values().index
    return int(umaban_group.loc[sorted_idx[rank - 1]])


def _get_top3_umaban(
    tanodds_group: pd.Series,
    umaban_group: pd.Series,
) -> set[int]:
    """単勝上位3頭のumabanセットを取得.

    Args:
        tanodds_group: 単一レースのtanodds Series
        umaban_group: 対応するumaban Series

    Returns:
        上位3頭のumabanセット
    """
    valid = tanodds_group.dropna()
    n = min(3, len(valid))
    if n == 0:
        return set()
    sorted_idx = valid.sort_values().index[:n]
    return set(umaban_group.loc[sorted_idx].astype(int).tolist())


def _compute_implied_probabilities(tanodds_group: pd.Series) -> pd.Series:
    """tanoddsから正規化インプライド確率を計算.

    P(i) = (1/tanodds_i) / sum(1/tanodds_j) per race

    Args:
        tanodds_group: 単一レースのtanodds Series (NaN/0除外済み)

    Returns:
        正規化インプライド確率のSeries (インデックスはtanodds_groupと同じ)
    """
    inv = 1.0 / tanodds_group
    total = inv.sum()
    if total == 0 or np.isnan(total):
        return pd.Series(np.nan, index=tanodds_group.index)
    return inv / total


def _compute_for_single_race(
    df: pd.DataFrame,
    tanodds: pd.Series,
    wide_race: pd.DataFrame,
    trio_race: pd.DataFrame,
) -> pd.DataFrame:
    """単一レースのMCF特徴量計算 (race_idなし、groupbyなし).

    build_features() パリティ用。DataFrame全体を1レースとして処理する。

    Args:
        df: 特徴量追加先のDataFrame
        tanodds: 前処理済みtanodds Series (NaN/0除外)
        wide_race: 単一レースのワイドオッズ (ninki=1フィルタ済み)
        trio_race: 単一レースの三連複オッズ (ninki=1フィルタ済み)

    Returns:
        MCF列が追加されたDataFrame
    """
    valid_odds = tanodds.dropna()

    if len(valid_odds) == 0:
        for col in MCF_COLS:
            df[col] = np.nan
        return df

    # 人気順位判定用umaban
    umaban = df["umaban"] if "umaban" in df.columns else pd.Series(range(1, len(df) + 1))

    # 1番人気のumaban
    fav1_umaban = _get_favorite_umaban(valid_odds, umaban)
    # 上位3頭のumabanセット
    top3_set = _get_top3_umaban(valid_odds, umaban)

    # 正規化インプライド確率
    p_norm = _compute_implied_probabilities(valid_odds)

    # ワイドninki=1組合せ
    wide_n1 = _filter_ninki1(wide_race)
    trio_n1 = _filter_ninki1(trio_race)

    # rl_favorite_in_wide_top1
    fav_in_wide = _check_favorite_in_wide(wide_n1, fav1_umaban)

    # ワイドHarville比率
    wide_harville_ratio = _compute_wide_harville_ratio(wide_n1, valid_odds, umaban, p_norm)

    # 三連複ninki=1の処理
    trio_overlap, market_consistency, trio_odds_ratio = _compute_trio_features(
        trio_n1, fav1_umaban, top3_set, valid_odds, umaban, p_norm,
    )

    df["rl_favorite_in_wide_top1"] = fav_in_wide
    df["rl_trio_overlap"] = trio_overlap
    df["rl_market_consistency"] = market_consistency
    df["rl_trio_odds_ratio"] = trio_odds_ratio
    df["rl_wide_harville_ratio"] = wide_harville_ratio

    return df


def _filter_ninki1(df: pd.DataFrame) -> pd.DataFrame:
    """ninki=1の行をフィルタ (文字列・整数両対応)."""
    if df.empty or "ninki" not in df.columns:
        return pd.DataFrame()
    ninki_num = pd.to_numeric(df["ninki"], errors="coerce")
    return df[ninki_num == 1]


def _check_favorite_in_wide(wide_n1: pd.DataFrame, fav_umaban: int | float) -> float:
    """1番人気がワイドninki=1組合せに含まれるか."""
    if wide_n1.empty or pd.isna(fav_umaban):
        return np.nan
    for _, row in wide_n1.iterrows():
        h1, h2 = _parse_wide_kumi(str(row["kumi"]))
        if h1 == fav_umaban or h2 == fav_umaban:
            return 1.0
    return 0.0


def _compute_wide_harville_ratio(
    wide_n1: pd.DataFrame,
    tanodds_valid: pd.Series,
    umaban: pd.Series,
    p_norm: pd.Series,
) -> float:
    """ワイドHarville理論オッズ比率を計算."""
    if wide_n1.empty:
        return np.nan

    row = wide_n1.iloc[0]
    h1, h2 = _parse_wide_kumi(str(row["kumi"]))

    # 中間オッズ
    if "oddslow" not in wide_n1.columns or "oddshigh" not in wide_n1.columns:
        return np.nan
    wide_mid = (float(row["oddslow"]) + float(row["oddshigh"])) / 2.0
    if wide_mid <= 0 or np.isnan(wide_mid):
        return np.nan

    # P(h1), P(h2) を取得
    p_h1, p_h2 = _get_prob_for_umaban(h1, h2, tanodds_valid, umaban, p_norm)
    if np.isnan(p_h1) or np.isnan(p_h2):
        return np.nan

    # Harville理論確率
    theo_prob = _harville_wide_prob(p_h1, p_h2)
    if theo_prob <= 0 or np.isnan(theo_prob):
        return np.nan

    # 理論オッズ = 1 / 理論確率
    theo_odds = 1.0 / theo_prob

    return wide_mid / theo_odds


def _get_prob_for_umaban(
    h1: int,
    h2: int,
    tanodds_valid: pd.Series,
    umaban: pd.Series,
    p_norm: pd.Series,
) -> tuple[float, float]:
    """指定umabanのインプライド確率を取得."""
    # umabanは元DataFrameのインデックスに対応
    p_h1 = np.nan
    p_h2 = np.nan
    for idx in tanodds_valid.index:
        try:
            ub = int(umaban.loc[idx])
        except (KeyError, ValueError, TypeError):
            continue
        if ub == h1 and idx in p_norm.index:
            p_h1 = float(p_norm.loc[idx])
        if ub == h2 and idx in p_norm.index:
            p_h2 = float(p_norm.loc[idx])
    return p_h1, p_h2


def _compute_trio_features(
    trio_n1: pd.DataFrame,
    fav_umaban: int | float,
    top3_set: set[int],
    tanodds_valid: pd.Series,
    umaban: pd.Series,
    p_norm: pd.Series,
) -> tuple[float, float, float]:
    """三連複関連特徴量の計算.

    Returns:
        (trio_overlap, market_consistency, trio_odds_ratio)
    """
    if trio_n1.empty:
        return np.nan, np.nan, np.nan

    row = trio_n1.iloc[0]
    h1, h2, h3 = _parse_trio_kumi(str(row["kumi"]))

    if h1 < 0 or h2 < 0 or h3 < 0:
        return np.nan, np.nan, np.nan

    trio_horses = {h1, h2, h3}

    # trio_overlap: 上位3頭とのオーバーラップ数
    overlap = float(len(trio_horses & top3_set))

    # market_consistency: 1番人気が三連複ninki=1に含まれるか
    if pd.isna(fav_umaban):
        consistency = np.nan
    else:
        consistency = 1.0 if int(fav_umaban) in trio_horses else 0.0

    # trio_odds_ratio: 実オッズ / Harville理論オッズ
    if "odds" not in trio_n1.columns:
        trio_ratio = np.nan
    else:
        actual_odds = float(row["odds"])
        if actual_odds <= 0 or np.isnan(actual_odds):
            trio_ratio = np.nan
        else:
            p_a, p_b = _get_prob_for_umaban(h1, h2, tanodds_valid, umaban, p_norm)
            _, p_c = _get_prob_for_umaban(h3, h3, tanodds_valid, umaban, p_norm)
            # p_cのために正しく取得
            p_h3 = np.nan
            for idx in tanodds_valid.index:
                try:
                    ub = int(umaban.loc[idx])
                except (KeyError, ValueError, TypeError):
                    continue
                if ub == h3 and idx in p_norm.index:
                    p_h3 = float(p_norm.loc[idx])

            if np.isnan(p_a) or np.isnan(p_b) or np.isnan(p_h3):
                trio_ratio = np.nan
            else:
                theo_prob = _harville_trio_prob(p_a, p_b, p_h3)
                if theo_prob <= 0 or np.isnan(theo_prob):
                    trio_ratio = np.nan
                else:
                    theo_odds = 1.0 / theo_prob
                    trio_ratio = actual_odds / theo_odds

    return overlap, consistency, trio_ratio


def _compute_for_multi_race(
    df: pd.DataFrame,
    tanodds: pd.Series,
    wide_df: pd.DataFrame,
    trio_df: pd.DataFrame,
) -> pd.DataFrame:
    """複数レースのMCF特徴量計算 (groupby("race_id")版).

    build_all() パス用。race_idでグループ化してレースごとに計算する。

    Args:
        df: 特徴量追加先のDataFrame
        tanodds: 前処理済みtanodds Series (NaN/0除外)
        wide_df: ワイドオッズDataFrame (race_id, kumi, oddslow, oddshigh, ninki)
        trio_df: 三連複オッズDataFrame (race_id, kumi, odds, ninki)

    Returns:
        MCF列が追加されたDataFrame
    """
    valid_mask = tanodds.notna()

    if not valid_mask.any():
        for col in MCF_COLS:
            df[col] = np.nan
        return df

    race_ids = df["race_id"]

    # ninki=1フィルタ
    wide_n1 = _filter_ninki1(wide_df)
    trio_n1 = _filter_ninki1(trio_df)

    # レースごとに計算 (for loopでname/groupを直接処理)
    results: dict[str, tuple[float, float, float, float, float]] = {}

    grouped = df.groupby("race_id", observed=True)
    for race_id_val, group in grouped:
        tanodds_race = pd.to_numeric(group["tanodds"], errors="coerce").replace(0, np.nan)
        valid = tanodds_race.dropna()

        if len(valid) == 0:
            results[race_id_val] = (np.nan, np.nan, np.nan, np.nan, np.nan)
            continue

        umaban_race = group["umaban"] if "umaban" in group.columns else pd.Series(
            range(1, len(group) + 1), index=group.index,
        )

        fav1 = _get_favorite_umaban(valid, umaban_race)
        top3 = _get_top3_umaban(valid, umaban_race)
        p_norm_race = _compute_implied_probabilities(valid)

        has_wide = not wide_n1.empty and "race_id" in wide_n1.columns
        has_trio = not trio_n1.empty and "race_id" in trio_n1.columns
        wide_race = wide_n1[wide_n1["race_id"] == race_id_val] if has_wide else pd.DataFrame()
        trio_race = trio_n1[trio_n1["race_id"] == race_id_val] if has_trio else pd.DataFrame()

        fav_in_wide = _check_favorite_in_wide(wide_race, fav1)
        wide_ratio = _compute_wide_harville_ratio(wide_race, valid, umaban_race, p_norm_race)
        trio_overlap, market_consistency, trio_ratio = _compute_trio_features(
            trio_race, fav1, top3, valid, umaban_race, p_norm_race,
        )

        results[race_id_val] = (
            fav_in_wide, trio_overlap, market_consistency, trio_ratio, wide_ratio,
        )

    # 結果をブロードキャスト
    result_series = pd.Series(results)

    df["rl_favorite_in_wide_top1"] = race_ids.map(result_series.map(lambda x: x[0]))
    df["rl_trio_overlap"] = race_ids.map(result_series.map(lambda x: x[1]))
    df["rl_market_consistency"] = race_ids.map(result_series.map(lambda x: x[2]))
    df["rl_trio_odds_ratio"] = race_ids.map(result_series.map(lambda x: x[3]))
    df["rl_wide_harville_ratio"] = race_ids.map(result_series.map(lambda x: x[4]))

    return df


def compute_market_cross_features(
    df: pd.DataFrame,
    wide_df: pd.DataFrame | None = None,
    trio_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """市場クロス整合性特徴量を計算 (MCF-01~06).

    単勝×ワイド×三連複のオッズ整合性から5特徴量を計算し、全馬にブロードキャストする。
    全特徴量は tanodds (pre-race snapshot) のみを使用し、POST_RACE列は一切使用しない。

    build_all() と build_features() の両パスから呼び出される:
    - build_all(): 複数レース → groupby("race_id") で計算
    - build_features(): 単一レース → groupbyなしで全体を1レースとして計算

    Args:
        df: race_id, tanodds, umaban を含むDataFrame
            (race_idはオプショナル — なしの場合は単一レースとして処理)
        wide_df: ワイドオッズDataFrame (optional)。None/空の場合はNaNフォールバック (D-06)
        trio_df: 三連複オッズDataFrame (optional)。None/空の場合はNaNフォールバック (D-06)

    Returns:
        MCF列が追加されたDataFrame (入力は変更されない)
    """
    df = df.copy()

    # wide_df/trio_dfがNoneまたは空の場合、全MCF列をNaNで初期化 (D-06)
    if wide_df is None or trio_df is None or wide_df.empty or trio_df.empty:
        for col in MCF_COLS:
            df[col] = np.nan
        return df

    # tanodds列なし → 全MCF列をNaNで初期化
    if "tanodds" not in df.columns:
        for col in MCF_COLS:
            df[col] = np.nan
        return df

    # tanoddsの前処理: 数値化 → 0をNaNに変換
    tanodds = pd.to_numeric(df["tanodds"], errors="coerce").replace(0, np.nan)

    # race_idの有無で分岐
    if "race_id" in df.columns:
        return _compute_for_multi_race(df, tanodds, wide_df, trio_df)
    else:
        return _compute_for_single_race(df, tanodds, wide_df, trio_df)
