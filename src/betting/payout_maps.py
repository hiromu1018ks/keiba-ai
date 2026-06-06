"""払戻マップ構築純粋関数 (D-09, D-12)

BacktestEngine と PaperReconciler で共有する払戻マップ構築ロジック。
I/O、EveryDB2 アクセス、クラス定義を含まない純粋関数モジュール。

各関数は payouts DataFrame を受け取り、(race_id, ...) -> odds_multiplier の dict を返す。
odds_multiplier = pay / 100.0 (100円あたりの円を倍率に変換、D-10 準拠)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) -> odds_multiplier のマップを構築。

    payfukusyopay は「100円あたりの円」なので、100で割って倍率に変換する。
    ベクトル化: melt + groupby で一括処理。同一 (race_id, umaban) の最大値を保持。
    """
    if payouts_df.empty:
        return {}
    id_vars = ["race_id"]
    maban_cols = [f"payfukusyoumaban{i}" for i in range(1, 6)]
    pay_cols = [f"payfukusyopay{i}" for i in range(1, 6)]

    maban_melted = payouts_df[id_vars + maban_cols].melt(
        id_vars=id_vars,
        value_vars=maban_cols,
        value_name="umaban",
    )
    pay_melted = payouts_df[id_vars + pay_cols].melt(
        id_vars=id_vars,
        value_vars=pay_cols,
        value_name="pay",
    )

    combined = pd.DataFrame(
        {
            "race_id": maban_melted["race_id"].values,
            "umaban": maban_melted["umaban"].values,
            "pay": pay_melted["pay"].values,
        }
    )
    combined = combined.dropna(subset=["umaban", "pay"])
    combined["umaban"] = combined["umaban"].astype(int)
    combined["pay_100"] = combined["pay"] / 100.0

    # 同一 (race_id, umaban) の最大値を保持
    idx = combined.groupby(["race_id", "umaban"], observed=True)["pay_100"].idxmax()
    deduped = combined.loc[idx]

    payout_map: dict[tuple[str, int], float] = {}
    for race_id, umaban, pay_100 in zip(
        deduped["race_id"].values, deduped["umaban"].values, deduped["pay_100"].values
    ):
        payout_map[(str(race_id), int(umaban))] = float(pay_100)
    return payout_map


def build_win_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int], float]:
    """payouts DataFrame から (race_id, umaban) -> odds_multiplier のマップを構築 (単勝用)。

    paytansyopay1 は「100円あたりの円」なので、100で割って倍率に変換する。
    ベクトル化: dropna -> astype -> dict comprehension。
    """
    if payouts_df.empty:
        return {}
    df = payouts_df.dropna(subset=["paytansyoumaban1", "paytansyopay1"]).copy()
    if df.empty:
        return {}
    df["umaban"] = df["paytansyoumaban1"].astype(int)
    df["pay_100"] = df["paytansyopay1"] / 100.0
    return {
        (str(race_id), int(umaban)): float(pay_100)
        for (race_id, umaban), pay_100 in df.set_index(["race_id", "umaban"])["pay_100"].items()
    }


def build_wide_payout_map(
    payouts_df: pd.DataFrame,
) -> dict[tuple[str, int, int], float]:
    """payouts DataFrame から (race_id, umaban_lo, umaban_hi) -> odds_multiplier のマップを構築。

    ワイド払戻は paywidekumi1-7 と paywidepay1-7 (100円あたり円) を使用。
    kumi 形式は非ゼロ埋め: "513" = 馬5+馬13, "1113" = 馬11+馬13, "15" = 馬1+馬5。
    ベクトル化: melt + str vectorized ops で一括処理。
    """
    if payouts_df.empty:
        return {}

    id_vars = ["race_id"]
    kumi_cols = [f"paywidekumi{i}" for i in range(1, 8)]
    pay_cols = [f"paywidepay{i}" for i in range(1, 8)]

    kumi_melted = payouts_df[id_vars + kumi_cols].melt(
        id_vars=id_vars,
        value_vars=kumi_cols,
        value_name="kumi",
    )
    pay_melted = payouts_df[id_vars + pay_cols].melt(
        id_vars=id_vars,
        value_vars=pay_cols,
        value_name="pay",
    )

    combined = pd.DataFrame(
        {
            "race_id": kumi_melted["race_id"].values,
            "kumi": kumi_melted["kumi"].values,
            "pay": pay_melted["pay"].values,
        }
    )
    combined = combined.dropna(subset=["kumi", "pay"])
    # BUG-FIX: Parquet may store kumi as float64 (e.g. 513.0).
    # Convert to str and strip trailing ".0" from float-as-string, preserving zero-padded strings.
    combined["kumi"] = combined["kumi"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    combined = combined[combined["kumi"] != ""]

    if combined.empty:
        return {}

    # Vectorized kumi parsing based on string length
    lengths = combined["kumi"].str.len()

    # Initialize lo/hi columns
    lo = pd.Series(np.nan, index=combined.index, dtype=float)
    hi = pd.Series(np.nan, index=combined.index, dtype=float)

    # Length 2: "XY" -> lo=X, hi=Y (e.g., "15" -> 1, 5)
    mask2 = lengths == 2
    if mask2.any():
        lo.loc[mask2] = combined.loc[mask2, "kumi"].str.slice(0, 1).astype(int)
        hi.loc[mask2] = combined.loc[mask2, "kumi"].str.slice(1, 2).astype(int)

    # Length 3: "XYZ" -- ambiguous: could be (X, YZ) or (XY, Z)
    # Heuristic: if int(XY) <= 18, use (XY, Z); else use (X, YZ)
    mask3 = lengths == 3
    if mask3.any():
        first_two = combined.loc[mask3, "kumi"].str.slice(0, 2).astype(int)
        use_first_two = first_two <= 18
        idx3 = combined.index[mask3]

        # Where first two digits form a valid horse number (1-18)
        split_at_2 = idx3[use_first_two]
        if len(split_at_2) > 0:
            lo.loc[split_at_2] = combined.loc[split_at_2, "kumi"].str.slice(0, 2).astype(int)
            hi.loc[split_at_2] = combined.loc[split_at_2, "kumi"].str.slice(2, 3).astype(int)

        # Otherwise split at 1
        split_at_1 = idx3[~use_first_two]
        if len(split_at_1) > 0:
            lo.loc[split_at_1] = combined.loc[split_at_1, "kumi"].str.slice(0, 1).astype(int)
            hi.loc[split_at_1] = combined.loc[split_at_1, "kumi"].str.slice(1, 3).astype(int)

    # Length 4: "XXYY" -> lo=XX, hi=YY (e.g., "1113" -> 11, 13)
    mask4 = lengths == 4
    if mask4.any():
        lo.loc[mask4] = combined.loc[mask4, "kumi"].str.slice(0, 2).astype(int)
        hi.loc[mask4] = combined.loc[mask4, "kumi"].str.slice(2, 4).astype(int)

    # Length 5: "XXYYZ" (rare, e.g. zero-padded "01113") -> treat as (XX, YYZ) or (XXX, YZ)
    mask5 = lengths >= 5
    if mask5.any():
        # Use same 2+3 or 3+2 logic based on first 2 digits
        first_two = combined.loc[mask5, "kumi"].str.slice(0, 2).astype(int)
        use_first_two = first_two <= 18
        idx5 = combined.index[mask5]

        split_at_2 = idx5[use_first_two]
        if len(split_at_2) > 0:
            lo.loc[split_at_2] = combined.loc[split_at_2, "kumi"].str.slice(0, 2).astype(int)
            hi.loc[split_at_2] = combined.loc[split_at_2, "kumi"].str.slice(2).astype(int)

        split_at_3 = idx5[~use_first_two]
        if len(split_at_3) > 0:
            lo.loc[split_at_3] = combined.loc[split_at_3, "kumi"].str.slice(0, -2).astype(int)
            hi.loc[split_at_3] = combined.loc[split_at_3, "kumi"].str.slice(-2).astype(int)

    combined["lo"] = lo
    combined["hi"] = hi
    combined = combined.dropna(subset=["lo", "hi"])
    combined["lo"] = combined["lo"].astype(int)
    combined["hi"] = combined["hi"].astype(int)
    combined["pay_100"] = combined["pay"] / 100.0

    # Ensure lo <= hi
    combined["_lo"] = np.minimum(combined["lo"], combined["hi"])
    combined["_hi"] = np.maximum(combined["lo"], combined["hi"])
    combined["lo"] = combined["_lo"]
    combined["hi"] = combined["_hi"]
    combined = combined.drop(columns=["_lo", "_hi", "kumi"])

    # Keep max payout per key
    idx = combined.groupby(["race_id", "lo", "hi"], observed=True)["pay_100"].idxmax()
    deduped = combined.loc[idx]

    wide_payout_map: dict[tuple[str, int, int], float] = {}
    for race_id, lo_val, hi_val, pay_100 in zip(
        deduped["race_id"].values,
        deduped["lo"].values,
        deduped["hi"].values,
        deduped["pay_100"].values,
    ):
        wide_payout_map[(str(race_id), int(lo_val), int(hi_val))] = float(pay_100)
    return wide_payout_map


# Alias for CONTEXT.md reference clarity
build_place_payout_map = build_payout_map
