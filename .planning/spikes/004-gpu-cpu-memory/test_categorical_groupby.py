"""Categorical dtype の groupby 回帰を詳しく調査。"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def main() -> None:
    n = 500_000
    df = pd.DataFrame({
        "race_id": [f"R{i % 20000:06d}" for i in range(n)],
        "umaban": [f"{i % 18:02d}" for i in range(n)],
        "kisyucode": [f"J{i % 1000:04d}" for i in range(n)],
        "chokyosicode": [f"T{i % 800:04d}" for i in range(n)],
        "value": np.random.rand(n),
    })

    print("=== Single-column groupby ===")
    for col in ["race_id", "kisyucode", "chokyosicode"]:
        df_obj = df.copy()
        df_cat = df.copy()
        df_cat[col] = df_cat[col].astype("category")

        t0 = time.time()
        _ = df_obj.groupby(col, observed=False)["value"].mean()
        obj_t = time.time() - t0

        t1 = time.time()
        _ = df_cat.groupby(col, observed=True)["value"].mean()
        cat_t = time.time() - t1

        print(f"  groupby({col}): obj={obj_t:.4f}s, cat(observed=True)={cat_t:.4f}s, "
              f"speedup={obj_t/cat_t:.1f}x")

    print("\n=== Multi-column groupby ===")
    for cols in [["race_id", "umaban"], ["race_id", "kisyucode"],
                  ["race_id", "umaban", "kisyucode"]]:
        df_obj = df.copy()
        df_cat = df.copy()
        for c in cols:
            df_cat[c] = df_cat[c].astype("category")

        t0 = time.time()
        _ = df_obj.groupby(cols, observed=False)["value"].mean()
        obj_t = time.time() - t0

        # observed=True で空カテゴリを除外
        t1 = time.time()
        _ = df_cat.groupby(cols, observed=True)["value"].mean()
        cat_t = time.time() - t1

        print(f"  groupby({cols}): obj={obj_t:.4f}s, cat(observed=True)={cat_t:.4f}s, "
              f"speedup={obj_t/cat_t:.1f}x")

    print("\n=== Merge on single column ===")
    df2 = df.groupby("race_id", observed=False)["value"].mean().reset_index()
    for label, use_cat in [("object", False), ("category", True)]:
        d = df.copy()
        d2 = df2.copy()
        if use_cat:
            d["race_id"] = d["race_id"].astype("category")
            d2["race_id"] = d2["race_id"].astype("category")

        t0 = time.time()
        _ = d.merge(d2, on="race_id", how="left")
        elapsed = time.time() - t0
        print(f"  merge(on=race_id) [{label}]: {elapsed:.4f}s")

    print("\n=== isin ===")
    sample_ids = df["race_id"].unique()[:1000]
    for label, use_cat in [("object", False), ("category", True)]:
        d = df.copy()
        if use_cat:
            d["race_id"] = d["race_id"].astype("category")

        t0 = time.time()
        _ = d[d["race_id"].isin(sample_ids)]
        elapsed = time.time() - t0
        print(f"  isin(1000 ids) [{label}]: {elapsed:.4f}s")

    print("\n=== Memory ===")
    mem_obj = df.memory_usage(deep=True).sum() / 1024 / 1024
    df_cat = df.copy()
    for c in ["race_id", "umaban", "kisyucode", "chokyosicode"]:
        df_cat[c] = df_cat[c].astype("category")
    mem_cat = df_cat.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"  object: {mem_obj:.1f}MB, category: {mem_cat:.1f}MB, "
          f"reduction: {(1-mem_cat/mem_obj)*100:.1f}%")


if __name__ == "__main__":
    main()
