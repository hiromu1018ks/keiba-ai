"""Spike 001 (follow-up): JRAレース限定での payouts 欠損分析。

race_id から jyocd を抽出し、JRAレース (jyocd 1-10) に限定した場合の
payouts カバレッジを確認する。

Usage:
    cd <project root>
    python .planning/spikes/001-win-payout-missing/diagnose_jra_only.py
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, str(Path(ROOT) / "src"))

import pandas as pd
from db.parquet_store import ParquetStore


def extract_jyocd(race_id: str) -> int | None:
    """race_id から jyocd を抽出。
    Format: YYYYMMDD + jyocd(2) + kaiji(2) + nichiji(2) + racenum(2) = 16桁
    """
    s = str(race_id)
    if len(s) >= 12:
        try:
            return int(s[8:10])
        except (ValueError, IndexError):
            return None
    return None


def main() -> None:
    store = ParquetStore()

    payouts_df = store.read("raw", "payouts")
    entries_df = store.read("raw", "entries")
    races_df = store.read("raw", "races")

    print(f"payouts: {len(payouts_df):,} rows")
    print(f"entries: {len(entries_df):,} rows")
    print(f"races:   {len(races_df):,} rows")
    print()

    # jyocd を抽出
    payouts_df["jyocd_extracted"] = payouts_df["race_id"].apply(extract_jyocd)
    entries_df["jyocd_extracted"] = entries_df["race_id"].apply(extract_jyocd)
    races_df["jyocd_extracted"] = races_df["race_id"].apply(extract_jyocd) if "race_id" in races_df.columns else None

    # --- JRAレース (jyocd 1-10) に限定 ---
    jra_payouts = payouts_df[payouts_df["jyocd_extracted"].between(1, 10)]
    jra_entries = entries_df[entries_df["jyocd_extracted"].between(1, 10)]
    jra_races = races_df[races_df["jyocd_extracted"].between(1, 10)] if "jyocd_extracted" in races_df.columns else races_df

    print("=== JRA-only (jyocd 1-10) ===")
    print(f"  JRA payouts: {len(jra_payouts):,}")
    print(f"  JRA entries: {len(jra_entries):,}")
    print(f"  JRA races:   {len(jra_races):,}")
    print()

    # --- JRAレースの race_id coverage ---
    jra_entry_race_ids = set(jra_entries["race_id"].dropna().astype(str).unique())
    jra_payout_race_ids = set(jra_payouts["race_id"].dropna().astype(str).unique())

    jra_only_entries = jra_entry_race_ids - jra_payout_race_ids
    jra_only_payouts = jra_payout_race_ids - jra_entry_race_ids

    print("=== JRA race_id coverage ===")
    print(f"  JRA races in entries: {len(jra_entry_race_ids):,}")
    print(f"  JRA races in payouts: {len(jra_payout_race_ids):,}")
    print(f"  JRA only in entries (no payout): {len(jra_only_entries):,}")
    print(f"  JRA only in payouts (no entry):  {len(jra_only_payouts):,}")

    if jra_only_entries:
        # 欠損レースの jyocd 分布
        missing_jyocd = {}
        for rid in jra_only_entries:
            jcd = extract_jyocd(rid)
            missing_jyocd[jcd] = missing_jyocd.get(jcd, 0) + 1
        print("  missing by jyocd:")
        for jcd in sorted(missing_jyocd):
            print(f"    jyocd={jcd:02d}: {missing_jyocd[jcd]:,}")

        # 欠損レースの年度分布
        missing_years = {}
        for rid in jra_only_entries:
            yr = str(rid)[:4]
            missing_years[yr] = missing_years.get(yr, 0) + 1
        print("  missing by year:")
        for yr in sorted(missing_years):
            print(f"    {yr}: {missing_years[yr]:,}")

        # サンプル
        sample = sorted(jra_only_entries)[:10]
        print(f"  sample: {sample}")

    print()

    # --- entries に jyocd 列がある場合の確認 ---
    if "jyocd" in entries_df.columns:
        print("=== entries jyocd column values ===")
        jyocd_vals = entries_df["jyocd"].dropna().unique()
        print(f"  unique jyocd values: {sorted(jyocd_vals)[:30]}")
        print()

    if "jyocd" in payouts_df.columns:
        print("=== payouts jyocd column values ===")
        jyocd_vals = payouts_df["jyocd"].dropna().unique()
        print(f"  unique jyocd values: {sorted(jyocd_vals)[:30]}")
        print()

    # --- JRA 1着馬の payouts カバレッジ ---
    jra_winners = jra_entries[jra_entries["kakuteijyuni"] == 1][["race_id", "umaban"]].copy()
    jra_winners["race_id"] = jra_winners["race_id"].astype(str)
    jra_winners["umaban"] = jra_winners["umaban"].astype(int)
    jra_winner_keys = set(zip(jra_winners["race_id"], jra_winners["umaban"]))

    jra_valid_tansyo = jra_payouts.dropna(subset=["paytansyoumaban1", "paytansyopay1"]).copy()
    jra_valid_tansyo["umaban"] = jra_valid_tansyo["paytansyoumaban1"].astype(int)
    jra_valid_tansyo["race_id"] = jra_valid_tansyo["race_id"].astype(str)
    jra_payout_winner_keys = set(zip(jra_valid_tansyo["race_id"], jra_valid_tansyo["umaban"]))

    jra_real_missing = jra_winner_keys - jra_payout_winner_keys

    print("=== JRA winner payout coverage ===")
    print(f"  JRA winners: {len(jra_winner_keys):,}")
    print(f"  JRA payout entries: {len(jra_payout_winner_keys):,}")
    print(f"  JRA WINNERS missing: {len(jra_real_missing):,} ({len(jra_real_missing)/len(jra_winner_keys):.1%})")

    if jra_real_missing:
        jra_missing_years = {}
        for rid, _ in jra_real_missing:
            yr = str(rid)[:4]
            jra_missing_years[yr] = jra_missing_years.get(yr, 0) + 1
        print("  JRA missing by year:")
        for yr in sorted(jra_missing_years):
            print(f"    {yr}: {jra_missing_years[yr]:,}")
    print()

    # --- engine.py のフィルタと同じ条件をシミュレート ---
    # engine.py line 540-544:
    #   jyocd_int = pd.to_numeric(race_df["jyocd"], errors="coerce")
    #   jra_race_ids = race_df.loc[jyocd_int.between(1, 10), "race_id"].drop_duplicates()
    #   race_df = race_df[race_df["race_id"].isin(jra_race_ids)]
    if "jyocd" in races_df.columns:
        jyocd_int = pd.to_numeric(races_df["jyocd"], errors="coerce")
        engine_jra_rids = set(races_df.loc[jyocd_int.between(1, 10), "race_id"].dropna().astype(str).unique())
        engine_jra_payouts = jra_payout_race_ids & engine_jra_rids
        engine_jra_missing = engine_jra_rids - jra_payout_race_ids
        print("=== Engine simulation (races table jyocd filter) ===")
        print(f"  races after jyocd 1-10 filter: {len(engine_jra_rids):,}")
        print(f"  of those, in payouts: {len(engine_jra_payouts):,}")
        print(f"  of those, MISSING from payouts: {len(engine_jra_missing):,}")

        if engine_jra_missing:
            engine_missing_years = {}
            for rid in engine_jra_missing:
                yr = str(rid)[:4]
                engine_missing_years[yr] = engine_missing_years.get(yr, 0) + 1
            print("  missing by year:")
            for yr in sorted(engine_missing_years):
                print(f"    {yr}: {engine_missing_years[yr]:,}")
    print()


if __name__ == "__main__":
    main()
