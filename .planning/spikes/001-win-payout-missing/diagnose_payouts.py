"""Spike 001: Win payout missing の原因調査。

payouts parquet と entries parquet を比較し、
「payouts に単勝払戻データがないレース」を特定・分類する。

Usage:
    cd <project root>
    python .planning/spikes/001-win-payout-missing/diagnose_payouts.py
"""

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, str(Path(ROOT) / "src"))

import pandas as pd
from db.parquet_store import ParquetStore


def main() -> None:
    store = ParquetStore()

    # --- 1. payouts を全期間ロード ---
    payouts_df = store.read("raw", "payouts")
    print(f"payouts rows: {len(payouts_df):,}")
    print(f"payouts columns: {list(payouts_df.columns)}")
    print()

    # --- 2. entries を全期間ロード ---
    entries_df = store.read("raw", "entries")
    print(f"entries rows: {len(entries_df):,}")
    print()

    # --- 3. payouts の NULL 分析 ---
    # paytansyoumaban1 / paytansyopay1 が NULL の割合
    total_payouts = len(payouts_df)
    null_tansyo_umaban = payouts_df["paytansyoumaban1"].isna().sum()
    null_tansyo_pay = payouts_df["paytansyopay1"].isna().sum()
    both_null = (payouts_df["paytansyoumaban1"].isna() & payouts_df["paytansyopay1"].isna()).sum()

    print("=== payouts NULL analysis ===")
    print(f"  total payout rows: {total_payouts:,}")
    print(f"  paytansyoumaban1 NULL: {null_tansyo_umaban:,} ({null_tansyo_umaban/total_payouts:.1%})")
    print(f"  paytansyopay1 NULL:    {null_tansyo_pay:,} ({null_tansyo_pay/total_payouts:.1%})")
    print(f"  both NULL:             {both_null:,} ({both_null/total_payouts:.1%})")
    print()

    # --- 4. entries のレース一覧 vs payouts のレース一覧 ---
    if "race_id" in entries_df.columns:
        entry_race_ids = set(entries_df["race_id"].dropna().astype(str).unique())
    else:
        print("ERROR: entries has no race_id column")
        return

    payout_race_ids = set(payouts_df["race_id"].dropna().astype(str).unique())

    only_in_entries = entry_race_ids - payout_race_ids
    only_in_payouts = payout_race_ids - entry_race_ids

    print("=== race_id coverage ===")
    print(f"  races in entries:  {len(entry_race_ids):,}")
    print(f"  races in payouts:  {len(payout_race_ids):,}")
    print(f"  only in entries (no payout): {len(only_in_entries):,}")
    print(f"  only in payouts (no entry):  {len(only_in_payouts):,}")

    # Sample races that are in entries but not in payouts
    if only_in_entries:
        sample = sorted(only_in_entries)[:20]
        print(f"  sample missing (entries only): {sample}")
    print()

    # --- 5. payouts に race_id があるが paytansyoumaban1 が NULL ---
    has_race_no_tansyo = payouts_df[
        payouts_df["paytansyoumaban1"].isna() | payouts_df["paytansyopay1"].isna()
    ]
    if not has_race_no_tansyo.empty:
        print(f"=== payouts with NULL tansyo ({len(has_race_no_tansyo)} races) ===")
        # 年度別集計
        if "race_id" in has_race_no_tansyo.columns:
            has_race_no_tansyo = has_race_no_tansyo.copy()
            has_race_no_tansyo["year"] = has_race_no_tansyo["race_id"].astype(str).str[:4]
            year_counts = has_race_no_tansyo.groupby("year").size()
            print("  by year:")
            for yr, cnt in year_counts.items():
                print(f"    {yr}: {cnt}")
        print()

    # --- 6. entries の1着馬と payouts の比較 ---
    # entries 側の1着馬を特定
    if "kakuteijyuni" in entries_df.columns:
        winners = entries_df[entries_df["kakuteijyuni"] == 1][["race_id", "umaban"]].copy()
        winners["race_id"] = winners["race_id"].astype(str)
        winners["umaban"] = winners["umaban"].astype(int)
        winner_keys = set(zip(winners["race_id"], winners["umaban"]))
        print(f"  total winner entries (kakuteijyuni==1): {len(winner_keys):,}")
    else:
        print("  WARNING: no kakuteijyuni in entries")
        winner_keys = set()

    # payouts 側の単勝払戻馬
    valid_tansyo = payouts_df.dropna(subset=["paytansyoumaban1", "paytansyopay1"]).copy()
    if not valid_tansyo.empty:
        valid_tansyo["umaban"] = valid_tansyo["paytansyoumaban1"].astype(int)
        valid_tansyo["race_id"] = valid_tansyo["race_id"].astype(str)
        payout_winner_keys = set(zip(valid_tansyo["race_id"], valid_tansyo["umaban"]))
        print(f"  payout tansyo entries: {len(payout_winner_keys):,}")
    else:
        payout_winner_keys = set()
        print("  payout tansyo entries: 0")

    # 1着馬なのに payouts にないケース（真の欠損）
    if winner_keys:
        real_missing = winner_keys - payout_winner_keys
        print(f"  WINNERS missing from payout map: {len(real_missing):,} ({len(real_missing)/len(winner_keys):.1%})")
        if real_missing:
            sample_missing = sorted(real_missing)[:20]
            print(f"  sample: {sample_missing}")
            # 年度別
            missing_years = {}
            for rid, _ in real_missing:
                yr = str(rid)[:4]
                missing_years[yr] = missing_years.get(yr, 0) + 1
            print("  by year:")
            for yr in sorted(missing_years):
                print(f"    {yr}: {missing_years[yr]}")
    print()

    # --- 7. 2020年10月3日 (ユーザー報告の race_id) の詳細 ---
    print("=== 2020-10-03 races (user-reported date) ===")
    oct_races = payouts_df[payouts_df["race_id"].astype(str).str.startswith("20201003")]
    print(f"  payout rows for 20201003: {len(oct_races)}")
    if not oct_races.empty:
        for _, row in oct_races.iterrows():
            rid = row["race_id"]
            uma1 = row.get("paytansyoumaban1", "N/A")
            pay1 = row.get("paytansyopay1", "N/A")
            print(f"    {rid}: umaban1={uma1}, pay1={pay1}")

    oct_entries = entries_df[entries_df["race_id"].astype(str).str.startswith("20201003")]
    print(f"  entry rows for 20201003: {len(oct_entries)}")
    if not oct_entries.empty:
        oct_winners = oct_entries[oct_entries["kakuteijyuni"] == 1]
        print(f"  winner rows for 20201003: {len(oct_winners)}")
        for _, row in oct_winners.head(5).iterrows():
            print(f"    race_id={row['race_id']}, umaban={row['umaban']}, kakuteijyuni={row['kakuteijyuni']}")
    print()

    # --- 8. ノイズ（非1着馬のベットによる警告）の推定 ---
    # バックテストではベット対象馬が全て win_payout_map を lookup する
    # 1着以外の馬は常に miss → ノイズ
    total_entries_with_finish = len(entries_df[entries_df["kakuteijyuni"].notna()])
    non_winners = len(entries_df[(entries_df["kakuteijyuni"].notna()) & (entries_df["kakuteijyuni"] != 1)])
    print("=== noise estimate ===")
    print(f"  entries with finish pos: {total_entries_with_finish:,}")
    print(f"  non-winner entries: {non_winners:,} ({non_winners/total_entries_with_finish:.1%})")
    print(f"  → Any WIN bet on non-winner triggers the warning (pure noise)")
    print()


if __name__ == "__main__":
    main()
