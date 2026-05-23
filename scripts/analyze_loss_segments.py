"""Backtest loss segment analysis — 12 dimensions."""
import json
import sys
import io
import pandas as pd
import numpy as np
from pathlib import Path

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Load data ──────────────────────────────────────────────────────────────
DATA = Path(__file__).resolve().parent.parent / "data" / "backtest" / "multi_year_bet_history.json"
with open(DATA, encoding="utf-8") as f:
    records = json.load(f)

df = pd.DataFrame(records)
print(f"Total bets: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# ── Pre-processing ─────────────────────────────────────────────────────────
# result > 0 means the bet won; result is the actual payout amount
df["win"] = (df["result"] > 0).astype(int)
df["payout"] = df["result"]  # actual payout (already the monetary return)
df["profit"] = df["payout"] - df["stake"]
df["race_month"] = pd.to_datetime(df["race_date"]).dt.month

# Normalize regime
df["regime_short"] = df["regime"].str.replace("RegimeState.", "", regex=False)

# Grade
df["is_grade"] = ~df["grade_code"].isin(["X", "", None]) & df["grade_code"].notna()

# Distance band
def dist_band(k):
    if k <= 1400:
        return "Short(<=1400)"
    elif k <= 1800:
        return "Mile(1401-1800)"
    elif k <= 2400:
        return "Mid(1801-2400)"
    else:
        return "Long(2401+)"
df["dist_band"] = df["kyori"].apply(dist_band)

# Odds band
def odds_band(o):
    if o < 2:
        return "1-2"
    elif o < 3:
        return "2-3"
    elif o < 5:
        return "3-5"
    elif o < 7:
        return "5-7"
    elif o < 10:
        return "7-10"
    elif o < 15:
        return "10-15"
    elif o < 20:
        return "15-20"
    elif o < 30:
        return "20-30"
    elif o < 50:
        return "30-50"
    else:
        return "50+"
df["odds_band"] = df["odds"].apply(odds_band)

# Popularity band
def pop_band(p):
    if p <= 9:
        return str(p)
    else:
        return "9+"
df["pop_band"] = df["popularity"].apply(pop_band)

# EV band
def ev_band(e):
    if e < 1.2:
        return "EV 1.0-1.2"
    elif e < 1.5:
        return "EV 1.2-1.5"
    else:
        return "EV 1.5+"
df["ev_band"] = df["ev"].apply(ev_band)

# Popularity group for cross-tab
df["pop_group"] = df["popularity"].apply(lambda p: "Pop1-3" if p <= 3 else ("Pop4-6" if p <= 6 else "Pop7+"))

# Track condition label
tc_map = {1: "Good", 2: "Yielding", 3: "Soft", 4: "Heavy"}
df["track_label"] = df["track_condition_code"].map(tc_map).fillna("Unknown")

# Jyocd label
jyo_map = {
    "1.0": "Sapporo", "2.0": "Hakodate", "3.0": "Fukushima", "4.0": "Niigata",
    "5.0": "Tokyo", "6.0": "Nakayama", "7.0": "Chukyo", "8.0": "Kyoto",
    "9.0": "Hanshin", "10.0": "Kokura"
}
df["jyo_name"] = df["jyocd"].map(jyo_map).fillna(df["jyocd"].astype(str))


# ── Helper ─────────────────────────────────────────────────────────────────
def segment_summary(group_col, sort_by="total_profit", ascending=True):
    g = df.groupby(group_col, observed=True).agg(
        bets=("profit", "count"),
        win_rate=("win", "mean"),
        avg_odds=("odds", "mean"),
        total_stake=("stake", "sum"),
        total_payout=("payout", "sum"),
        total_profit=("profit", "sum"),
    )
    g["roi"] = (g["total_payout"] / g["total_stake"] - 1) * 100
    g = g.sort_values(sort_by, ascending=ascending)
    return g[["bets", "win_rate", "avg_odds", "roi", "total_profit"]]


def print_table(title, summary_df):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(summary_df.to_string(
        formatters={
            "win_rate": lambda x: f"{x:.1%}",
            "avg_odds": lambda x: f"{x:.1f}",
            "roi": lambda x: f"{x:+.2f}%",
            "total_profit": lambda x: f"{x:+,.0f}",
            "bets": lambda x: f"{x:,}",
        }
    ))


# ══════════════════════════════════════════════════════════════════════════
# Overall summary
# ══════════════════════════════════════════════════════════════════════════
total_stake = df["stake"].sum()
total_payout = df["payout"].sum()
total_profit = df["profit"].sum()
overall_roi = (total_payout / total_stake - 1) * 100

print(f"\n{'='*80}")
print(f"  OVERALL SUMMARY")
print(f"{'='*80}")
print(f"Total bets:     {len(df):,}")
print(f"Total stake:    {total_stake:,.0f}")
print(f"Total payout:   {total_payout:,.0f}")
print(f"Total P&L:      {total_profit:+,.0f}")
print(f"Overall ROI:    {overall_roi:+.2f}%")
print(f"Overall win%:   {df['win'].mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════
# 1. Surface (turf vs dirt)
# ══════════════════════════════════════════════════════════════════════════
print_table("1. SURFACE (Turf vs Dirt)", segment_summary("surface"))

# ══════════════════════════════════════════════════════════════════════════
# 2. Regime
# ══════════════════════════════════════════════════════════════════════════
print_table("2. REGIME (Conservative vs Aggressive)", segment_summary("regime_short"))

# ══════════════════════════════════════════════════════════════════════════
# 3. Odds band
# ══════════════════════════════════════════════════════════════════════════
odds_order = ["1-2", "2-3", "3-5", "5-7", "7-10", "10-15", "15-20", "20-30", "30-50", "50+"]
odds_summary = segment_summary("odds_band", sort_by="total_profit", ascending=True)
odds_summary = odds_summary.reindex([b for b in odds_order if b in odds_summary.index])
print_table("3. ODDS BAND", odds_summary)

# ══════════════════════════════════════════════════════════════════════════
# 4. Popularity
# ══════════════════════════════════════════════════════════════════════════
pop_order = [str(i) for i in range(1, 10)] + ["9+"]
pop_summary = segment_summary("pop_band", sort_by="total_profit", ascending=True)
pop_summary = pop_summary.reindex([b for b in pop_order if b in pop_summary.index])
print_table("4. POPULARITY", pop_summary)

# ══════════════════════════════════════════════════════════════════════════
# 5. Distance
# ══════════════════════════════════════════════════════════════════════════
dist_order = ["Short(<=1400)", "Mile(1401-1800)", "Mid(1801-2400)", "Long(2401+)"]
dist_summary = segment_summary("dist_band")
dist_summary = dist_summary.reindex([b for b in dist_order if b in dist_summary.index])
print_table("5. DISTANCE", dist_summary)

# ══════════════════════════════════════════════════════════════════════════
# 6. Grade
# ══════════════════════════════════════════════════════════════════════════
df["grade_label"] = df["is_grade"].map({True: "Graded", False: "Regular"})
print_table("6. GRADE (Graded vs Regular)", segment_summary("grade_label"))

# ══════════════════════════════════════════════════════════════════════════
# 7. Month
# ══════════════════════════════════════════════════════════════════════════
month_summary = segment_summary("race_month")
month_summary.index = month_summary.index.map(lambda m: f"{m}月")
print_table("7. MONTH", month_summary)

# ══════════════════════════════════════════════════════════════════════════
# 8. Jyo (racecourse)
# ══════════════════════════════════════════════════════════════════════════
print_table("8. RACECOURSE", segment_summary("jyo_name", sort_by="total_profit", ascending=True))

# ══════════════════════════════════════════════════════════════════════════
# 9. Track condition
# ══════════════════════════════════════════════════════════════════════════
tc_order = ["Good", "Yielding", "Soft", "Heavy"]
tc_summary = segment_summary("track_label")
tc_summary = tc_summary.reindex([b for b in tc_order if b in tc_summary.index])
print_table("9. TRACK CONDITION", tc_summary)

# ══════════════════════════════════════════════════════════════════════════
# 10. EV x Surface cross-tab
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"  10. EV x SURFACE CROSS-TAB")
print(f"{'='*80}")

ev_order = ["EV 1.0-1.2", "EV 1.2-1.5", "EV 1.5+"]
cross_ev_surface = df.groupby(["ev_band", "surface"], observed=True).agg(
    bets=("profit", "count"),
    total_stake=("stake", "sum"),
    total_payout=("payout", "sum"),
    total_profit=("profit", "sum"),
    win_rate=("win", "mean"),
    avg_odds=("odds", "mean"),
)
cross_ev_surface["roi"] = (cross_ev_surface["total_payout"] / cross_ev_surface["total_stake"] - 1) * 100

roi_pivot = cross_ev_surface["roi"].unstack("surface")
profit_pivot = cross_ev_surface["total_profit"].unstack("surface")
bets_pivot = cross_ev_surface["bets"].unstack("surface")
winrate_pivot = cross_ev_surface["win_rate"].unstack("surface")

roi_pivot = roi_pivot.reindex(ev_order)
print("\n--- ROI (%) ---")
print(roi_pivot.to_string(formatters={c: lambda x: f"{x:+.2f}%" for c in roi_pivot.columns}))
print("\n--- Profit (yen) ---")
profit_reindexed = profit_pivot.reindex(ev_order)
print(profit_reindexed.to_string(formatters={c: lambda x: f"{x:+,.0f}" for c in profit_reindexed.columns}))
print("\n--- Bet count ---")
bets_reindexed = bets_pivot.reindex(ev_order)
print(bets_reindexed.to_string(formatters={c: lambda x: f"{x:,}" for c in bets_reindexed.columns}))
print("\n--- Win rate ---")
wr_reindexed = winrate_pivot.reindex(ev_order)
print(wr_reindexed.to_string(formatters={c: lambda x: f"{x:.1%}" for c in wr_reindexed.columns}))

# ══════════════════════════════════════════════════════════════════════════
# 11. Popularity x Odds band cross-tab
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"  11. POPULARITY GROUP x ODDS BAND CROSS-TAB")
print(f"{'='*80}")

cross_pop_odds = df.groupby(["pop_group", "odds_band"], observed=True).agg(
    bets=("profit", "count"),
    total_stake=("stake", "sum"),
    total_payout=("payout", "sum"),
    total_profit=("profit", "sum"),
    win_rate=("win", "mean"),
    avg_odds=("odds", "mean"),
)
cross_pop_odds["roi"] = (cross_pop_odds["total_payout"] / cross_pop_odds["total_stake"] - 1) * 100

pop_groups = ["Pop1-3", "Pop4-6", "Pop7+"]
for pg in pop_groups:
    print(f"\n--- {pg} ---")
    if pg in cross_pop_odds.index.get_level_values(0):
        sub = cross_pop_odds.loc[pg]
        sub = sub.reindex([b for b in odds_order if b in sub.index])
        display_df = sub[["bets", "win_rate", "avg_odds", "roi", "total_profit"]].copy()
        print(display_df.to_string(
            formatters={
                "bets": lambda x: f"{x:,}",
                "win_rate": lambda x: f"{x:.1%}",
                "avg_odds": lambda x: f"{x:.1f}",
                "roi": lambda x: f"{x:+.2f}%",
                "total_profit": lambda x: f"{x:+,.0f}",
            }
        ))
    else:
        print("  (no data)")

# ══════════════════════════════════════════════════════════════════════════
# 12. Top 5 profit sources and Top 5 loss sources
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"  12. TOP PROFIT / TOP LOSS SEGMENTS")
print(f"{'='*80}")

all_segments = []

for col, label in [
    ("surface", "Surface"),
    ("regime_short", "Regime"),
    ("odds_band", "Odds"),
    ("pop_band", "Popularity"),
    ("dist_band", "Distance"),
    ("grade_label", "Grade"),
    ("jyo_name", "Racecourse"),
    ("track_label", "Track Cond"),
]:
    g = df.groupby(col, observed=True).agg(
        bets=("profit", "count"),
        win_rate=("win", "mean"),
        avg_odds=("odds", "mean"),
        total_stake=("stake", "sum"),
        total_payout=("payout", "sum"),
        total_profit=("profit", "sum"),
    )
    g["roi"] = (g["total_payout"] / g["total_stake"] - 1) * 100
    for idx, row in g.iterrows():
        all_segments.append({
            "dimension": label,
            "segment": str(idx),
            "bets": int(row["bets"]),
            "win_rate": row["win_rate"],
            "avg_odds": row["avg_odds"],
            "roi": row["roi"],
            "total_profit": row["total_profit"],
        })

# Add month
g = df.groupby("race_month", observed=True).agg(
    bets=("profit", "count"),
    win_rate=("win", "mean"),
    avg_odds=("odds", "mean"),
    total_stake=("stake", "sum"),
    total_payout=("payout", "sum"),
    total_profit=("profit", "sum"),
)
g["roi"] = (g["total_payout"] / g["total_stake"] - 1) * 100
for idx, row in g.iterrows():
    all_segments.append({
        "dimension": "Month",
        "segment": f"{idx}月",
        "bets": int(row["bets"]),
        "win_rate": row["win_rate"],
        "avg_odds": row["avg_odds"],
        "roi": row["roi"],
        "total_profit": row["total_profit"],
    })

seg_df = pd.DataFrame(all_segments)
seg_df["warning"] = seg_df["bets"].apply(lambda x: " ** LOW-N" if x < 100 else "")

# Top 5 profit
print("\n--- TOP 5 PROFIT SEGMENTS ---")
top5_profit = seg_df.nlargest(5, "total_profit")
print(top5_profit[["dimension", "segment", "bets", "roi", "total_profit", "warning"]].to_string(
    index=False,
    formatters={
        "bets": lambda x: f"{x:,}",
        "roi": lambda x: f"{x:+.2f}%",
        "total_profit": lambda x: f"{x:+,.0f}",
    }
))

# Top 5 loss
print("\n--- TOP 5 LOSS SEGMENTS ---")
top5_loss = seg_df.nsmallest(5, "total_profit")
print(top5_loss[["dimension", "segment", "bets", "roi", "total_profit", "warning"]].to_string(
    index=False,
    formatters={
        "bets": lambda x: f"{x:,}",
        "roi": lambda x: f"{x:+.2f}%",
        "total_profit": lambda x: f"{x:+,.0f}",
    }
))

# Full table sorted by profit
print("\n--- ALL SEGMENTS (sorted by profit) ---")
seg_sorted = seg_df.sort_values("total_profit", ascending=False)
print(seg_sorted[["dimension", "segment", "bets", "roi", "total_profit", "warning"]].to_string(
    index=False,
    formatters={
        "bets": lambda x: f"{x:,}",
        "roi": lambda x: f"{x:+.2f}%",
        "total_profit": lambda x: f"{x:+,.0f}",
    }
))

# ══════════════════════════════════════════════════════════════════════════
# KEY FINDINGS
# ══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print(f"  KEY FINDINGS")
print(f"{'='*80}")

# Best surface
best_surf = df.groupby("surface").agg(profit=("profit", "sum"), roi_pct=("profit", lambda x: x.sum()/df[df["surface"]==x.name]["stake"].sum()*100))
for s in df["surface"].unique():
    sub = df[df["surface"]==s]
    r = (sub["payout"].sum() / sub["stake"].sum() - 1) * 100
    print(f"  Surface {s}: ROI={r:+.2f}%, Profit={sub['profit'].sum():+,.0f}, Bets={len(sub):,}, Win%={sub['win'].mean():.1%}")

print()
for rg in df["regime_short"].unique():
    sub = df[df["regime_short"]==rg]
    r = (sub["payout"].sum() / sub["stake"].sum() - 1) * 100
    print(f"  Regime {rg}: ROI={r:+.2f}%, Profit={sub['profit'].sum():+,.0f}, Bets={len(sub):,}, Win%={sub['win'].mean():.1%}")

print()
# Worst losing segments
print("  --- Worst segments (by absolute loss) ---")
worst = seg_df.nsmallest(5, "total_profit")
for _, row in worst.iterrows():
    flag = " ** LOW-N <100 bets" if row["bets"] < 100 else ""
    print(f"    {row['dimension']:12s} | {row['segment']:20s} | ROI={row['roi']:+.2f}% | Profit={row['total_profit']:+,.0f} | Bets={row['bets']:,}{flag}")

print()
print("  --- Best segments (by absolute profit) ---")
best = seg_df.nlargest(5, "total_profit")
for _, row in best.iterrows():
    flag = " ** LOW-N <100 bets" if row["bets"] < 100 else ""
    print(f"    {row['dimension']:12s} | {row['segment']:20s} | ROI={row['roi']:+.2f}% | Profit={row['total_profit']:+,.0f} | Bets={row['bets']:,}{flag}")
