#!/usr/bin/env python3
"""
分布シフト & リーク疑い特徴量 検査スクリプト
=============================================
A. 共通列の分布比較 (KS統計量, Cohen's d)
B. リーク疑い特徴量の検査
C. OOF予測品質
D. BT専用列 (546 - 510 = 36列)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────
print("=" * 100)
print("  競馬AI 分布シフト & リーク検査 レポート")
print("=" * 100)

print("\n[0] データ読み込み中...")
train = pd.read_parquet(PROJECT_ROOT / "data/features/horse_features.parquet")
bt_feat = pd.read_parquet(PROJECT_ROOT / "data/backtest/bt_2024_horse_features.parquet")
oof = pd.read_parquet(PROJECT_ROOT / "data/oof/oof_predictions.parquet")
bt_pred = pd.read_parquet(PROJECT_ROOT / "data/backtest/predictions/2024.parquet")

print(f"  train features:       {train.shape}")
print(f"  backtest features:    {bt_feat.shape}")
print(f"  OOF predictions:      {oof.shape}")
print(f"  BT predictions:       {bt_pred.shape}")

# ──────────────────────────────────────────────
# A. 共通列の分布比較
# ──────────────────────────────────────────────
print("\n" + "=" * 100)
print("  A. 共通列の分布比較 (train vs backtest)")
print("=" * 100)

common_cols = sorted(set(train.columns) & set(bt_feat.columns))
train_only = sorted(set(train.columns) - set(bt_feat.columns))
bt_only = sorted(set(bt_feat.columns) - set(train.columns))

print(f"\n  共通列数: {len(common_cols)}")
print(f"  Train専用列数: {len(train_only)}")
print(f"  BT専用列数: {len(bt_only)}")

# 数値列のみを対象
numeric_common = [c for c in common_cols if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(bt_feat[c])]
print(f"  数値型共通列数: {len(numeric_common)}")

# KS統計量 & Cohen's d を計算
print("\n  計算中 (KS test + Cohen's d)...")
results = []
for col in numeric_common:
    tr = train[col].dropna()
    bt = bt_feat[col].dropna()
    if len(tr) < 10 or len(bt) < 10:
        continue
    # サンプリング (高速化)
    tr_sample = tr.sample(n=min(10000, len(tr)), random_state=42).values
    bt_sample = bt.sample(n=min(10000, len(bt)), random_state=42).values

    ks_stat, ks_pval = stats.ks_2samp(tr_sample, bt_sample)

    # Cohen's d
    mean_diff = tr.mean() - bt.mean()
    pooled_std = np.sqrt((tr.std()**2 + bt.std()**2) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 1e-12 else 0.0

    null_train = train[col].isna().mean() * 100
    null_bt = bt_feat[col].isna().mean() * 100

    results.append({
        "column": col,
        "train_mean": tr.mean(),
        "train_std": tr.std(),
        "train_null%": null_train,
        "bt_mean": bt.mean(),
        "bt_std": bt.std(),
        "bt_null%": null_bt,
        "null_diff_pp": abs(null_train - null_bt),
        "ks_stat": ks_stat,
        "ks_pval": ks_pval,
        "cohens_d": abs(cohens_d),
        "cohens_d_signed": cohens_d,
    })

df_ks = pd.DataFrame(results).sort_values("ks_stat", ascending=False)

# Top 20 分布シフト
print("\n  ─── Top 20 分布シフト (KS統計量順) ───")
top20 = df_ks.head(20)
for i, row in top20.iterrows():
    shift_level = "!!!CRITICAL!!!" if row["ks_stat"] > 0.5 else ("!!HIGH!!" if row["ks_stat"] > 0.3 else "MODERATE")
    print(f"  {row['column']:45s}  KS={row['ks_stat']:.4f}  d={row['cohens_d']:.3f}  "
          f"train_null={row['train_null%']:.1f}%  bt_null={row['bt_null%']:.1f}%  [{shift_level}]")

# KS > 0.1 の統計
high_ks = df_ks[df_ks["ks_stat"] > 0.1]
print(f"\n  KS統計量 > 0.1 の列数: {len(high_ks)} / {len(numeric_common)}")
critical_ks = df_ks[df_ks["ks_stat"] > 0.5]
print(f"  KS統計量 > 0.5 (CRITICAL): {len(critical_ks)} 列")
very_high_ks = df_ks[(df_ks["ks_stat"] > 0.3) & (df_ks["ks_stat"] <= 0.5)]
print(f"  KS統計量 > 0.3 (HIGH): {len(very_high_ks)} 列")

# null率が大きく異なる列
print("\n  ─── Null率差 > 10pp の列 ───")
null_shift = df_ks[df_ks["null_diff_pp"] > 10].sort_values("null_diff_pp", ascending=False)
if len(null_shift) > 0:
    for i, row in null_shift.iterrows():
        print(f"  {row['column']:45s}  train_null={row['train_null%']:.1f}%  bt_null={row['bt_null%']:.1f}%  "
              f"差={row['null_diff_pp']:.1f}pp")
else:
    print("  (該当なし)")

# ──────────────────────────────────────────────
# B. リーク疑い特徴量の検査
# ──────────────────────────────────────────────
print("\n" + "=" * 100)
print("  B. リーク疑い特徴量の検査")
print("=" * 100)

# リーク候補パターン
leak_patterns = [
    "confirmed_odds", "tanodds", "fukuoddslow", "tanninki",
    "kakuteijyuni", "chakusacdp", "chakusacdpp",
    "honsyokin", "fukasyokin", "jyuni",
    "umakigo", "tyakujun", "cyakujun",
    "kakutei", "haityou", "nyuusen",
    "rank", "prize", "syokin",
]

# 予測対象 (kakuteijyuni = 確定順位) を使って相関をチェック
target_col = None
for c in ["kakuteijyuni", "kakuteijyuni_1"]:
    if c in train.columns:
        target_col = c
        break

if target_col:
    target = train[target_col]
    is_win = (target == 1).astype(int)
else:
    is_win = None
    print("  [WARNING] kakuteijyuni が train に見つかりません")

# 候補列を検索
leak_candidates = []
for pat in leak_patterns:
    matches = [c for c in train.columns if pat.lower() in c.lower()]
    leak_candidates.extend(matches)

leak_candidates = sorted(set(leak_candidates))
print(f"\n  リーク候補パターンにマッチする列数: {len(leak_candidates)}")

print("\n  ─── リーク候補列の詳細検査 ───")
leak_results = []
for col in leak_candidates:
    in_train = col in train.columns
    in_bt = col in bt_feat.columns

    # Train統計
    if in_train:
        tr_col = train[col]
        null_rate = tr_col.isna().mean() * 100
        nunique = tr_col.nunique()
        dtype = str(tr_col.dtype)

        # 相関 (数値列のみ)
        corr_with_win = np.nan
        if is_win is not None and pd.api.types.is_numeric_dtype(tr_col):
            valid = ~(tr_col.isna() | is_win.isna())
            if valid.sum() > 100:
                corr_with_win = np.corrcoef(tr_col[valid], is_win[valid])[0, 1]

        # 予測時のnull率 (BT)
        bt_null = np.nan
        if in_bt:
            bt_null = bt_feat[col].isna().mean() * 100
    else:
        null_rate = nunique = dtype = corr_with_win = bt_null = "N/A"

    leak_risk = "LOW"
    risk_reasons = []
    if isinstance(corr_with_win, float) and abs(corr_with_win) > 0.3:
        leak_risk = "HIGH"
        risk_reasons.append(f"高相関({corr_with_win:.3f})")
    if isinstance(bt_null, float) and bt_null < 5 and in_bt:
        risk_reasons.append("BT時も利用可能")
    if any(kw in col.lower() for kw in ["kakutei", "jyuni", "tyakujun", "rank", "chakusa"]):
        risk_reasons.append("確定後情報の可能性")
        if leak_risk != "HIGH":
            leak_risk = "MEDIUM"

    leak_results.append({
        "column": col,
        "in_train": in_train,
        "in_bt": in_bt,
        "dtype": dtype,
        "null_rate%": null_rate,
        "bt_null%": bt_null,
        "nunique": nunique,
        "corr_with_win": corr_with_win,
        "leak_risk": leak_risk,
        "reasons": "; ".join(risk_reasons) if risk_reasons else "-",
    })

df_leak = pd.DataFrame(leak_results)

# リスクレベル順にソート
risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "N/A": 3}
df_leak["_sort"] = df_leak["leak_risk"].map(risk_order)
df_leak = df_leak.sort_values("_sort")

print(f"\n  {'列名':45s} {'Train':>5s} {'BT':>5s} {'Null%':>7s} {'BT_Null%':>8s} {'Corr':>7s} {'Risk':>7s} 理由")
print("  " + "-" * 120)
for _, row in df_leak.iterrows():
    corr_str = f"{row['corr_with_win']:.3f}" if isinstance(row['corr_with_win'], (int, float)) else str(row['corr_with_win'])
    null_str = f"{row['null_rate%']:.1f}" if isinstance(row['null_rate%'], (int, float)) else str(row['null_rate%'])
    bt_null_str = f"{row['bt_null%']:.1f}" if isinstance(row['bt_null%'], (int, float)) else str(row['bt_null%'])
    print(f"  {row['column']:45s} {'Y' if row['in_train'] else 'N':>5s} {'Y' if row['in_bt'] else 'N':>5s} "
          f"{null_str:>7s} {bt_null_str:>8s} {corr_str:>7s} {row['leak_risk']:>7s} {row['reasons']}")

# 追加: 確定オッズ系の列が予測前に利用可能か
print("\n  ─── 確定オッズ列の値分布確認 ───")
odds_cols = [c for c in train.columns if any(kw in c.lower() for kw in ["tanodds", "fukuodds", "odds"])]
if odds_cols:
    for col in odds_cols[:10]:
        tr_col = train[col]
        print(f"  {col}: mean={tr_col.mean():.2f}, median={tr_col.median():.2f}, "
              f"null={tr_col.isna().mean()*100:.1f}%, nunique={tr_col.nunique()}")

# ──────────────────────────────────────────────
# C. OOF予測品質
# ──────────────────────────────────────────────
print("\n" + "=" * 100)
print("  C. OOF予測品質")
print("=" * 100)

# OOF列一覧
pred_cols = [c for c in oof.columns if any(kw in c.lower() for kw in ["pred", "corrected", "ev_", "calibrated", "label"])]
print(f"\n  OOF 予測関連列: {pred_cols}")

# 目的変数
target_oof = None
for c in ["kakuteijyuni", "label", "is_win", "win_label"]:
    if c in oof.columns:
        target_oof = c
        break

if target_oof is None:
    # kakuteijyuni を探す
    jyuni_cols = [c for c in oof.columns if "jyuni" in c.lower()]
    if jyuni_cols:
        target_oof = jyuni_cols[0]
    else:
        print("  [WARNING] 目的変数列が見つかりません。利用可能列:")
        print(f"  {sorted(oof.columns[:50].tolist())}...")

if target_oof:
    print(f"  目的変数: {target_oof}")
    y_true = (oof[target_oof] == 1).astype(int) if oof[target_oof].dtype in [np.int64, np.float64] else oof[target_oof]
    print(f"  1着率: {y_true.mean():.4f} ({y_true.sum()}/{len(y_true)})")

    # p_win_pred の AUC
    from sklearn.metrics import roc_auc_score, brier_score_loss

    pcols_to_check = [c for c in ["p_win_pred", "p_win_corrected", "p_place_pred", "p_place_corrected"] if c in oof.columns]
    print(f"\n  ─── 予測品質メトリクス ───")
    print(f"  {'列名':30s} {'AUC':>8s} {'Brier':>8s} {'Mean':>8s} {'Min':>8s} {'Max':>8s} {'Std':>8s}")
    print("  " + "-" * 110)

    for pcol in pcols_to_check:
        pred = oof[pcol]
        valid = ~(pred.isna() | y_true.isna())
        if valid.sum() > 100:
            auc = roc_auc_score(y_true[valid], pred[valid])
            brier = brier_score_loss(y_true[valid], pred[valid])
            print(f"  {pcol:30s} {auc:>8.4f} {brier:>8.4f} {pred.mean():>8.4f} {pred.min():>8.4f} {pred.max():>8.4f} {pred.std():>8.4f}")

    # キャリブレーション
    print("\n  ─── キャリブレーション (p_win_pred) ───")
    pcol = "p_win_pred"
    if pcol in oof.columns:
        pred = oof[pcol]
        valid = ~(pred.isna() | y_true.isna())
        pred_v = pred[valid]
        y_v = y_true[valid]

        bins = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        print(f"  {'Bin':20s} {'Count':>7s} {'MeanPred':>10s} {'ActualRate':>11s} {'Diff':>8s}")
        print("  " + "-" * 60)
        for i in range(len(bins) - 1):
            mask = (pred_v >= bins[i]) & (pred_v < bins[i+1])
            count = mask.sum()
            if count > 0:
                mean_pred = pred_v[mask].mean()
                actual_rate = y_v[mask].mean()
                diff = actual_rate - mean_pred
                print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}){'':<8s} {count:>7d} {mean_pred:>10.4f} {actual_rate:>11.4f} {diff:>+8.4f}")

    # EV とオッズの相関
    print("\n  ─── EV予測とオッズの相関 ───")
    ev_col = "ev_win" if "ev_win" in oof.columns else None
    odds_col = "tanodds" if "tanodds" in oof.columns else None
    if ev_col and odds_col:
        valid = ~(oof[ev_col].isna() | oof[odds_col].isna())
        if valid.sum() > 100:
            corr = np.corrcoef(oof.loc[valid, ev_col], 1.0 / oof.loc[valid, odds_col])[0, 1]
            corr_odds = np.corrcoef(oof.loc[valid, ev_col], oof.loc[valid, odds_col])[0, 1]
            print(f"  ev_win vs 1/tanodds 相関: {corr:.4f}")
            print(f"  ev_win vs tanodds 相関:   {corr_odds:.4f}")

    # OOF vs BT 予測分布比較
    print("\n  ─── OOF vs BT 予測分布比較 (p_win_pred) ───")
    pcol = "p_win_pred"
    oof_has = pcol in oof.columns
    bt_has = pcol in bt_pred.columns
    if oof_has and bt_has:
        oof_vals = oof[pcol].dropna().sample(n=min(10000, len(oof[pcol].dropna())), random_state=42).values
        bt_vals = bt_pred[pcol].dropna().sample(n=min(10000, len(bt_pred[pcol].dropna())), random_state=42).values
        ks_stat, ks_pval = stats.ks_2samp(oof_vals, bt_vals)
        print(f"  OOF p_win_pred: mean={oof[pcol].mean():.4f}, std={oof[pcol].std():.4f}")
        print(f"  BT  p_win_pred: mean={bt_pred[pcol].mean():.4f}, std={bt_pred[pcol].std():.4f}")
        print(f"  KS test: stat={ks_stat:.4f}, p-value={ks_pval:.2e}")
        if ks_stat > 0.1:
            print(f"  [WARNING] OOFとBTの予測分布に有意なシフトあり (KS={ks_stat:.4f})")
        else:
            print(f"  [OK] OOFとBTの予測分布は概ね一致")

# ──────────────────────────────────────────────
# D. BT専用列の調査 (546 - 510 = 36列)
# ──────────────────────────────────────────────
print("\n" + "=" * 100)
print("  D. BT専用列の調査 (bt_2024_horse_features にのみ存在)")
print("=" * 100)

print(f"\n  BT専用列数: {len(bt_only)}")
print(f"\n  {'列名':50s} {'Dtype':>10s} {'Null%':>7s} {'NUnique':>8s} {'Mean/Sample':>20s}")
print("  " + "-" * 100)

for col in sorted(bt_only):
    bt_col = bt_feat[col]
    dtype = str(bt_col.dtype)
    null_pct = bt_col.isna().mean() * 100
    nunique = bt_col.nunique()

    if pd.api.types.is_numeric_dtype(bt_col):
        sample_info = f"mean={bt_col.mean():.4f}, std={bt_col.std():.4f}"
    else:
        sample_info = f"samples={bt_col.dropna().head(3).tolist()}"

    # リーク判定
    leak_flag = ""
    col_lower = col.lower()
    if any(kw in col_lower for kw in ["jyuni", "kakutei", "chakusa", "tyakujun", "rank"]):
        leak_flag = " *** LEAK SUSPECT ***"
    elif any(kw in col_lower for kw in ["odds", "tan", "fuku", "ninki"]):
        leak_flag = " * ODDS-RELATED *"
    elif any(kw in col_lower for kw in ["syokin", "prize", "haityou"]):
        leak_flag = " * PRIZE-RELATED *"

    print(f"  {col:50s} {dtype:>10s} {null_pct:>6.1f}% {nunique:>8d} {sample_info:>20s}{leak_flag}")

# ──────────────────────────────────────────────
# E. サマリー
# ──────────────────────────────────────────────
print("\n" + "=" * 100)
print("  E. サマリー")
print("=" * 100)

print(f"""
  [分布シフト]
    - KS > 0.5 (CRITICAL): {len(critical_ks)} 列
    - KS > 0.3 (HIGH):     {len(very_high_ks)} 列
    - KS > 0.1 (全体):     {len(high_ks)} 列 / {len(numeric_common)} 数値共通列

  [リーク検査]
    - HIGH リスク: {len(df_leak[df_leak['leak_risk'] == 'HIGH'])} 列
    - MEDIUM リスク: {len(df_leak[df_leak['leak_risk'] == 'MEDIUM'])} 列

  [BT専用列]
    - {len(bt_only)} 列がBTにのみ存在 (リーク要注意)

  [推奨アクション]
    1. KS > 0.5 の特徴量は学習に使わない または 分布補正を検討
    2. HIGH リスク列は特徴量から除外
    3. BT専用列のうち「確定後情報」は学習に含めないよう確認
    4. OOF AUCが極端に高い場合 (>.95) はリークの強い兆候
""")

print("=" * 100)
print("  分析完了")
print("=" * 100)
