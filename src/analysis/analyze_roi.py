import pandas as pd
import numpy as np

# Load data
print("Loading data...")
df = pd.read_csv('simulation_predictions.csv')
df['ev'] = df['pred_prob'] * df['odds']

print(f'\nTotal records: {len(df)}')
print(f'Total bets (EV > 2.0): {len(df[df["ev"] > 2.0])}')

# 1. Odds Bin Analysis
bins = [1, 2, 5, 10, 20, 50, 100, 1000]
labels = ['1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '100+']
df['odds_bin'] = pd.cut(df['odds'], bins=bins, labels=labels, right=False)

print('\n=== 1. 全体: オッズ帯別の予測傾向 (Bias check) ===')
print(f'{"Bin":<10} {"Count":<8} {"AvgProb":<10} {"ActualWin":<10} {"Bias":<10} {"AvgEV":<10}')
for label in labels:
    sub = df[df['odds_bin'] == label]
    if len(sub) == 0: continue
    
    count = len(sub)
    avg_prob = sub['pred_prob'].mean()
    actual_win = len(sub[sub["rank"] == 1]) / count
    # Bias = 予測勝率 / 実勝率 (1.0より大きいと過大評価)
    bias = avg_prob / actual_win if actual_win > 0 else 0
    ev_mean = sub['ev'].mean()
    
    print(f'{label:<10} {count:<8} {avg_prob:.4f}     {actual_win:.4f}     {bias:.2f}       {ev_mean:.2f}')

# 2. High EV Bias Analysis (Where EV > 2.0)
print('\n=== 2. ベット対象 (EV > 2.0): オッズ帯別の回収率 ===')
high_ev = df[df['ev'] > 2.0].copy()
high_ev['odds_bin'] = pd.cut(high_ev['odds'], bins=bins, labels=labels, right=False)

print(f'{"Bin":<10} {"Bets":<8} {"Hits":<6} {"WinRate":<8} {"Return":<8} {"ROI":<8}')
for label in labels:
    sub = high_ev[high_ev['odds_bin'] == label]
    if len(sub) == 0: continue
    
    bets = len(sub)
    hits = len(sub[sub["rank"] == 1])
    win_rate = hits / bets
    ret = sub[sub["rank"] == 1]['odds'].sum()
    bet_amt = bets # assuming 1 unit
    roi = (ret / bet_amt * 100)
    
    print(f'{label:<10} {bets:<8} {hits:<6} {win_rate:.4f}   {int(ret):<8} {roi:.1f}%')
