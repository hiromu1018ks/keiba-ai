import pandas as pd

# Load predictions
df = pd.read_csv('simulation_predictions.csv', low_memory=False)

# Ensure numeric
df['odds'] = pd.to_numeric(df['odds'], errors='coerce')
df['pred_prob'] = pd.to_numeric(df['pred_prob'], errors='coerce')
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

# Calculate EV
df['ev'] = df['pred_prob'] * df['odds']

# Filter by Best Threshold (3.0)
threshold = 3.0
bet_df = df[df['ev'] >= threshold].copy()

# Calculate Hit Rate
total_bets = len(bet_df)
hits = len(bet_df[bet_df['rank'] == 1])
hit_rate = (hits / total_bets * 100) if total_bets > 0 else 0

# Calculate Average Odds of Hits
avg_win_odds = bet_df[bet_df['rank'] == 1]['odds'].mean()

print(f"Total Bets: {total_bets}")
print(f"Hits: {hits}")
print(f"Hit Rate: {hit_rate:.2f}%")
print(f"Average Win Odds: {avg_win_odds:.2f}")

