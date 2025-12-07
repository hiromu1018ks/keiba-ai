import pandas as pd

def main():
    try:
        df = pd.read_csv('simulation_predictions.csv')
        # EV calc
        # Note: Simulation used raw probs (sum ~0.84)
        df['ev'] = df['pred_prob'] * df['odds']
        
        # Filter: EV > 1.5, Odds > 100 filter?
        # Simulation strategy.py: odds > 100 continue, prob < 0.05 continue
        
        df_bets = df[
            (df['odds'] <= 100.0) & 
            (df['pred_prob'] >= 0.05) & 
            (df['ev'] > 1.5)
        ]
        
        total_races = df['race_id'].nunique()
        total_bets = len(df_bets)
        
        print(f"Total Races: {total_races}")
        print(f"Total Bets: {total_bets}")
        print(f"Bets per Race: {total_bets / total_races:.4f}")
        
        # Let's see what threshold matches ~1 bet per race (typical for 'Strict' strategy)
        # or matches the user's expected volume.
        
    except FileNotFoundError:
        print("File not found.")

if __name__ == "__main__":
    main()
