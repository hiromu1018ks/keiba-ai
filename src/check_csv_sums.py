import pandas as pd

def main():
    try:
        df = pd.read_csv('simulation_predictions.csv')
        # Check columns
        if 'race_id' not in df.columns or 'pred_prob' not in df.columns:
            print("Columns missing.")
            print(df.columns)
            return

        # Group by race_id and sum pred_prob
        sums = df.groupby('race_id')['pred_prob'].sum()
        
        print(f"Mean Sum: {sums.mean():.4f}")
        print(f"Median Sum: {sums.median():.4f}")
        print(f"Min Sum: {sums.min():.4f}")
        print(f"Max Sum: {sums.max():.4f}")
        
        # Show a few examples
        print("\nExamples:")
        print(sums.head())
        
    except FileNotFoundError:
        print("File not found.")

if __name__ == "__main__":
    main()
