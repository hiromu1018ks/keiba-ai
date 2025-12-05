import pandas as pd
import os

def extract_hits(input_file='simulation_predictions.csv', output_file='simulation_hits.csv'):
    """
    Extracts races where the simulation predicted a hit (EV >= 3.0 and Rank 1).
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)

    # Ensure numeric types for calculation
    cols_to_numeric = ['odds', 'pred_prob', 'rank']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate Expected Value (EV)
    df['ev'] = df['pred_prob'] * df['odds']

    # Filter for Hits: EV >= 3.0 and Rank == 1 (Win)
    threshold = 3.0
    hits_df = df[(df['ev'] >= threshold) & (df['rank'] == 1)].copy()

    # Sort by date (descending)
    if 'date' in hits_df.columns:
        # Handle Japanese date format like '2023年07月22日'
        hits_df['date'] = pd.to_datetime(hits_df['date'], format='%Y年%m月%d日', errors='coerce')
        hits_df = hits_df.sort_values('date', ascending=False)

    # Select columns to export/display
    # Adjust these columns based on what's available and useful
    target_columns = [
        'date', 'place', 'race_id', 'horse_name', 'jockey', 'trainer',
        'odds', 'pred_prob', 'ev', 'rank', 'race_laps', 'weather', 'surface', 'distance'
    ]
    
    # Filter only existing columns
    export_columns = [col for col in target_columns if col in hits_df.columns]
    
    export_df = hits_df[export_columns]

    print(f"Saving {len(export_df)} hits to {output_file}...")
    export_df.to_csv(output_file, index=False)

    print("\n--- Summary ---")
    print(f"Total Hits: {len(export_df)}")
    if len(export_df) > 0:
        print("\nTop 5 Recent Hits:")
        print(export_df.head().to_string(index=False))
        print(f"\nFull list saved to {output_file}")
    else:
        print("No hits found meeting the criteria.")

if __name__ == "__main__":
    extract_hits()
