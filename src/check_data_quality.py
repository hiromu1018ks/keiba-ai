import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('check_data')

def main():
    df = pd.read_csv('data/common/raw_data/results.csv')
    
    # Check popularity NaNs
    if 'popularity' in df.columns:
        # Some might be non-numeric '---'
        df['popularity_num'] = pd.to_numeric(df['popularity'], errors='coerce')
        nan_count = df['popularity_num'].isna().sum()
        total = len(df)
        zeros = (df['popularity_num'] == 0).sum()
        
        logger.info(f"Total rows: {total}")
        logger.info(f"Missing Popularity: {nan_count} ({nan_count/total:.2%})")
        logger.info(f"Zero Popularity: {zeros}")
        
    else:
        logger.error("No popularity column")

if __name__ == "__main__":
    main()
