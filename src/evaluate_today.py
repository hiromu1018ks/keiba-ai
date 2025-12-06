import pandas as pd
import argparse
import os
import datetime
from src.data.scraper_playwright import PlaywrightScraper
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Evaluate daily predictions against actual results.')
    parser.add_argument('--date', type=str, help='Date YYYYMMDD (default: today)')
    args = parser.parse_args()

    if args.date:
        target_date_str = args.date
    else:
        target_date_str = datetime.datetime.now().strftime("%Y%m%d")

    # 1. Load Predictions from CSV
    csv_path = f'output/predictions_{target_date_str}.csv'
    if not os.path.exists(csv_path):
        logger.error(f"Prediction CSV not found: {csv_path}")
        logger.info("Please run predict_today.py first.")
        return

    logger.info(f"Loading predictions from {csv_path}...")
    df_preds = pd.read_csv(csv_path)
    
    # Ensure race_id is string
    df_preds['race_id'] = df_preds['race_id'].astype(str)

    # Filter for 'Buy' or 'Strong Buy'
    # rec_class in ['buy', 'strong-buy'] means we bet on them.
    # Logic in predict_today: matches simulation logic exactly.
    df_bets = df_preds[df_preds['rec_class'].isin(['buy', 'strong-buy'])].copy()

    if df_bets.empty:
        logger.warning("No bets placed for this date.")
        return

    race_ids = df_bets['race_id'].unique().tolist()
    logger.info(f"Found {len(df_bets)} bets across {len(race_ids)} races.")

    # 2. Scrape Results
    logger.info("Scraping race results...")
    scraper = PlaywrightScraper()
    results = scraper.scrape_race_results(race_ids)

    if not results:
        logger.error("Failed to scrape any results.")
        return

    # 3. Calculate Metrics
    total_bets = 0
    total_cost = 0
    total_return = 0
    hits = 0

    print(f"\n{'='*20} Evaluation Result ({target_date_str}) {'='*20}")
    print(f"{'RaceID':<14} {'Horse':<20} {'Bet':<5} {'Result':<8} {'Return'}")
    print("-" * 70)

    # Process each bet
    # We aggregate bets per race to handle multiple bets per race correctly if needed (Summing costs)
    # But sim logic is flat: each row is a bet ticket (Tansho 100yen)
    
    bet_unit = 100

    for idx, row in df_bets.iterrows():
        rid = str(row['race_id'])
        h_num = int(row['horse_num'])
        h_name = row['horse_name']
        
        res = results.get(rid)
        
        result_status = "WAIT"
        return_amount = 0
        
        if res:
            win_horse = res['win_horse_num']
            win_payout = res['win_payout']
            
            if h_num == win_horse:
                result_status = "WIN"
                return_amount = win_payout
                hits += 1
            else:
                result_status = "LOSE"
                return_amount = 0
        else:
             result_status = "N/A" # Result not found (maybe cancelled or scraper error)

        total_bets += 1
        total_cost += bet_unit
        total_return += return_amount
        
        print(f"{rid:<14} {h_name:<20} {bet_unit:<5} {result_status:<8} {return_amount}")

    # 4. Summary
    recovery_rate = (total_return / total_cost * 100) if total_cost > 0 else 0
    hit_rate = (hits / total_bets * 100) if total_bets > 0 else 0

    print(f"\n{'='*20} Summary {'='*20}")
    print(f"Total Bets:     {total_bets}")
    print(f"Hit Count:      {hits}")
    print(f"Hit Rate:       {hit_rate:.1f}%")
    print("-" * 40)
    print(f"Total Cost:     {total_cost} yen")
    print(f"Total Return:   {total_return} yen")
    print(f"Net Profit:     {total_return - total_cost} yen")
    print(f"Recovery Rate:  {recovery_rate:.1f}%")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
