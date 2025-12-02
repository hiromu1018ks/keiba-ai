import pandas as pd
import os
import glob
from common.src.parser import NetkeibaParser
from common.src.utils.logger import setup_logger
import re

logger = setup_logger(__name__)

def process_html_files(html_dir='common/data/html', output_dir='common/data/rawdf'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parser = NetkeibaParser()
    all_results = []
    all_payouts = []
    
    # Find all race HTML files
    # Pattern: race_*.html
    files = glob.glob(os.path.join(html_dir, 'race_*.html'))
    logger.info(f"Found {len(files)} race HTML files to process.")

    # Single-threaded processing (Low load)
    for i, filepath in enumerate(files):
        res_df, pay_df = process_single_file(filepath)
        if res_df is not None:
            all_results.append(res_df)
        if pay_df is not None:
            all_payouts.append(pay_df)
        
        # Log progress every 1000 files
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(files)} files...")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(output_dir, 'results.csv')
        final_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}. Total rows: {len(final_df)}")
    else:
        logger.warning("No result data extracted.")

    if all_payouts:
        final_pay_df = pd.concat(all_payouts, ignore_index=True)
        pay_output_path = os.path.join(output_dir, 'payouts.csv')
        final_pay_df.to_csv(pay_output_path, index=False)
        logger.info(f"Saved payout data to {pay_output_path}. Total rows: {len(final_pay_df)}")
    else:
        logger.warning("No payout data extracted.")

def process_single_file(filepath):
    try:
        filename = os.path.basename(filepath)
        match = re.search(r'race_(\d+)\.html', filename)
        race_id = match.group(1) if match else None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        parser = NetkeibaParser()
        df = parser.parse_race_result(html_content, race_id=race_id)
        payout_df = parser.parse_payout(html_content, race_id=race_id)
        
        return (df if not df.empty else None), (payout_df if not payout_df.empty else None)
    except Exception as e:
        # Print error to see it in console during parallel execution
        # print(f"Failed to process {filepath}: {e}")
        return None, None

if __name__ == "__main__":
    process_html_files()
