import pandas as pd
import os
import glob
from src.data.parser import NetkeibaParser
from src.utils.logger import setup_logger
import re

logger = setup_logger(__name__)

def process_html_files(html_dir='data/html', output_dir='data/common/raw_data'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parser = NetkeibaParser()
    all_results = []
    
    # Find all race HTML files
    # Pattern: race_*.html
    files = glob.glob(os.path.join(html_dir, 'race_*.html'))
    logger.info(f"Found {len(files)} race HTML files to process.")

    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            # Extract race_id from filename: race_202301010101.html
            match = re.search(r'race_(\d+)\.html', filename)
            race_id = match.group(1) if match else None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            df = parser.parse_race_result(html_content, race_id=race_id)
            if not df.empty:
                all_results.append(df)
            
        except Exception as e:
            logger.error(f"Failed to process file {filepath}: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(output_dir, 'results.csv')
        final_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}. Total rows: {len(final_df)}")
    else:
        logger.warning("No data extracted.")

if __name__ == "__main__":
    process_html_files()
