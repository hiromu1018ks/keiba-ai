import sys
import os
import glob
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.scraper import NetkeibaScraper
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
logger.setLevel("WARNING")

def get_all_horse_ids(html_dir='data/html'):
    """
    Extracts all unique horse IDs from race HTML files.
    """
    horse_ids = set()
    files = glob.glob(os.path.join(html_dir, 'race_*.html'))
    logger.info(f"Scanning {len(files)} race files for horse IDs...")
    
    for i, filepath in enumerate(files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Pattern: /horse/2015104961/
                matches = re.findall(r'/horse/(\d+)/', content)
                horse_ids.update(matches)
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            
        if (i + 1) % 1000 == 0:
            logger.info(f"Scanned {i + 1}/{len(files)} files. Found {len(horse_ids)} unique horses so far.")
            
    logger.info(f"Total unique horses found: {len(horse_ids)}")
    return sorted(list(horse_ids))

def scrape_horse(scraper, horse_id):
    """
    Scrapes profile and pedigree for a single horse.
    """
    try:
        # Profile
        scraper.scrape_horse_profile(horse_id)
        # Pedigree
        scraper.scrape_horse_pedigree(horse_id)
        return True
    except Exception as e:
        logger.error(f"Error scraping horse {horse_id}: {e}")
        return False

def main():
    scraper = NetkeibaScraper()
    # Suppress scraper logs as well
    import logging
    logging.getLogger("src.data.scraper").setLevel(logging.WARNING)
    
    # 1. Get all horse IDs
    horse_ids = get_all_horse_ids(scraper.data_dir)
    
    if not horse_ids:
        logger.warning("No horse IDs found. Make sure you have scraped race data first.")
        return

    # 2. Scrape horses
    logger.info(f"Starting scraping for {len(horse_ids)} horses...")
    
    # Use ThreadPool for faster scraping? 
    # Netkeiba might block if too fast. Scraper class has 1s sleep.
    # If we use threads, we should be careful.
    # Let's stick to sequential for safety or use a small pool.
    # Scraper._get_html has sleep(1).
    
    from tqdm import tqdm
    
    # Use tqdm for progress bar
    for horse_id in tqdm(horse_ids, desc="Scraping Horses"):
        scrape_horse(scraper, horse_id)

if __name__ == "__main__":
    main()
