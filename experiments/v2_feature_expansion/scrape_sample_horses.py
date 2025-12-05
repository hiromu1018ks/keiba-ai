import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"sys.path: {sys.path}")

try:
    import src
    print(f"src location: {src.__file__}")
except ImportError as e:
    print(f"Failed to import src: {e}")

from src.data.scraper import NetkeibaScraper
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    scraper = NetkeibaScraper()
    
    # Sample Horse IDs
    # 2016104880: Contracheck (from race_201801010101)
    horse_ids = ['2016104880']
    
    for horse_id in horse_ids:
        logger.info(f"Scraping horse {horse_id}...")
        success_profile = scraper.scrape_horse_profile(horse_id)
        success_pedigree = scraper.scrape_horse_pedigree(horse_id)
        
        if success_profile:
            logger.info(f"Successfully scraped horse profile {horse_id}")
        else:
            logger.error(f"Failed to scrape horse profile {horse_id}")
            
        if success_pedigree:
            logger.info(f"Successfully scraped horse pedigree {horse_id}")
        else:
            logger.error(f"Failed to scrape horse pedigree {horse_id}")

if __name__ == "__main__":
    main()
