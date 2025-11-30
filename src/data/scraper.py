import requests
from bs4 import BeautifulSoup
import time
import os
import re
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class NetkeibaScraper:
    BASE_URL = "https://db.netkeiba.com"

    def __init__(self, data_dir='data/html'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def _get_html(self, url):
        """
        Fetches HTML content from a URL with a delay to be polite.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            time.sleep(1) # Politeness delay
            response = requests.get(url, headers=headers)
            response.encoding = 'euc-jp' # Netkeiba uses EUC-JP
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            return None

    def _save_html(self, content, filename):
        """
        Saves HTML content to a file.
        """
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Saved HTML to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save file {filepath}: {e}")

    def scrape_race_calendar(self, year, month):
        """
        Scrapes the race calendar for a specific year and month.
        Returns a list of race IDs found.
        """
        url = f"{self.BASE_URL}/?pid=race_top&date={year}{month:02d}"
        logger.info(f"Scraping calendar: {url}")
        html = self._get_html(url)
        if not html:
            return []

        self._save_html(html, f"calendar_{year}{month:02d}.html")
        
        soup = BeautifulSoup(html, 'html.parser')
        daily_links = []
        
        # Extract daily list links
        # Example link: /race/list/20230105/
        for a in soup.find_all('a', href=True):
            href = a['href']
            match = re.search(r'/race/list/(\d+)/', href)
            if match:
                daily_links.append(href)
        
        unique_daily_links = sorted(list(set(daily_links)))
        logger.info(f"Found {len(unique_daily_links)} days with races for {year}-{month:02d}")

        race_ids = []
        for daily_link in unique_daily_links:
            daily_url = f"{self.BASE_URL}{daily_link}"
            logger.info(f"Scraping daily list: {daily_url}")
            daily_html = self._get_html(daily_url)
            if not daily_html:
                continue
            
            # Save daily html
            date_str = re.search(r'/race/list/(\d+)/', daily_link).group(1)
            self._save_html(daily_html, f"daily_{date_str}.html")

            daily_soup = BeautifulSoup(daily_html, 'html.parser')
            for a in daily_soup.find_all('a', href=True):
                href = a['href']
                # Race ID link: /race/202301010101/
                match = re.search(r'/race/(\d+)/', href)
                if match:
                    race_ids.append(match.group(1))
        
        unique_race_ids = sorted(list(set(race_ids)))
        logger.info(f"Found {len(unique_race_ids)} races for {year}-{month:02d}")
        return unique_race_ids

    def scrape_race_result(self, race_id):
        """
        Scrapes the result page for a specific race ID.
        """
        url = f"{self.BASE_URL}/race/{race_id}/"
        logger.info(f"Scraping race result: {url}")
        html = self._get_html(url)
        if html:
            self._save_html(html, f"race_{race_id}.html")
            return True
        return False

if __name__ == "__main__":
    # Simple test
    scraper = NetkeibaScraper()
    # Try scraping one month (e.g., Jan 2023)
    race_ids = scraper.scrape_race_calendar(2023, 1)
    if race_ids:
        # Scrape first 3 races as a test
        for rid in race_ids[:3]:
            scraper.scrape_race_result(rid)
