import requests
from bs4 import BeautifulSoup
import time
import os
import re
from common.src.utils.logger import setup_logger

logger = setup_logger(__name__)

class NetkeibaScraper:
    BASE_URL = "https://db.netkeiba.com"

    def __init__(self, data_dir='common/data/html'):
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
        calendar_filename = f"calendar_{year}{month:02d}.html"
        calendar_filepath = os.path.join(self.data_dir, calendar_filename)
        
        if os.path.exists(calendar_filepath):
            logger.info(f"Loading calendar from local file: {calendar_filename}")
            with open(calendar_filepath, 'r', encoding='utf-8') as f:
                html = f.read()
        else:
            url = f"{self.BASE_URL}/?pid=race_top&date={year}{month:02d}"
            logger.info(f"Scraping calendar: {url}")
            html = self._get_html(url)
            if not html:
                return []
            self._save_html(html, calendar_filename)
        
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
            date_str = re.search(r'/race/list/(\d+)/', daily_link).group(1)
            daily_filename = f"daily_{date_str}.html"
            daily_filepath = os.path.join(self.data_dir, daily_filename)
            
            if os.path.exists(daily_filepath):
                # logger.info(f"Loading daily list from local file: {daily_filename}")
                with open(daily_filepath, 'r', encoding='utf-8') as f:
                    daily_html = f.read()
            else:
                daily_url = f"{self.BASE_URL}{daily_link}"
                logger.info(f"Scraping daily list: {daily_url}")
                daily_html = self._get_html(daily_url)
                if not daily_html:
                    continue
                self._save_html(daily_html, daily_filename)

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
        Skips if the file already exists.
        """
        filename = f"race_{race_id}.html"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.info(f"File {filename} already exists. Skipping.")
            return True

        url = f"{self.BASE_URL}/race/{race_id}/"
        logger.info(f"Scraping race result: {url}")
        html = self._get_html(url)
        if html:
            self._save_html(html, filename)
            return True
        return False

if __name__ == "__main__":
    scraper = NetkeibaScraper()
    
    # Scrape data from 2018 to the current month
    # Note: This will take several hours to complete.
    import datetime
    now = datetime.datetime.now()
    start_year = 2018
    current_year = now.year
    current_month = now.month

    for year in range(start_year, current_year + 1):
        for month in range(1, 13):
            # Stop if we exceed the current month in the current year
            if year == current_year and month > current_month:
                break

            logger.info(f"--- Starting scraping for {year}-{month:02d} ---")
            
            # 1. Get all race IDs for the month
            race_ids = scraper.scrape_race_calendar(year, month)
            
            if not race_ids:
                logger.warning(f"No races found for {year}-{month:02d}")
                continue

            # 2. Scrape result for each race
            for rid in race_ids:
                # The scraper has a built-in 1s delay in _get_html, so we just call it.
                # It also skips existing files, so it's safe to re-run.
                scraper.scrape_race_result(rid)
            
            logger.info(f"--- Completed scraping for {year}-{month:02d} ---")

