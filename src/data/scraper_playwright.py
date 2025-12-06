import asyncio
import os
import datetime
import random
import pandas as pd
from playwright.async_api import async_playwright
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PlaywrightScraper:
    def __init__(self, data_dir='data/html/shutsuba'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.base_url = "https://race.netkeiba.com"
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    async def _fetch_race_ids_async(self, date_str):
        ids = []
        url = f"{self.base_url}/top/race_list.html?kaisai_date={date_str}"
        logger.info(f"Fetching Race List: {url}")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=self.user_agent)
            page = await context.new_page()

            try:
                await page.goto(url, timeout=60000, wait_until='domcontentloaded')
                try:
                    await page.wait_for_selector('.RaceList_Box', timeout=5000)
                except:
                    pass
                # await page.wait_for_load_state("domcontentloaded") # Redundant if in goto

                # Extract IDs
                # href example: ../race/shutuba.html?race_id=202506050101&rf=race_list
                locator = page.locator('a[href*="race_id="]')
                count = await locator.count()
                
                for i in range(count):
                    href = await locator.nth(i).get_attribute("href")
                    if "race_id=" in href and ("shutuba.html" in href or "result.html" in href):
                        # Simple extraction
                        # split by race_id= and take the next part until &
                        try:
                            rid_part = href.split("race_id=")[1]
                            if "&" in rid_part:
                                rid = rid_part.split("&")[0]
                            else:
                                rid = rid_part
                            if rid not in ids:
                                ids.append(rid)
                        except:
                            continue
            except Exception as e:
                logger.error(f"Error fetching race list: {e}")
            finally:
                await browser.close()
        
        return sorted(ids)

    async def _fetch_and_save_race_data_async(self, race_ids):
        """
        Fetches Shutuba HTML (for Static Parser) AND Odds (for dynamic data) in batch.
        """
        results = {} # race_id -> {'html_saved': bool, 'odds': pd.DataFrame or None}

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=self.user_agent)
            page = await context.new_page()

            for rid in race_ids:
                results[rid] = {'html_saved': False, 'odds': None}
                
                # 1. Fetch Shutuba HTML (to verify Odds are present in DOM for local debugging/parsing)
                # But actually, users want the Odds Data. 
                # Integrating user's "Scrape Odds to DataFrame" logic directly is better.
                # However, our existing PARSER relies on the HTML file to get Horse Names, Jockeys, etc.
                # So we MUST save the HTML.
                
                url_shutuba = f"{self.base_url}/race/shutuba.html?race_id={rid}"
                logger.info(f"Processing {rid}: {url_shutuba}")
                
                try:
                    await page.goto(url_shutuba, timeout=60000, wait_until='domcontentloaded')
                    
                    # Wait for odds to be NUMERIC (not ---.-) 
                    # This is critical: the page loads with ---.- initially, then JS updates.
                    # We use page.wait_for_function to check for real odds.
                    try:
                        await page.wait_for_function(
                            """() => {
                                const odds = document.querySelectorAll('.Txt_R.Popular span');
                                if (odds.length === 0) return false;
                                // Check if at least one has numeric value
                                for (let o of odds) {
                                    const val = o.textContent.trim();
                                    if (val && !val.includes('---') && !isNaN(parseFloat(val))) {
                                        return true;
                                    }
                                }
                                return false;
                            }""",
                            timeout=15000
                        )
                        logger.info(f"Odds loaded for {rid}")
                    except Exception as wait_err:
                        logger.warning(f"Odds not loaded for {rid} (timeout or no odds): {wait_err}")
                    
                    # Save HTML
                    content = await page.content()
                    filename = f"shutsuba_{rid}.html"
                    filepath = os.path.join(self.data_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    results[rid]['html_saved'] = True
                    
                    # 2. Extract Odds DIRECTLY here?
                    # If we trust the parser, saving the fully rendered HTML *should* be enough.
                    # As proven by the "browser subagent" seeing the odds.
                    # If parser fails, we can implement direct extraction here.
                    
                    # Let's try to extract odds df from the page for robustness
                    # Switch to odds page? or just read from shutuba? 
                    # User suggested odds/index.html is better for pandas.
                    # But shutuba has names.
                    
                    # Let's save the HTML (rendered) -> This should fix Parser.
                    # AND let's trying grabbing odds if possible to double check?
                    # No, keep it simple first. If Playwright saves the rendered DOM, 
                    # the HTML file WILL contain "96.3" instead of "---.-".
                    # So existing parser should Just Work.
                    
                except Exception as e:
                    logger.error(f"Failed to process {rid}: {e}")
                
                # Sleep a bit to be polite
                await asyncio.sleep(1)

            await browser.close()
        return results

    # Synchronous Wrappers
    def scrape_race_ids(self, date_str):
        return asyncio.run(self._fetch_race_ids_async(date_str))

    def scrape_race_cards(self, race_ids):
        """
        Batch scrape race cards.
        """
        return asyncio.run(self._fetch_and_save_race_data_async(race_ids))

    async def _fetch_results_async(self, race_ids):
        """
        Fetch race results (1st place and Tansho payout).
        """
        results = {}
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=self.user_agent)
            page = await context.new_page()

            for rid in race_ids:
                url = f"{self.base_url}/race/result.html?race_id={rid}"
                logger.info(f"Fetching Result: {url}")
                try:
                    await page.goto(url, timeout=60000, wait_until='domcontentloaded')
                    
                    # Wait for Result Table
                    try:
                        await page.wait_for_selector('table.RaceTable01', timeout=10000)
                    except:
                        logger.warning(f"Timeout waiting for RaceTable01 in {rid}")
                        continue
                        
                    # 1. 1st Place Horse Info from Result Table
                    # First row after header is Rank 1
                    first_row = page.locator('table.RaceTable01 tr').nth(1)
                    
                    if await first_row.count() == 0:
                        logger.warning(f"No result rows found for {rid}")
                        continue
                        
                    # Check Rank (ensure it is 1)
                    rank_el = first_row.locator('div.Rank')
                    if await rank_el.count() > 0:
                        rank_text = await rank_el.text_content()
                        if rank_text.strip() != '1':
                            logger.warning(f"First row rank is not 1 for {rid} (Rank: {rank_text})")
                            # Should search for rank 1 if not sorted? Usually sorted.
                    
                    # Horse Num (3rd col)
                    horse_num_el = first_row.locator('td').nth(2)
                    horse_num_text = await horse_num_el.text_content()
                    horse_num = int(horse_num_text.strip())
                    
                    # Odds (Class 'Odds')
                    # Could be multiple Odds classes (OddsPeople, Odds_Ninki), generally td.Odds has text.
                    odds_el = first_row.locator('td.Odds').last # usually multiple, last is ninki?
                    # Let's use strict selector from HTML inspection
                    # <td class="Odds Txt_R"><span class="Odds_Ninki">1.7</span></td>
                    odds_el = first_row.locator('span.Odds_Ninki')
                    if await odds_el.count() == 0:
                         # Fallback
                         odds_el = first_row.locator('td.Odds').last
                    
                    odds_text = await odds_el.text_content()
                    try:
                        odds_val = float(odds_text.strip())
                        payout = int(odds_val * 100)
                    except:
                        payout = 0
                        logger.warning(f"Could not parse odds '{odds_text}' for {rid}")
                    
                    results[rid] = {
                        'win_horse_num': horse_num,
                        'win_payout': payout
                    }
                    logger.info(f"Race {rid}: Win #{horse_num}, Pay {payout}")
                    
                except Exception as e:
                    logger.error(f"Error scraping result for {rid}: {e}")
                
                await asyncio.sleep(2.0)
            
            await browser.close()
        return results

    def scrape_race_results(self, race_ids):
        return asyncio.run(self._fetch_results_async(race_ids))
