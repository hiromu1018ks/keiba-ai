import pandas as pd
from bs4 import BeautifulSoup
import re
import os
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ShutsubaParser:
    def parse(self, html_content, race_id=None):
        """
        Parses the Shutsuba-hyo HTML and returns a DataFrame.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # --- Race Info ---
        race_info = {'race_id': race_id}
        
        try:
            # Race Data Box
            # <div class="RaceData01">
            #   09:50発走 / 芝2000m (右) / 天候:晴 / 馬場:良
            # </div>
            race_data_01 = soup.find('div', class_='RaceData01')
            if race_data_01:
                text = race_data_01.get_text(strip=True)
                
                # Extract Surface and Distance
                # "芝2000m" or "ダ1800m" or "障3000m"
                match = re.search(r'(芝|ダ|障)(\d+)m', text)
                if match:
                    race_info['surface'] = match.group(1)
                    race_info['distance'] = int(match.group(2))
                
                # Weather
                match = re.search(r'天候:(\S+)', text)
                if match:
                    race_info['weather'] = match.group(1)
                    
                # Condition
                match = re.search(r'馬場:(\S+)', text)
                if match:
                    race_info['condition'] = match.group(1)
            
            # Additional Info (Date, Place)
            race_data_02 = soup.find('div', class_='RaceData02')
            date_found = False
            if race_data_02:
                spans = race_data_02.find_all('span')
                if len(spans) >= 1:
                     txt = spans[0].get_text(strip=True)
                     # Validate if it looks like a date (YYYY年MM月DD日)
                     if re.search(r'\d+年\d+月\d+日', txt):
                         race_info['date'] = txt
                         date_found = True
                     else:
                         # Sometimes date is in other spans or just not here in shutuba
                         pass
                
                # Place extraction...
                if len(spans) >= 2:
                    # place check...
                    pass

            if not date_found:
                 # Check Title for date (Result page fallback or Shutuba fallback)
                 if soup.title:
                     title_text = soup.title.get_text()
                     match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)', title_text)
                     if match:
                         race_info['date'] = match.group(1)

        except Exception as e:
            logger.warning(f"Failed to parse race info for {race_id}: {e}")

        # --- Horse Table ---
        entries = []
        try:
            # table class="Shutuba_Table" (Note: "Shutuba" not "Shutsuba")
            table = soup.find('table', class_='Shutuba_Table')
            if table:
                rows = table.find_all('tr', class_='HorseList')
                for row in rows:
                    try:
                        # Extract Cells (Shutsuba Style)
                        tds = row.find_all('td')
                        
                        # 1. Bracket
                        # Class can be Waku, Waku1, Waku2... use Regex
                        waku_td = row.find('td', class_=re.compile(r'Waku'))
                        bracket = None
                        if waku_td:
                             txt = waku_td.get_text(strip=True)
                             if txt.isdigit(): bracket = int(txt)
                        
                        # 2. Horse Num
                        # Class can be Umaban, Num...
                        umaban_td = row.find('td', class_=re.compile(r'Umaban|Num'))
                        horse_num = None
                        if umaban_td:
                             txt = umaban_td.get_text(strip=True)
                             if txt.isdigit(): horse_num = int(txt)
                        
                        # 3. Horse Info
                        horse_info_td = row.find('td', class_='HorseInfo')
                        horse_name, horse_id = None, None
                        if horse_info_td:
                            a_tag = horse_info_td.find('a', href=True)
                            if a_tag:
                                horse_name = a_tag.get_text(strip=True)
                                match = re.search(r'/horse/(\d+)', a_tag['href'])
                                if match: horse_id = match.group(1)

                        # 4. Jockey
                        jockey_td = row.find('td', class_='Jockey')
                        jockey, jockey_id = None, None
                        if jockey_td:
                            a_tag = jockey_td.find('a', href=True)
                            if a_tag:
                                jockey = a_tag.get_text(strip=True)
                                match = re.search(r'/jockey/.*?(\d+)', a_tag['href'])
                                if match: jockey_id = match.group(1)

                        # 5. Trainer
                        trainer_td = row.find('td', class_='Trainer')
                        trainer, trainer_id = None, None
                        if trainer_td:
                            a_tag = trainer_td.find('a', href=True)
                            if a_tag:
                                trainer = a_tag.get_text(strip=True)
                                match = re.search(r'/trainer/.*?(\d+)', a_tag['href'])
                                if match: trainer_id = match.group(1)
                                
                        # Parsing logic for other fields (simplified from original)
                        gender, age, weight, horse_weight, weight_change, odds, popularity = None, None, None, None, None, None, None
                        
                        # Age/Sex (Index check or class check)
                        # ... (Reuse existing logic or simplified)
                        # Re-implementing simplified logic to ensure robustness
                        
                        if len(tds) > 4:
                             age_sex_text = tds[4].get_text(strip=True)
                             if len(age_sex_text) >= 2:
                                 gender = age_sex_text[0]
                                 age = age_sex_text[1:]
                        if len(tds) > 5:
                             try: weight = float(tds[5].get_text(strip=True))
                             except: pass
                        if len(tds) > 8:
                             hw_text = tds[8].get_text(strip=True)
                             match = re.search(r'(\d+)', hw_text)
                             if match: horse_weight = int(match.group(1))
                             match_ch = re.search(r'\(([-+]?\d+)\)', hw_text)
                             if match_ch: weight_change = int(match_ch.group(1))
                        
                        # 6. Odds
                        # <td class="Txt_R Popular"><span id="odds-1_01">4.5</span></td>
                        odds_td = row.find('td', class_=re.compile(r'Txt_R\s+Popular'))
                        odds = 0.0
                        if odds_td:
                            txt = odds_td.get_text(strip=True)
                            try:
                                odds = float(txt)
                            except ValueError:
                                odds = 0.0 # '---.-' or similar

                        # Popularity (assuming it's still tds[10] if odds moved)
                        if len(tds) > 10:
                             try: popularity = int(tds[10].get_text(strip=True))
                             except: pass

                        entry = {
                            'race_id': race_id,
                            'bracket': bracket,
                            'horse_num': horse_num,
                            'horse_name': horse_name,
                            'horse_id': horse_id,
                            'jockey': jockey,
                            'jockey_id': jockey_id,
                            'trainer': trainer,
                            'trainer_id': trainer_id,
                            'gender': gender,
                            'age': int(age) if age and age.isdigit() else None,
                            'weight': weight,
                            'horse_weight': horse_weight,
                            'weight_change': weight_change,
                            'odds': odds,
                            'popularity': popularity
                        }
                        entry.update(race_info)
                        entries.append(entry)
                    except Exception as e:
                        continue
            else:
                # Fallback to Result Style parsing
                return self._parse_result_style(soup, race_info, race_id)

        except Exception as e:
            logger.error(f"Failed to parse shutsuba table for {race_id}: {e}")

        # If entries is empty, try fallback just in case or return empty
        if not entries and not table:
             return self._parse_result_style(soup, race_info, race_id)

        return pd.DataFrame(entries)

    def _parse_result_style(self, soup, race_info, race_id):
        """
        Parses result.html style table (RaceTable01) for entry data.
        """
        entries = []
        try:
            # Look for RaceTable01 or any table with horses
            table = soup.find('table', class_='RaceTable01')
            if not table:
                # db uses race_table_01
                table = soup.find('table', class_='race_table_01')
            
            if table:
                rows = table.find_all('tr')
                # Header row is usually first, data rows follow
                for row in rows:
                    tds = row.find_all('td')
                    if not tds: continue # Skip header
                    
                    try:
                        # Assuming Standard Result Columns (0-based)
                        # 0: Rank, 1: Bracket, 2: No, 3: Name, 4: SexAge, 5: Impost, 6: Jockey, 7: Time, 8: Margin, ... 
                        # 9: Popularity?, 10: ?, 11: Odds? (Check netkeiba specific)
                        
                        # Verify column count to be safe
                        if len(tds) < 10: continue

                        # 1. Bracket (Col 1)
                        try: bracket = int(tds[1].get_text(strip=True))
                        except: bracket = None
                        
                        # 2. Horse Num (Col 2)
                        try: horse_num = int(tds[2].get_text(strip=True))
                        except: horse_num = None
                        
                        # 3. Horse Name (Col 3)
                        a_name = tds[3].find('a')
                        horse_name = a_name.get_text(strip=True) if a_name else tds[3].get_text(strip=True)
                        horse_id = None
                        if a_name and 'href' in a_name.attrs:
                             match = re.search(r'/horse/(\d+)', a_name['href'])
                             if match: horse_id = match.group(1)
                        
                        # 4. Sex/Age (Col 4)
                        gender, age = None, None
                        txt_sa = tds[4].get_text(strip=True)
                        if len(txt_sa) >= 2:
                            gender = txt_sa[0]
                            age = txt_sa[1:]
                        
                        # 5. Impost/Weight (Col 5)
                        try: weight = float(tds[5].get_text(strip=True))
                        except: weight = None
                        
                        # 6. Jockey (Col 6)
                        a_jockey = tds[6].find('a')
                        jockey = a_jockey.get_text(strip=True) if a_jockey else tds[6].get_text(strip=True)
                        jockey_id = None
                        if a_jockey and 'href' in a_jockey.attrs:
                            match = re.search(r'/jockey/.*?(\d+)', a_jockey['href'])
                            if match: jockey_id = match.group(1)

                        # Trainer (Index varies, usually after Time/Margin)
                        # In race.netkeiba, Trainer is often col 13 or similar?
                        # Let's inspect typical structure text
                        # But for now, leave Trainer empty if hard to find, or guessing.
                        # Legacy parser said Trainer is Col 18? (db site)
                        # Race site might differ.
                        trainer, trainer_id = None, None # TODO: Locate trainer column if critical
                        
                        # Odds/Pop
                        # In Result table, Odds is usually Col 11 or 12. Pop is 10 or 13.
                        # Let's try to parse from typical indices.
                        # Legacy db: Odds 12, Pop 13.
                        odds, popularity = None, None
                        if len(tds) > 11:
                            try: odds = float(tds[11].get_text(strip=True))
                            except: pass
                        if len(tds) > 10:
                            try: popularity = int(tds[10].get_text(strip=True))
                            except: pass
                        
                        # Horse Weight (Col 14 or similar?)
                        horse_weight, weight_change = None, None
                        if len(tds) > 14:
                             hw_text = tds[14].get_text(strip=True)
                             match = re.search(r'(\d+)', hw_text)
                             if match: horse_weight = int(match.group(1))
                             match_ch = re.search(r'\(([-+]?\d+)\)', hw_text)
                             if match_ch: weight_change = int(match_ch.group(1))

                        entry = {
                            'race_id': race_id,
                            'bracket': bracket,
                            'horse_num': horse_num,
                            'horse_name': horse_name,
                            'horse_id': horse_id,
                            'jockey': jockey,
                            'jockey_id': jockey_id,
                            'trainer': trainer,
                            'trainer_id': trainer_id,
                            'gender': gender,
                            'age': int(age) if age and age.isdigit() else None,
                            'weight': weight,
                            'horse_weight': horse_weight,
                            'weight_change': weight_change,
                            'odds': odds,
                            'popularity': popularity
                        }
                        entry.update(race_info)
                        entries.append(entry)
                    except: continue

        except Exception as e:
             logger.error(f"Failed to parse result style table for {race_id}: {e}")
        
        return pd.DataFrame(entries)

if __name__ == "__main__":
    pass
