import pandas as pd
from bs4 import BeautifulSoup
import re
import os
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class NetkeibaParser:
    def parse_race_result(self, html_content, race_id=None):
        """
        Parses the race result HTML and returns a DataFrame of results and a dictionary of race info.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # --- Race Info ---
        race_info = {'race_id': race_id}
        
        # Title and basic info
        try:
            # Example: "2023年1月5日 1回中山1日目 3歳未勝利"
            # The structure depends on the page layout.
            # Usually inside .data_intro
            
            data_intro = soup.find('div', class_='data_intro')
            if data_intro:
                # Date and Course
                # <p class="smalltxt">2023年1月5日 1回中山1日目 3歳未勝利  (混)[指]  (芝右1600)</p>
                smalltxt = data_intro.find('p', class_='smalltxt')
                if smalltxt:
                    text = smalltxt.get_text(strip=True)
                    # Extract date: 2023年1月5日
                    date_match = re.search(r'(\d+年\d+月\d+日)', text)
                    if date_match:
                        race_info['date'] = date_match.group(1)
                    
                    # Extract course details: (芝右1600)
                    course_match = re.search(r'\((芝|ダ|障).+?(\d+)\)', text)
                    if course_match:
                        race_info['surface'] = course_match.group(1)
                        race_info['distance'] = int(course_match.group(2))
                
                # Alternative location for course info (e.g., inside dd p span)
                if 'surface' not in race_info:
                    racedata = data_intro.find('dl', class_='racedata')
                    if racedata:
                        dd = racedata.find('dd')
                        if dd:
                            span = dd.find('span')
                            if span:
                                span_text = span.get_text(strip=True)
                                # Example: "ダ左1800m / 天候 : 晴 / ..."
                                # Extract Surface and Distance
                                # Match "ダ" or "芝" or "障" followed by optional direction and digits
                                match = re.search(r'(芝|ダ|障).+?(\d+)m', span_text)
                                if match:
                                    race_info['surface'] = match.group(1)
                                    race_info['distance'] = int(match.group(2))

                # Weather and Condition
                # <dl class="racedata fc"> ... <dt>天候 :</dt><dd>晴</dd> ... <dt>芝 :</dt><dd>良</dd>
                racedata = data_intro.find('dl', class_='racedata')
                if racedata:
                    texts = racedata.get_text(strip=True)
                    # Simple extraction by searching text
                    if '天候' in texts:
                        weather_match = re.search(r'天候\s*:\s*(\S+)', texts)
                        if weather_match:
                            race_info['weather'] = weather_match.group(1)
                    
                    if '芝' in texts:
                        cond_match = re.search(r'芝\s*:\s*(\S+)', texts)
                        if cond_match:
                            race_info['condition'] = cond_match.group(1)
                    elif 'ダート' in texts:
                        cond_match = re.search(r'ダート\s*:\s*(\S+)', texts)
                        if cond_match:
                            race_info['condition'] = cond_match.group(1)

        except Exception as e:
            logger.warning(f"Failed to parse race info for {race_id}: {e}")

        # --- Race Results Table ---
        results = []
        try:
            table = soup.find('table', class_='race_table_01')
            if table:
                rows = table.find_all('tr')
                # Header is usually row 0
                # Columns: 着順, 枠番, 馬番, 馬名, 性齢, 斤量, 騎手, タイム, 着差, ...
                
                for row in rows[1:]: # Skip header
                    cols = row.find_all('td')
                    # We need at least 19 columns to get Trainer (index 18)
                    if len(cols) < 19:
                        continue
                    
                    # Helper to get text safely
                    def get_col_text(idx):
                        return cols[idx].get_text(strip=True)

                    try:
                        rank = get_col_text(0)
                        # Handle non-numeric ranks (e.g., 取, 除, 中)
                        if not rank.isdigit():
                            continue
                            
                        # Extract Horse ID
                        horse_a = cols[3].find('a')
                        horse_id = None
                        if horse_a and 'href' in horse_a.attrs:
                            # /horse/2015104961/
                            match = re.search(r'/horse/(\d+)/', horse_a['href'])
                            if match:
                                horse_id = match.group(1)

                        # Extract Jockey ID
                        jockey_a = cols[6].find('a')
                        jockey_id = None
                        if jockey_a and 'href' in jockey_a.attrs:
                            # /jockey/result/recent/01088/
                            match = re.search(r'/jockey/.+/(\d+)/', jockey_a['href'])
                            if match:
                                jockey_id = match.group(1)

                        # Extract Basic Info
                        horse_name = get_col_text(3)
                        jockey = get_col_text(6)
                        
                        gender_age = get_col_text(4)
                        gender = None
                        age = None
                        if gender_age and len(gender_age) >= 2:
                            gender = gender_age[0]
                            age = gender_age[1:]

                        # Extract Trainer Info (Col 18)
                        trainer = get_col_text(18)
                        trainer_a = cols[18].find('a')
                        trainer_id = None
                        if trainer_a and 'href' in trainer_a.attrs:
                            # /trainer/result/recent/01088/
                            match = re.search(r'/trainer/.+/(\d+)/', trainer_a['href'])
                            if match:
                                trainer_id = match.group(1)

                        # Extract Horse Weight (Col 14)
                        # Format: 480(+2) or 480(0) or 計不
                        weight_text = get_col_text(14)
                        horse_weight = None
                        weight_change = None
                        if weight_text and weight_text != '計不':
                            match = re.search(r'(\d+)\((.+)\)', weight_text)
                            if match:
                                horse_weight = int(match.group(1))
                                change_str = match.group(2)
                                try:
                                    weight_change = int(change_str)
                                except ValueError:
                                    weight_change = 0 # Handle cases like "前計不" if any

                        odds = get_col_text(12)
                        popularity = get_col_text(13)
                        time_str = get_col_text(7)
                        
                        # New Features
                        passing_order = get_col_text(10)
                        agari_3f = get_col_text(11)
                        owner = get_col_text(19)
                        
                        # Extract Prize Money (Col 20)
                        prize = 0.0
                        if len(cols) > 20:
                            prize_text = get_col_text(20)
                            if prize_text:
                                try:
                                    # Format: 1,100.0 or 1100
                                    prize = float(prize_text.replace(',', ''))
                                except ValueError:
                                    prize = 0.0

                        row_data = {
                            'race_id': race_id,
                            'rank': int(rank),
                            'bracket': int(get_col_text(1)),
                            'horse_num': int(get_col_text(2)),
                            'horse_name': horse_name,
                            'horse_id': horse_id,
                            'gender': gender,
                            'age': int(age) if age and age.isdigit() else None,
                            'jockey': jockey,
                            'jockey_id': jockey_id,
                            'trainer': trainer,
                            'trainer_id': trainer_id,
                            'owner': owner,
                            'horse_weight': horse_weight,
                            'weight_change': weight_change,
                            'time': time_str,
                            'passing_order': passing_order,
                            'agari_3f': float(agari_3f) if agari_3f and agari_3f.replace('.','',1).isdigit() else None,
                            'odds': float(odds) if odds and odds.replace('.','',1).isdigit() else None,
                            'popularity': int(popularity) if popularity and popularity.isdigit() else None,
                            'prize': prize
                        }
                        
                        # Merge race info
                        row_data.update(race_info)
                        results.append(row_data)
                        
                    except Exception as e:
                        # Log but continue processing other rows
                        # logger.debug(f"Error parsing row in {race_id}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to parse result table for {race_id}: {e}")

        # --- Detailed Race Info (Corners & Laps) ---
        try:
            # Corner Passing
            corner_table = soup.find('table', summary='コーナー通過順位')
            if corner_table:
                corner_rows = corner_table.find_all('tr')
                for row in corner_rows:
                    th = row.find('th')
                    td = row.find('td')
                    if th and td:
                        label = th.get_text(strip=True)
                        value = td.get_text(strip=True)
                        # Add to all results
                        for res in results:
                            res[f'corner_{label}'] = value

            # Lap Times
            lap_table = soup.find('table', summary='ラップタイム')
            if lap_table:
                # Usually header is distances (200m, 400m...) and row is time
                # Or header is "ラップ" and "ペース"
                # Structure:
                # <tr><th>200m</th><th>400m</th>...</tr>
                # <tr><td>12.5</td><td>11.0</td>...</tr>
                
                # Actually netkeiba format:
                # tr1: th class="lb" (Header: 200, 400...)
                # tr2: td (Times: 12.2, 10.9...)
                
                rows = lap_table.find_all('tr')
                if len(rows) >= 2:
                    headers = [th.get_text(strip=True) for th in rows[0].find_all('th')]
                    times = [td.get_text(strip=True) for td in rows[1].find_all('td')]
                    
                    # Store as a single string representation or detailed columns
                    # For simplicity, store as string "12.2-10.9-..."
                    lap_str = '-'.join(times)
                    for res in results:
                        res['race_laps'] = lap_str
                        
                    # Also extract Pace (if available in another row or derived)
                    # Usually there is a 'pace' row or table, but often just laps.
                    
        except Exception as e:
            logger.warning(f"Failed to parse detailed race info for {race_id}: {e}")

        return pd.DataFrame(results)

    def parse_payout(self, html_content, race_id=None):
        """
        Parses the payout information (not implemented yet).
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        payouts = []
        
        try:
            pay_block = soup.find('dl', class_='pay_block')
            if not pay_block:
                return pd.DataFrame()
            
            tables = pay_block.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    th = row.find('th')
                    if not th: continue
                    
                    ticket_type = th.get_text(strip=True) # 単勝, 複勝, etc.
                    
                    tds = row.find_all('td')
                    if len(tds) < 2: continue
                    
                    # Helper to split cell content by <br> tags
                    def get_lines(cell):
                        return [text for text in cell.stripped_strings]

                    nums = get_lines(tds[0])
                    amounts = get_lines(tds[1])
                    
                    # Ensure lengths match (sometimes they might not if parsing fails, but usually they do)
                    for n, a in zip(nums, amounts):
                        try:
                            amount_int = int(a.replace(',', ''))
                            payouts.append({
                                'race_id': race_id,
                                'ticket_type': ticket_type,
                                'horse_nums': n,
                                'payout': amount_int
                            })
                        except ValueError:
                            continue
                            
        except Exception as e:
            logger.error(f"Failed to parse payout for {race_id}: {e}")
            
        return pd.DataFrame(payouts)

    def parse_horse_profile(self, html_content, horse_id=None):
        """
        Parses the horse profile HTML to extract Owner, Breeder, and Production Area.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        data = {'horse_id': horse_id}
        
        try:
            # Owner
            # <th>馬主</th><td><a href="...">Name</a></td>
            owner_th = soup.find('th', string='馬主')
            if owner_th:
                owner_td = owner_th.find_next_sibling('td')
                if owner_td:
                    a_tag = owner_td.find('a')
                    if a_tag:
                        data['owner_name'] = a_tag.get_text(strip=True)
                        # Extract owner ID if needed
                        # /owner/486800/
                        match = re.search(r'/owner/(\d+)/', a_tag['href'])
                        if match:
                            data['owner_id'] = match.group(1)

            # Breeder
            # <th>生産者</th><td><a href="...">Name</a></td>
            breeder_th = soup.find('th', string='生産者')
            if breeder_th:
                breeder_td = breeder_th.find_next_sibling('td')
                if breeder_td:
                    a_tag = breeder_td.find('a')
                    if a_tag:
                        data['breeder_name'] = a_tag.get_text(strip=True)
                        match = re.search(r'/breeder/(\d+)/', a_tag['href'])
                        if match:
                            data['breeder_id'] = match.group(1)

            # Production Area
            # <th>産地</th><td>Name</td>
            area_th = soup.find('th', string='産地')
            if area_th:
                area_td = area_th.find_next_sibling('td')
                if area_td:
                    data['production_area'] = area_td.get_text(strip=True)
                    
        except Exception as e:
            logger.error(f"Failed to parse horse profile for {horse_id}: {e}")
            
        return data

    def parse_horse_pedigree(self, html_content, horse_id=None):
        """
        Parses the horse pedigree HTML to extract Sire, Dam, and Broodmare Sire.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        data = {'horse_id': horse_id}
        
        try:
            table = soup.find('table', class_='blood_table')
            if not table:
                return data
                
            rows = table.find_all('tr')
            
            # Sire is in the first row, first cell
            if len(rows) > 0:
                cols = rows[0].find_all('td')
                if len(cols) > 0:
                    sire_a = cols[0].find('a')
                    if sire_a:
                        data['sire_name'] = sire_a.get_text(strip=True)
                        match = re.search(r'/horse/(\w+)/', sire_a['href'])
                        if match:
                            data['sire_id'] = match.group(1)
            
            # Dam is the first cell in the row that starts the second half.
            total_rows = len(rows)
            dam_row_idx = total_rows // 2
            
            if dam_row_idx < total_rows:
                dam_row = rows[dam_row_idx]
                dam_cols = dam_row.find_all('td')
                if len(dam_cols) > 0:
                    dam_a = dam_cols[0].find('a')
                    if dam_a:
                        data['dam_name'] = dam_a.get_text(strip=True)
                        match = re.search(r'/horse/(\w+)/', dam_a['href'])
                        if match:
                            data['dam_id'] = match.group(1)
                            
                    # Broodmare Sire is the NEXT cell in the SAME row (if Dam is 1st gen)
                    if len(dam_cols) > 1:
                        bms_a = dam_cols[1].find('a')
                        if bms_a:
                            data['bms_name'] = bms_a.get_text(strip=True)
                            match = re.search(r'/horse/(\w+)/', bms_a['href'])
                            if match:
                                data['bms_id'] = match.group(1)

        except Exception as e:
            logger.error(f"Failed to parse horse pedigree for {horse_id}: {e}")
            
        return data

if __name__ == "__main__":
    pass
