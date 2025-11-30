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
                    if len(cols) < 10:
                        continue
                    
                    # Helper to get text safely
                    def get_col_text(idx):
                        return cols[idx].get_text(strip=True)

                    try:
                        rank = get_col_text(0)
                        # Handle non-numeric ranks (e.g., 取, 除, 中)
                        if not rank.isdigit():
                            continue
                            
                        horse_name = get_col_text(3)
                        
                        # Gender and Age: 牡3 -> Gender: 牡, Age: 3
                        gender_age = get_col_text(4)
                        gender = gender_age[0] if gender_age else None
                        age = gender_age[1:] if len(gender_age) > 1 else None
                        
                        jockey = get_col_text(6)
                        time_str = get_col_text(7)
                        
                        # Odds (Single Win): Usually col 9 or 10 depending on layout
                        # Let's assume standard layout: 
                        # 0:着順, 1:枠, 2:馬番, 3:馬名, 4:性齢, 5:斤量, 6:騎手, 7:タイム, 8:着差
                        # 9:タイム指数, 10:通過, 11:上り, 12:単勝, 13:人気
                        odds = get_col_text(12)
                        popularity = get_col_text(13)
                        
                        row_data = {
                            'race_id': race_id,
                            'rank': int(rank),
                            'bracket': int(get_col_text(1)),
                            'horse_num': int(get_col_text(2)),
                            'horse_name': horse_name,
                            'gender': gender,
                            'age': int(age) if age and age.isdigit() else None,
                            'jockey': jockey,
                            'time': time_str,
                            'odds': float(odds) if odds and odds.replace('.','',1).isdigit() else None,
                            'popularity': int(popularity) if popularity and popularity.isdigit() else None
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

        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test with a sample file if exists
    pass
