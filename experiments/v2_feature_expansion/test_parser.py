import sys
import os
import pandas as pd

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, project_root)

from src.data.parser import NetkeibaParser
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def main():
    parser = NetkeibaParser()
    
    # 1. Test Race Result Parsing
    race_id = '201801010101'
    # Check both locations
    race_file = f'data/html/race_{race_id}.html'
    if not os.path.exists(race_file):
        race_file = f'common/data/html/race_{race_id}.html'
        
    if os.path.exists(race_file):
        with open(race_file, 'r', encoding='utf-8') as f:
            html = f.read()
        df = parser.parse_race_result(html, race_id)
        print("\n--- Race Result Parsing ---")
        print(df[['horse_name', 'agari_3f', 'passing_order', 'owner']].head())
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if corner/lap info is added
        corner_cols = [c for c in df.columns if 'corner_' in c]
        print(f"Corner Columns: {corner_cols}")
        if corner_cols:
            print(df[corner_cols].iloc[0])
            
        if 'race_laps' in df.columns:
            print(f"Race Laps: {df['race_laps'].iloc[0]}")
    else:
        print(f"Race file {race_file} not found.")

    # 2. Test Horse Profile Parsing
    horse_id = '2016104880'
    profile_file = f'data/html/horse/horse_{horse_id}.html'
    if os.path.exists(profile_file):
        with open(profile_file, 'r', encoding='utf-8') as f:
            html = f.read()
        data = parser.parse_horse_profile(html, horse_id)
        print("\n--- Horse Profile Parsing ---")
        print(data)
    else:
        print(f"Profile file {profile_file} not found.")

    # 3. Test Horse Pedigree Parsing
    pedigree_file = f'data/html/horse/ped_{horse_id}.html'
    if os.path.exists(pedigree_file):
        with open(pedigree_file, 'r', encoding='utf-8') as f:
            html = f.read()
        data = parser.parse_horse_pedigree(html, horse_id)
        print("\n--- Horse Pedigree Parsing ---")
        print(data)
    else:
        print(f"Pedigree file {pedigree_file} not found.")

if __name__ == "__main__":
    main()
