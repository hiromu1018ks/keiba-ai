import os
import sys
from src.data.scrape_horses import get_all_horse_ids
from src.data.scraper import NetkeibaScraper

def check_skip_logic():
    scraper = NetkeibaScraper()
    horse_ids = get_all_horse_ids(scraper.data_dir)
    horse_ids = sorted(list(horse_ids))
    
    print(f"Total horses: {len(horse_ids)}")
    print("Checking first 10 horses:")
    
    for i, horse_id in enumerate(horse_ids[:10]):
        horse_dir = os.path.join(scraper.data_dir, 'horse')
        
        # Profile
        filename_prof = f"horse_{horse_id}.html"
        filepath_prof = os.path.join(horse_dir, filename_prof)
        exists_prof = os.path.exists(filepath_prof) and os.path.getsize(filepath_prof) > 0
        
        # Pedigree
        filename_ped = f"ped_{horse_id}.html"
        filepath_ped = os.path.join(horse_dir, filename_ped)
        exists_ped = os.path.exists(filepath_ped) and os.path.getsize(filepath_ped) > 0
        
        print(f"[{i}] ID: {horse_id}")
        print(f"  Profile: {filepath_prof} -> Exists: {exists_prof}")
        print(f"  Pedigree: {filepath_ped} -> Exists: {exists_ped}")

if __name__ == "__main__":
    check_skip_logic()
