import requests
from bs4 import BeautifulSoup
import re

url = "https://race.netkeiba.com/top/schedule.html"
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
try:
    response = requests.get(url, headers=headers)
    response.encoding = 'euc-jp'
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    print("--- Schedule Links ---")
    race_ids = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'race_id' in href:
            match = re.search(r'race_id=(\d+)', href)
            if match:
                race_ids.append(match.group(1))
    
    print(f"Found {len(race_ids)} race IDs.")
    print(race_ids[:10])
    
except Exception as e:
    print(e)
