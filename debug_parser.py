from src.data.parser import NetkeibaParser
import os

def test_parser():
    filepath = 'data/html/race_201807040505.html'
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        html_content = f.read()

    parser = NetkeibaParser()
    try:
        df = parser.parse_race_result(html_content, race_id='201807040505')
        print("Parsed DataFrame:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parser()
