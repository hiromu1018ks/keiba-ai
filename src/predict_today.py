import pandas as pd
import numpy as np
import pickle
import os
import datetime
import argparse
import webbrowser
from src.data.parser_shutsuba import ShutsubaParser
from src.data.scraper_playwright import PlaywrightScraper
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Course code mapping
COURSE_NAMES = {
    '01': '札幌', '02': '函館', '03': '福島', '04': '新潟',
    '05': '東京', '06': '中山', '07': '中京', '08': '京都',
    '09': '阪神', '10': '小倉'
}

def parse_race_id(race_id: str) -> dict:
    """Parse race ID into components.
    
    Format: YYYYCCKKDDRR
    - YYYY: Year
    - CC: Course code (01-10)
    - KK: Kai (meeting number)
    - DD: Day
    - RR: Race number
    """
    rid = str(race_id)
    return {
        'year': rid[0:4],
        'course_code': rid[4:6],
        'kai': int(rid[6:8]),
        'day': int(rid[8:10]),
        'race_num': int(rid[10:12]),
        'course_name': COURSE_NAMES.get(rid[4:6], '不明')
    }

def format_race_title(race_id: str) -> str:
    """Format race ID into readable title like '阪神5回1日 第1R'"""
    info = parse_race_id(race_id)
    return f"{info['course_name']}{info['kai']}回{info['day']}日 第{info['race_num']}R"


def main():
    parser = argparse.ArgumentParser(description='Predict race results for a specific date.')
    parser.add_argument('--date', type=str, help='Date in YYYYMMDD format (default: today)')
    args = parser.parse_args()

    # 1. Determine Date
    if args.date:
        target_date_str = args.date
    else:
        target_date_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # Process target date for comparison
    target_date = pd.to_datetime(target_date_str, format='%Y%m%d')
    logger.info(f"Target Date: {target_date_str}")

    # 2. Load Historical Data
    raw_data_path = 'data/common/raw_data/results.csv'
    if not os.path.exists(raw_data_path):
        logger.error(f"Historical data not found at {raw_data_path}. Cannot calculate history features.")
        return
    
    logger.info("Loading historical data...")
    df_history = pd.read_csv(raw_data_path)
    # Ensure date usage is consistent (Handle mixed formats: YYYY-MM-DD or YYYY年MM月DD日)
    if 'date' in df_history.columns and not pd.api.types.is_datetime64_any_dtype(df_history['date']):
         # Try Japanese format first
         date_jp = pd.to_datetime(df_history['date'], format='%Y年%m月%d日', errors='coerce')
         # Try standard pandas guess (YYYY-MM-DD etc)
         date_std = pd.to_datetime(df_history['date'], errors='coerce')
         # Combine: use JP if valid, else standard
         df_history['date'] = date_jp.fillna(date_std)
         
         logger.info(f"History date range: {df_history['date'].min()} to {df_history['date'].max()}")
         logger.info(f"NaT count in history: {df_history['date'].isna().sum()}")

    # 3. Scrape Today's Data (Playwright Batch)
    scraper = PlaywrightScraper()
    race_ids = scraper.scrape_race_ids(target_date_str)
    
    if not race_ids:
        logger.warning(f"No races found for {target_date_str}.")
        return
        
    logger.info(f"Found {len(race_ids)} races. Starting batch scrape...")
    # Batch scrape all race cards
    scraper.scrape_race_cards(race_ids)

    parser_obj = ShutsubaParser()
    logger.info(f"Parsing {len(race_ids)} races...")
    df_today_list = []
    
    for race_id in race_ids:
        # Check if html already scraped or scrape it
        filename = f"shutsuba_{race_id}.html"
        filepath = os.path.join(scraper.data_dir, filename)


            
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                html = f.read()
            df_race = parser_obj.parse(html, race_id=race_id)
            if not df_race.empty:
               df_today_list.append(df_race)
            else:
               logger.warning(f"Empty dataframe for {race_id}")
    
    if not df_today_list:
        logger.error("Failed to parse any races.")
        return

    df_today = pd.concat(df_today_list, axis=0)
    
    # Preprocess df_today
    def parse_jp_date(date_str):
        import re
        try:
            match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', str(date_str))
            if match:
                return pd.to_datetime(f"{match.group(1)}-{match.group(2)}-{match.group(3)}")
            return pd.to_datetime(date_str)
        except:
            return pd.NaT

    if 'date' in df_today.columns:
        df_today['date'] = df_today['date'].apply(parse_jp_date)
    else:
        df_today['date'] = target_date
        
    logger.info(f"Today dates: {df_today['date'].unique()}")
    logger.info(f"Columns: {df_today.columns}")

    # --- Filter Races without Horse Weight (Not yet announced) ---
    # If column missing, it means parser didn't find any weights or failed to create key?
    # Actually if all are None, col should exist.
    if 'horse_weight' not in df_today.columns:
         logger.warning("horse_weight column missing! Assuming no weights announced.")
         # If strict, drop all?
         # For now, let's create the column with NaNs to allow logic to proceed?
         df_today['horse_weight'] = np.nan

    if 'horse_weight' in df_today.columns:
        valid_races = []
        dropped_races = []
        
        for rid, grp in df_today.groupby('race_id'):
            # Check ratio of missing weights
            # "計不" or NaN counts as missing. 
            # In parser, invalid weight becomes None/NaN or remains text if parse fail?
            # Parser tries regex r'(\d+)'. If fail, it's None.
            
            non_null_count = grp['horse_weight'].notna().sum()
            total_count = len(grp)
            
            if non_null_count / total_count < 0.5:
                dropped_races.append(rid)
            else:
                valid_races.append(rid)
        
        if dropped_races:
            logger.info(f"Skipping {len(dropped_races)} races (Weights not announced): {dropped_races}")
            df_today = df_today[df_today['race_id'].isin(valid_races)]
            
            if df_today.empty:
                logger.warning("No races have weight data yet. (Too early?)")
                return

    # Filter JRA Only (01-10)
    # Simulation is trained only on JRA data.
    jra_places = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    df_today['place_code'] = df_today['race_id'].astype(str).str[4:6]
    initial_count = len(df_today)
    df_today = df_today[df_today['place_code'].isin(jra_places)].copy()
    if len(df_today) < initial_count:
        logger.info(f"Filtered out {initial_count - len(df_today)} non-JRA races.")
        if df_today.empty:
            logger.warning("No JRA races datasets found.")
            return

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    
    # Ensure ID columns are consistent with training data (Float/Numeric)
    # results.csv loads IDs as floats/ints. forcing to str breaks Target Encoding keys.
    id_cols = ['horse_id', 'jockey_id', 'trainer_id', 'owner_id', 'sire_id', 'dam_id', 'bms_id', 'breeder_id']
    for col in id_cols:
        if col in df_history.columns:
            df_history[col] = pd.to_numeric(df_history[col], errors='coerce')
        if col in df_today.columns:
            df_today[col] = pd.to_numeric(df_today[col], errors='coerce')
    
    # race_id might be needed as string for date parsing, but feature engineer handles casting.
    # Let's keep race_id as is (usually numeric in csv, string in parser)
    # Unified to string for race_id to be safe for regex slicing
    if 'race_id' in df_history.columns:
        df_history['race_id'] = df_history['race_id'].astype(str)
    if 'race_id' in df_today.columns:
        df_today['race_id'] = df_today['race_id'].astype(str)
    
    # 4. Combine Data
    logger.info(f"Scraped {len(df_today)} entries. Combining with {len(df_history)} history entries.")
    df_combined = pd.concat([df_history, df_today], axis=0, ignore_index=True)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    
    # 5. Feature Engineering
    model_dir = 'models'
    fe_path = os.path.join(model_dir, 'feature_engineer.pkl')
    if not os.path.exists(fe_path):
        logger.error("FeatureEngineer not found. Train model first.")
        return
        
    logger.info("Feature engineering...")
    with open(fe_path, 'rb') as f:
        fe = pickle.load(f)
        
    df_processed = fe.transform(df_combined)
    
    # 6. Predict
    # Filter for today
    df_inference = df_processed[df_processed['date'] == target_date].copy()
    
    if df_inference.empty:
        logger.error(f"No inference data found for {target_date}. Check date matching.")
        # Debug info
        logger.info(f"Target: {target_date}")
        if 'date' in df_processed.columns:
             logger.info(f"Processed dates sample: {df_processed['date'].unique()[-5:]}")
        return

    logger.info(f"Predicting for {len(df_inference)} entries...")
    
    # Load Model
    model_path = os.path.join(model_dir, 'lgbm_calibrated.pkl')
    if not os.path.exists(model_path):
        logger.error("Model not found.")
        return
        

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Feature Selection (try to match model)
    feature_names = None
    try:
        base = model.calibrated_classifiers_[0].base_estimator
        feature_names = base.feature_name_
    except:
        pass
    
    # ... (skipping feature selection logic for brevity in replacement if unchanged) ...
    # Wait, I need to keep the context.
    # Feature Selection mechanism
    model_features = []
    try:
        # Attempt to extract features from the model object
        if hasattr(model, 'calibrated_classifiers_') and model.calibrated_classifiers_:
            base_model = model.calibrated_classifiers_[0].estimator
            if hasattr(base_model, 'booster_'):
                model_features = base_model.booster_.feature_name()
        elif hasattr(model, 'booster_'):
             model_features = model.booster_.feature_name()
        
        if not model_features and hasattr(model, 'feature_name_'):
             model_features = model.feature_name_
             
    except Exception as e:
        logger.warning(f"Could not extract feature names from model: {e}")

    X = df_inference.select_dtypes(include=[np.number])
    exclude_cols = ['rank', 'target', 'is_win', 'is_place', 'is_show', 'prize', 'date', 'race_id', 'horse_id']
    X = X[[c for c in X.columns if c not in exclude_cols]]

    if model_features:
        logger.info(f"Extracted {len(model_features)} features from model.")
        
        # Ensure all features exist
        missing_cols = [c for c in model_features if c not in X.columns]
        if missing_cols:
            logger.warning(f"Missing {len(missing_cols)} features (filling with NaN): {missing_cols[:5]}...")
            for c in missing_cols:
                X[c] = np.nan
        
        # Filter to exact columns
        X = X[model_features]
    else:
        logger.warning("No feature list found. Using all available columns.")

    # Predict Probabilities
    probs = model.predict_proba(X)[:, 1]
    df_inference['pred_prob'] = probs

    # Normalize Probabilities per Race
    # REMOVED: Normalization and scaling (x0.85) logic to match simulation (simulate.py) exactly.
    # Simulation uses raw probabilities from the binary classifier.
    # Previous logic inflated probabilities for low-confidence races (where sum < 0.85), leading to excess bets.
    # logger.info("Normalizing and Scaling probabilities per race (Target Sum: 0.85)...")
    # Logic removed. Using raw 'pred_prob' directly.
            
    # 7. Display Results and Generate HTML
    print(f"\n=== Predictions for {target_date.date()} ===")
    
    # Prepare HTML content
    html_rows = []
    
    # Simulation Settings
    SIM_EV_THRESHOLD = 1.5
    
    for rid, grp in df_inference.groupby('race_id'):
        race_title = format_race_title(rid)
        print(f"\n--- {race_title} ({rid}) ---")
        
        # Sort by Win Probability
        grp = grp.sort_values('pred_prob', ascending=False)
        
        # Normalize Probabilities per Race
        # Since the model is a binary classifier, the raw probabilities are not mutually exclusive and can sum > 1.0.
        # We must normalize them to represent a valid win probability distribution for EV calculation.
        prob_sum = grp['pred_prob'].sum()
        if prob_sum > 0:
            normalized_probs = grp['pred_prob'] / prob_sum
            # Optional: Scale to ~0.9 to account for margin/uncertainty, but 1.0 is standard.
            # Update the 'pred_prob' column in the original df_inference for this race group
            df_inference.loc[grp.index, 'pred_prob'] = normalized_probs.values
            # Re-assign grp to the updated slice to ensure subsequent calculations use normalized probs
            grp = df_inference.loc[grp.index].copy()

        print(f"{'No.':<4} {'Horse':<20} {'Prob':<8} {'Odds':<6} {'EV':<6} {'Rec'}")
        print("-" * 60)
        
        race_rows = []
        for _, row in grp.iterrows():
            prob = row['pred_prob']
            odds = row.get('odds', 0.0)
            if pd.isna(odds): odds = 0.0
            
            # Expected Value = Prob * Odds
            ev = prob * odds
            
            # Recommendation Logic (Aligned with Simulation)
            rec = ""
            rec_class = ""
            
            is_candidate = True
            if odds > 100.0: is_candidate = False # Max Odds Filter
            if prob < 0.05: is_candidate = False  # Min Probability Filter
            
            if is_candidate:
                if ev > 1.5: 
                    rec = "◎" # Strong Buy (User Request: EV > 1.5)
                    rec_class = "strong-buy"
                elif ev >= 1.2: 
                    rec = "○" # Buy
                    rec_class = "buy"
                elif ev > 1.0: 
                    rec = "△" # Watch
                    rec_class = "watch"
            
            print(f"{row['horse_num']:<4} {row['horse_name']:<20} {prob:.4f}   {odds:<6.1f} {ev:<6.2f} {rec}")
            
            race_rows.append({
                'horse_num': row['horse_num'],
                'horse_name': row['horse_name'],
                'prob': prob,
                'odds': odds,
                'ev': ev,
                'rec': rec,
                'rec_class': rec_class
            })
        
        html_rows.append({'race_id': rid, 'race_title': race_title, 'horses': race_rows})
    
    # Generate HTML
    generate_html_output(target_date_str, html_rows)

    # Save CSV for Evaluation
    output_dir = 'output' # Define output_dir here
    os.makedirs(output_dir, exist_ok=True)
 
    csv_data = []
    for race in html_rows:
        rid = race['race_id']
        for h in race['horses']:
            csv_data.append({
                'race_id': rid,
                'horse_num': h['horse_num'],
                'horse_name': h['horse_name'],
                'prob': h['prob'],
                'odds': h['odds'],
                'ev': h['ev'],
                'rec': h['rec'],
                'rec_class': h['rec_class']
            })
    
    if csv_data:
        df_csv = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, f'predictions_{target_date_str}.csv')
        df_csv.to_csv(csv_path, index=False)
        print(f"✅ CSV output saved: {csv_path}")


def generate_html_output(date_str, race_data):
    """Generate HTML output file with predictions."""
    
    html_template = """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>競馬予測 - {date}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Hiragino Kaku Gothic ProN', 'Meiryo', sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        h1 {{ 
            text-align: center; 
            color: #00d4ff; 
            margin-bottom: 30px;
            font-size: 2rem;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }}
        .race-card {{ 
            background: rgba(255,255,255,0.05); 
            border-radius: 12px; 
            margin-bottom: 25px; 
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .race-title {{ 
            font-size: 1.3rem; 
            color: #ffd700; 
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }}
        .race-id {{ font-size: 0.8rem; color: #888; margin-left: 10px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ 
            background: rgba(0, 212, 255, 0.2); 
            padding: 12px 8px; 
            text-align: left;
            font-weight: 600;
        }}
        td {{ padding: 10px 8px; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        .strong-buy {{ 
            background: linear-gradient(90deg, rgba(255,0,100,0.3), transparent) !important;
            font-weight: bold;
        }}
        .strong-buy .rec {{ color: #ff006a; font-size: 1.2rem; }}
        .buy {{ background: linear-gradient(90deg, rgba(255,165,0,0.2), transparent) !important; }}
        .buy .rec {{ color: #ffa500; }}
        .watch {{ background: linear-gradient(90deg, rgba(100,200,255,0.1), transparent) !important; }}
        .watch .rec {{ color: #64c8ff; }}
        .rec {{ font-weight: bold; text-align: center; }}
        .ev-high {{ color: #00ff88; font-weight: bold; }}
        .prob {{ color: #aaa; }}
        .odds {{ color: #ffd700; }}
        .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.8rem; }}
    </style>
</head>
<body>
    <h1>🏇 競馬AI予測 - {date}</h1>
    {races}
    <div class="footer">Generated at {generated_at} | EV > 1.2 = ○ Buy | EV > 1.5 = ◎ Strong Buy</div>
</body>
</html>"""

    race_html = ""
    for race in race_data:
        rows = ""
        for h in race['horses']:
            ev_class = 'ev-high' if h['ev'] > 3.0 else ''
            rows += f"""
            <tr class="{h['rec_class']}">
                <td>{h['horse_num']}</td>
                <td>{h['horse_name']}</td>
                <td class="prob">{h['prob']:.2%}</td>
                <td class="odds">{h['odds']:.1f}</td>
                <td class="{ev_class}">{h['ev']:.2f}</td>
                <td class="rec">{h['rec']}</td>
            </tr>"""
        
        race_html += f"""
        <div class="race-card">
            <div class="race-title">{race['race_title']} <span class="race-id">{race['race_id']}</span></div>
            <table>
                <thead>
                    <tr><th>馬番</th><th>馬名</th><th>確率</th><th>オッズ</th><th>EV</th><th>推奨</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>"""
    
    html_content = html_template.format(
        date=date_str,
        races=race_html,
        generated_at=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    # Save HTML
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'predictions_{date_str}.html')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML output saved: {output_path}")
    
    # Open in browser
    webbrowser.open(f'file://{os.path.abspath(output_path)}')


if __name__ == "__main__":
    main()
