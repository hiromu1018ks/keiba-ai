import pandas as pd
import numpy as np
import pickle
import os
import datetime
import argparse
from src.data.parser_shutsuba import ShutsubaParser
from src.data.scraper_playwright import PlaywrightScraper
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

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

    # -------------------------------------------------------------
    # -------------------------------------------------------------


    # Ensure ID columns are string to prevent TypeError in mixing with history
    id_cols = ['race_id', 'horse_id', 'jockey_id', 'trainer_id']
    for col in id_cols:
        if col in df_history.columns:
            df_history[col] = df_history[col].astype(str)
        if col in df_today.columns:
            df_today[col] = df_today[col].astype(str)
    
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
        
    if feature_names is None:
        logger.warning("Using fallback feature selection.")
        X = df_inference.select_dtypes(include=[np.number])
        exclude_cols = ['rank', 'target', 'is_win', 'is_place', 'is_show', 'prize', 'date', 'race_id', 'horse_id'] 
        # Also exclude unencoded categorical IDs if numeric? (Usually they are string now)
        X = X[[c for c in X.columns if c not in exclude_cols]]
    else:
        missing_cols = [c for c in feature_names if c not in df_inference.columns]
        if missing_cols:
            for c in missing_cols:
                df_inference[c] = 0
        X = df_inference[feature_names]

    # Predict
    # Feature Selection mechanism
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
        logger.warning("No feature list found. Using all available columns. This may cause a shape mismatch error.")

    probs = model.predict_proba(X)[:, 1]
    df_inference['pred_prob'] = probs
    
    # 7. Display Results
    # Join with basic info for display if needed, but df_inference retains info

    # --- Result Display ---
    print(f"\n=== Predictions for {target_date.date()} ===")
    
    for rid, grp in df_inference.groupby('race_id'):
        print(f"\n--- Race {rid} ---")
        # Columns: horse_name, horse_num, pred_prob, odds (if available)
        
        # Sort by Win Probability
        grp = grp.sort_values('pred_prob', ascending=False)
        
        print(f"{'No.':<4} {'Horse':<20} {'Prob':<8} {'Odds':<6} {'EV':<6} {'Rec'}")
        print("-" * 60)
        
        for _, row in grp.iterrows():
            prob = row['pred_prob']
            odds = row.get('odds', 0.0)
            if pd.isna(odds): odds = 0.0
            
            # Expected Value = Prob * Odds
            ev = prob * odds
            
            # Recommendation Logic (Aligned with src/strategy/betting.py)
            # Filters: Odds <= 100, Prob >= 0.05
            rec = ""
            
            is_candidate = True
            if odds > 100.0: is_candidate = False
            if prob < 0.05: is_candidate = False
            
            if is_candidate:
                if ev > 1.5: rec = "◎ (Strong Buy)"
                elif ev > 1.2: rec = "○ (Buy)"
                elif ev > 1.0: rec = "△"
            
            print(f"{row['horse_num']:<4} {row['horse_name']:<20} {prob:.4f}   {odds:<6.1f} {ev:<6.2f} {rec}")

if __name__ == "__main__":
    main()
