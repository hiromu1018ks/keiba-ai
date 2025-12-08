import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class HistoryFeature:
    def __init__(self):
        pass

    def _parse_passing_order(self, order_str):
        """
        Parses passing_order string (e.g., "10-10-9-9") into list of integers.
        Returns first and last position.
        """
        try:
            if pd.isna(order_str):
                return np.nan, np.nan
            parts = str(order_str).split('-')
            # Filter non-numeric (some times has chars?)
            p_ints = []
            for p in parts:
                try:
                    p_ints.append(int(p))
                except:
                    pass
            if not p_ints:
                return np.nan, np.nan
            return p_ints[0], p_ints[-1]
        except:
            return np.nan, np.nan

    def create_history_features(self, df):
        """
        Create Lag features and Trend statistics.
        Refactored to avoid DataFrame fragmentation using pd.concat.
        """
        df = df.copy()
        
        # Ensure date is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
             df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
        elif 'date' in df.columns:
            df['date_dt'] = df['date']
            
        if 'horse_id' not in df.columns or 'date_dt' not in df.columns:
            logger.warning(f"Missing horse_id or date for history features. Cols: {df.columns.tolist()}")
            return df

        # Sort for rolling calculation
        df = df.sort_values(['horse_id', 'date_dt'])

        # Pre-calculate derived columns for aggregation
        # Parse passing order
        if 'passing_order' in df.columns:
            pass_stats = df['passing_order'].apply(self._parse_passing_order)
            df['first_corner_pos'] = [x[0] for x in pass_stats]
            df['final_corner_pos'] = [x[1] for x in pass_stats]
            
            # Position Gain (Final - First). Negative is good (moved up).
            df['position_gain'] = df['final_corner_pos'] - df['first_corner_pos']

        # Group by horse
        grouped = df.groupby('horse_id')
        
        # Store new features in a dict
        new_features = {}
        
        # --- Time / Interval ---
        new_features['interval'] = grouped['date_dt'].diff().dt.days.fillna(0)
        # Seasonality
        new_features['month_sin'] = np.sin(2 * np.pi * df['date_dt'].dt.month / 12)
        new_features['month_cos'] = np.cos(2 * np.pi * df['date_dt'].dt.month / 12)

        # --- Lag Features (Shift 1 to use PAST data only) ---
        # Window sizes
        windows = [3, 5, 10, 100]
        
        # Targets for aggregation
        targets = {
            'rank': ['mean', 'std', 'min', 'max', 'median'],
            'prize': ['mean', 'sum', 'max'],
            'agari_3f': ['mean', 'min', 'median'],
            'first_corner_pos': ['mean', 'median'],
            'final_corner_pos': ['mean', 'median'],
            'position_gain': ['mean', 'min', 'max'],
            'odds': ['mean', 'std', 'min', 'max'],
            'popularity': ['mean', 'max'], # worst popularity = max
            'weight': ['mean', 'std'], # impost
            'horse_weight': ['mean', 'std', 'min', 'max', 'median'] # body weight
        }
        
        # Raw Lags (Previous race)
        new_features['prev_rank'] = grouped['rank'].shift(1).fillna(99)
        new_features['prev_rank_2'] = grouped['rank'].shift(2).fillna(99)
        new_features['prev_rank_3'] = grouped['rank'].shift(3).fillna(99)
        
        new_features['prev_prize'] = grouped['prize'].shift(1).fillna(0)
        new_features['prev_prize_2'] = grouped['prize'].shift(2).fillna(0)
        new_features['prev_prize_3'] = grouped['prize'].shift(3).fillna(0)
        
        if 'odds' in df.columns:
            new_features['prev_odds'] = grouped['odds'].shift(1)
            new_features['prev_odds_2'] = grouped['odds'].shift(2)
            new_features['prev_odds_3'] = grouped['odds'].shift(3)
            
        if 'popularity' in df.columns:
            new_features['prev_popularity'] = grouped['popularity'].shift(1)
        if 'agari_3f' in df.columns:
            new_features['prev_agari_3f'] = grouped['agari_3f'].shift(1)

        # Same Jockey Flag
        if 'jockey_id' in df.columns:
            prev_jockey_id = grouped['jockey_id'].shift(1)
            new_features['same_jockey'] = (df['jockey_id'] == prev_jockey_id).astype(int)
        
        # Rolling Aggregations
        for target, funcs in targets.items():
            if target not in df.columns: continue
            
            for window in windows:
                feature_base = grouped[target].shift(1).rolling(window, min_periods=1)
                
                for func in funcs:
                    col_name = f"{target}_{window}_{func}"
                    if func == 'mean':
                        new_features[col_name] = feature_base.mean()
                    elif func == 'std':
                        new_features[col_name] = feature_base.std()
                    elif func == 'min':
                        new_features[col_name] = feature_base.min()
                    elif func == 'max':
                        new_features[col_name] = feature_base.max()
                    elif func == 'sum':
                        new_features[col_name] = feature_base.sum()
                    elif func == 'median':
                        new_features[col_name] = feature_base.median()
        
        # --- Advanced Trends ---
        # Trend of Rank (Slope of last 5 races)
        # Need dependencies
        rank_5_mean = new_features.get('rank_5_mean')
        prev_rank = new_features.get('prev_rank')
        
        if rank_5_mean is not None and prev_rank is not None:
            new_features['rank_momentum_5'] = rank_5_mean - prev_rank

        # --- Running Style History (Pace Prediction Base) ---
        if 'running_style' in df.columns:
            # One-hot encode running_style (Unknown will be ignored or separate)
            styles = ['Nigeru', 'Senko', 'Sashi', 'Oikomi']
            # We can't easily use get_dummies because we need to group by horse_id
            # So we create boolean series manually or use get_dummies and merge
            
            # Efficient way:
            for style in styles:
                # 1 if style matches, 0 otherwise
                is_style = (df['running_style'] == style).astype(int)
                
                # Rolling mean (Rate)
                # Group by horse_id, shift 1, rolling mean
                # Note: is_style has same index as df
                s_grouped = is_style.groupby(df['horse_id'])
                
                for w in [5, 10]:
                    new_features[f'rate_{style}_{w}'] = s_grouped.transform(
                        lambda x: x.shift(1).rolling(w, min_periods=1).mean()
                    ).fillna(0)

        # --- Context Specific History (Suitability) ---
        # 1. Distance
        if 'distance_category' in df.columns:
             for cat in ['Sprint', 'Mile', 'Intermediate', 'Long']:
                 # Create a helper series: Rank if cat, else NaN
                 # Need to act on ORIGINAL DF columns
                 s_cat = df['rank'].where(df['distance_category'] == cat)
                 # We can use grouped object on this new series? No, grouped is on df.
                 # Need to groupby again or use transform on the series with the same grouper (horse_id)
                 
                 # transform logic:
                 # s_cat.groupby(df['horse_id']).shift(1).expanding().mean().ffill()
                 # This aligns with index.
                 
                 # Note: df['horse_id'] matches index of s_cat.
                 avg_rank = s_cat.groupby(df['horse_id']).transform(lambda x: x.shift(1).expanding().mean().ffill()).fillna(99)
                 new_features[f'avg_rank_{cat}'] = avg_rank
                 
        # 2. Surface
        if 'surface' in df.columns:
            for surf in ['芝', 'ダ']:
                surf_name = 'turf' if surf == '芝' else 'dirt'
                s_surf = df['rank'].where(df['surface'] == surf)
                avg_rank = s_surf.groupby(df['horse_id']).transform(lambda x: x.shift(1).expanding().mean().ffill()).fillna(99)
                new_features[f'avg_rank_{surf_name}'] = avg_rank

        # 3. Place Suitability (Avg Rank per Place)
        if 'place' in df.columns:
            places = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            for p in places:
                s_place = df['rank'].where(df['place'] == p)
                avg_rank = s_place.groupby(df['horse_id']).transform(lambda x: x.shift(1).expanding().mean().ffill()).fillna(99)
                new_features[f'avg_rank_place_{p}'] = avg_rank
                
        # Final Concat
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
        return df
