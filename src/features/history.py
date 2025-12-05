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
        
        # --- Time / Interval ---
        df['interval'] = grouped['date_dt'].diff().dt.days.fillna(0)
        # Seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['date_dt'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date_dt'].dt.month / 12)

        # --- Lag Features (Shift 1 to use PAST data only) ---
        # Window sizes
        windows = [3, 5, 10]
        
        # Targets for aggregation
        targets = {
            'rank': ['mean', 'std', 'min', 'max'],
            'prize': ['mean', 'sum'],
            'agari_3f': ['mean'],
            'first_corner_pos': ['mean'],
            'final_corner_pos': ['mean'],
            'position_gain': ['mean']
        }
        
        # Raw Lags (Previous race)
        df['prev_rank'] = grouped['rank'].shift(1).fillna(99)
        df['prev_prize'] = grouped['prize'].shift(1).fillna(0)
        if 'agari_3f' in df.columns:
            df['prev_agari_3f'] = grouped['agari_3f'].shift(1)

        # Same Jockey Flag
        if 'jockey_id' in df.columns:
            df['prev_jockey_id'] = grouped['jockey_id'].shift(1)
            # If prev is NaN, flag is 0 or NaN? usually 0.
            # But compare string/object.
            # Be careful with NaN != NaN.
            df['same_jockey'] = (df['jockey_id'] == df['prev_jockey_id']).astype(int)
            # If prev_jockey_id is NaN, it will be False (0), which is correct.
            df.drop(columns=['prev_jockey_id'], inplace=True)
        
        # Rolling Aggregations
        for target, funcs in targets.items():
            if target not in df.columns: continue
            
            for window in windows:
                # We shift(1) then rolling(window).
                # closed='left' is not fully supported in all pandas versions with rolling, 
                # so shift(1) + rolling is safer.
                feature_base = grouped[target].shift(1).rolling(window, min_periods=1)
                
                for func in funcs:
                    col_name = f"{target}_{window}_{func}"
                    if func == 'mean':
                        df[col_name] = feature_base.mean()
                    elif func == 'std':
                        df[col_name] = feature_base.std()
                    elif func == 'min':
                        df[col_name] = feature_base.min()
                    elif func == 'max':
                        df[col_name] = feature_base.max()
                    elif func == 'sum':
                        df[col_name] = feature_base.sum()
                        
            # Fill NaNs for specific columns if needed
            # rank NaNs -> maybe mean rank? or 99?
            # For now leave NaN or fill later.
            
        # --- Advanced Trends ---
        # Trend of Rank (Slope of last 5 races)
        if 'rank_5_mean' in df.columns:
            df['rank_momentum_5'] = df['rank_5_mean'] - df['prev_rank']

        # --- Context Specific History (Suitability) ---
        # Distance Suitability (Avg Rank per Distance Category)
        if 'distance_category' in df.columns:
            for cat in ['Sprint', 'Mile', 'Intermediate', 'Long']:
                # Filter rows where distance_category == cat, create a column, then expanding mean
                # Note: We can't simply mask then shift, because shift takes PREVIOUS row regardless of category.
                # We want "Last time this horse ran Sprint".
                
                # Create a series with rank only when category matches
                is_cat = df['distance_category'] == cat
                cat_rank = df['rank'].where(is_cat)
                
                # Shift(1) on the GROUP implies accessing the previous row in the group.
                # If we want "Previous Sprint Rank", we need to shift within the filtered series?
                # No, standard approach:
                # 1. Mask non-matching rows as NaN.
                # 2. GroupBy Horse.
                # 3. Shift(1).
                # 4. Expanding Mean.
                
                # However, shift(1) of a masked series will get the value from previous row (which might be NaN).
                # Effectively, we want expanding statistics on the SUBSET of data for each horse.
                # The correct way:
                # subset = df[df['distance_category'] == cat]
                # subset_grouped = subset.groupby('horse_id')['rank'].apply(lambda x: x.shift(1).expanding().mean())
                # Then merge back?
                # Faster: df.groupby(['horse_id', 'distance_category'])['rank'].shift(1)...
                pass

        # Optimized implementation for Suitability
        # Group by Horse AND Context, then calc expanding mean
        # 1. Distance
        if 'distance_category' in df.columns:
             # Calculate expanding mean of rank per horse+distance
             # We must be careful about leakage. Shift(1) first.
             # Actually, we want: For this row, what is the average of PAST ranks in this category?
             # So we group by [horse, distance], shift(1), expanding mean.
             
             # But we need to assign it back to the original dataframe where categories match.
             # And for rows where category doesn't match... well, FeatureEngineer (old) did it by column:
             # df[f'avg_rank_{cat}']
             # So for a Sprint race, we want to know avg_rank_Sprint, avg_rank_Mile... (history of all categories?)
             # Usually we only care about CURRENT category suitability?
             # No, FeatureEngineer output many features: avg_rank_Sprint, avg_rank_Mile.
             # This allows model to see "He is bad at Sprint" even if today is Mile.
             
             for cat in ['Sprint', 'Mile', 'Intermediate', 'Long']:
                 # Create a helper series: Rank if cat, else NaN
                 s = df['rank'].where(df['distance_category'] == cat)
                 # Group by horse, calculate expanding mean (ignoring NaNs automatically?)
                 # Pandas expanding() ignores NaNs? Yes usually.
                 # But we need to propagate the last value forward? 
                 # If I didn't run Sprint today, my avg_rank_Sprint is same as yesterday.
                 # So: ffill().
                 
                 # Logic:
                 # 1. Extract series for the category.
                 # 2. Group by Horse.
                 # 3. Shift(1).
                 # 4. Expanding Mean.
                 # 5. FFill (to carry forward the knowledge).
                 # 6. Fillna(99) (for horses who never ran this category).
                 
                 df[f'avg_rank_{cat}'] = grouped[f'rank_{cat}'].transform(lambda x: x.shift(1).expanding().mean().ffill()).fillna(99) if f'rank_{cat}' in df else 99
                 # Wait, 'rank_cat' needs to be created first.
                 
                 df[f'temp_rank_{cat}'] = df['rank'].where(df['distance_category'] == cat)
                 df[f'avg_rank_{cat}'] = grouped[f'temp_rank_{cat}'].transform(lambda x: x.shift(1).expanding().mean().ffill()).fillna(99)
                 df.drop(columns=[f'temp_rank_{cat}'], inplace=True)
                 
        # 2. Surface
        if 'surface' in df.columns:
            for surf in ['芝', 'ダ']:
                surf_name = 'turf' if surf == '芝' else 'dirt'
                df[f'temp_rank_{surf_name}'] = df['rank'].where(df['surface'] == surf)
                df[f'avg_rank_{surf_name}'] = grouped[f'temp_rank_{surf_name}'].transform(lambda x: x.shift(1).expanding().mean().ffill()).fillna(99)
                df.drop(columns=[f'temp_rank_{surf_name}'], inplace=True)

        # 3. Place Suitability (Avg Rank per Place)
        # Assuming 'place' column exists (extracted in FeatureEngineer)
        if 'place' in df.columns:
            # We can pick major places or all
            # 01: Sapporo, 02: Hakodate, 03: Fukushima, 04: Niigata, 05: Tokyo, 06: Nakayama, 07: Chukyo, 08: Kyoto, 09: Hanshin, 10: Kokura
            # FeatureEngineer extract place code (05, 06 etc)
            places = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
            for p in places:
                # We can name it readable if we map codes, but code is fine for LightGBM
                df[f'temp_rank_place_{p}'] = df['rank'].where(df['place'] == p)
                df[f'avg_rank_place_{p}'] = grouped[f'temp_rank_place_{p}'].transform(lambda x: x.shift(1).expanding().mean().ffill()).fillna(99)
                df.drop(columns=[f'temp_rank_place_{p}'], inplace=True)
                
        return df
