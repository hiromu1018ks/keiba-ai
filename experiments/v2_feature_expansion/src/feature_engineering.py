import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from common.src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.categorical_cols = ['weather', 'condition', 'gender', 'surface', 'distance_category', 'weight_bin',
                                 'around', 'race_class', 'place', 'running_style']
        self.id_cols = ['jockey_id', 'trainer_id', 'jockey_trainer_pair']

    def _convert_time_to_seconds(self, time_str):
        try:
            if pd.isna(time_str): return np.nan
            parts = str(time_str).split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except:
            return np.nan

    def _categorize_distance(self, distance):
        if pd.isna(distance): return 'Unknown'
        if distance < 1400: return 'Sprint'
        elif distance < 1800: return 'Mile'
        elif distance < 2200: return 'Intermediate'
        else: return 'Long'

    def _bin_weight(self, weight):
        if pd.isna(weight): return 'Unknown'
        if weight < 440: return 'Light'
        elif weight < 480: return 'Medium'
        elif weight < 520: return 'Heavy'
        else: return 'SuperHeavy'

    def _classify_running_style(self, passing_order):
        """
        Classifies running style based on the position at the final corner (last number).
        Nigeru (1), Senko (2-4), Sashi (5-9), Oikomi (10+)
        """
        if pd.isna(passing_order): return 'Unknown'
        try:
            # Format: 10-10-9 or 1-1
            parts = str(passing_order).split('-')
            last_pos = int(parts[-1])
            
            if last_pos == 1: return 'Nigeru'
            elif last_pos <= 4: return 'Senko'
            elif last_pos <= 9: return 'Sashi'
            else: return 'Oikomi'
        except:
            return 'Unknown'

    def _convert_margin(self, margin_str):
        """
        Converts margin string to numeric value (lengths).
        """
        if pd.isna(margin_str): return np.nan
        try:
            val = 0.0
            margin_str = str(margin_str)
            if margin_str == '同着': return 0.0
            if margin_str == 'ハナ': return 0.05
            if margin_str == 'アタマ': return 0.1
            if margin_str == 'クビ': return 0.2
            if margin_str == '大差': return 10.0
            
            # Handle fractions like 1/2, 1 1/4
            if '/' in margin_str:
                parts = margin_str.split()
                if len(parts) == 2: # "1 1/2"
                    val += float(parts[0])
                    frac = parts[1].split('/')
                    val += float(frac[0]) / float(frac[1])
                else: # "1/2"
                    frac = margin_str.split('/')
                    val += float(frac[0]) / float(frac[1])
            else:
                val = float(margin_str)
            return val
        except:
            return np.nan

    def fit(self, df):
        """
        Fits encoders on the dataframe.
        """
        temp_df = df.copy()
        
        # 0. Pre-processing for encoding
        if 'distance' in temp_df.columns:
            temp_df['distance_category'] = temp_df['distance'].apply(self._categorize_distance)
        
        if 'horse_weight' in temp_df.columns:
            temp_df['weight_bin'] = temp_df['horse_weight'].apply(self._bin_weight)

        if 'passing_order' in temp_df.columns:
            temp_df['running_style'] = temp_df['passing_order'].apply(self._classify_running_style)

        if 'jockey_id' in temp_df.columns and 'trainer_id' in temp_df.columns:
            temp_df['jockey_trainer_pair'] = temp_df['jockey_id'].astype(str) + '_' + temp_df['trainer_id'].astype(str)

        # Target Creation
        if 'target' not in temp_df.columns and 'rank' in temp_df.columns:
            temp_df['target'] = temp_df['rank'].apply(lambda x: 1 if x == 1 else 0)
            # (Close 2nd logic omitted for simplicity in fit, usually handled in transform or pre-calc)
        
        if 'target' not in temp_df.columns:
            logger.warning("Target column not found in fit. Skipping Target Encoding.")
            return self

        # 1. Target Encoding for IDs (Global)
        for col in self.id_cols:
            if col in temp_df.columns:
                global_mean = temp_df['target'].mean()
                summary = temp_df.groupby(col)['target'].agg(['mean', 'count'])
                alpha = 10
                smoothed_mean = (summary['mean'] * summary['count'] + global_mean * alpha) / (summary['count'] + alpha)
                self.target_encoders[col] = {
                    'map': smoothed_mean.to_dict(),
                    'global_mean': global_mean
                }

        # 2. Label Encoding
        for col in self.categorical_cols:
            if col in temp_df.columns:
                le = LabelEncoder()
                le.fit(temp_df[col].astype(str))
                self.label_encoders[col] = le
        
        return self

    def transform(self, df):
        """
        Transforms the dataframe into features.
        """
        df = df.copy()
        
        # --- Basic Processing ---
        if 'time' in df.columns:
            df['time_seconds'] = df['time'].apply(self._convert_time_to_seconds)

        if 'distance' in df.columns:
            df['distance_category'] = df['distance'].apply(self._categorize_distance)
            
        if 'horse_weight' in df.columns:
            df['weight_bin'] = df['horse_weight'].apply(self._bin_weight)

        if 'passing_order' in df.columns:
            df['running_style'] = df['passing_order'].apply(self._classify_running_style)

        if 'jockey_id' in df.columns and 'trainer_id' in df.columns:
            df['jockey_trainer_pair'] = df['jockey_id'].astype(str) + '_' + df['trainer_id'].astype(str)

        # Target Creation
        if 'rank' in df.columns:
            df['target'] = df['rank'].apply(lambda x: 1 if x == 1 else 0)
            # Close 2nd logic could be added here if needed

        # --- Advanced Features ---
        
        # 1. Log Transformations
        if 'prize' in df.columns:
            df['log_prize'] = np.log1p(df['prize'].fillna(0))
        
        if 'odds' in df.columns:
            df['log_odds'] = np.log1p(df['odds'].fillna(0))

        if 'margin' in df.columns:
            df['margin_val'] = df['margin'].apply(self._convert_margin)

        # 2. Lag Features & Domain Knowledge
        if 'date' in df.columns and 'horse_id' in df.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
                else:
                    df['date_dt'] = df['date']
                    
                df = df.sort_values(['horse_id', 'date_dt'])
                grouped = df.groupby('horse_id')
                
                # Interval (Days since last race)
                df['interval'] = grouped['date_dt'].diff().dt.days.fillna(0)
                
                # Seasonality
                df['month_sin'] = np.sin(2 * np.pi * df['date_dt'].dt.month / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['date_dt'].dt.month / 12)

                # Basic Lags
                df['prev_rank'] = grouped['rank'].shift(1).fillna(99)
                df['ewma_rank_5'] = grouped['rank'].transform(lambda x: x.shift(1).ewm(span=5).mean()).fillna(99)
                
                # Requested Rolling Features (Mean)
                # Rank
                df['rank_3races'] = grouped['rank'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(99)
                df['rank_5races'] = grouped['rank'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(99)
                df['rank_10races'] = grouped['rank'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(99)
                df['rank_1000races'] = grouped['rank'].transform(lambda x: x.shift(1).expanding().mean()).fillna(99)
                
                # Rank Stats (Std, Min, Max) - Rolling 5
                df['rank_5_std'] = grouped['rank'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)
                df['rank_5_min'] = grouped['rank'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).min()).fillna(99)
                df['rank_5_max'] = grouped['rank'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).max()).fillna(99)

                # Prize
                if 'prize' in df.columns:
                    df['prize_3races'] = grouped['prize'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
                    df['prize_5races'] = grouped['prize'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
                    df['prize_10races'] = grouped['prize'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0)
                    df['prize_1000races'] = grouped['prize'].transform(lambda x: x.shift(1).expanding().mean()).fillna(0)
                    
                    # Prize Stats (Std, Max)
                    df['prize_5_std'] = grouped['prize'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)
                    df['prize_5_max'] = grouped['prize'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).max()).fillna(0)

                # Margin Stats
                if 'margin_val' in df.columns:
                    df['avg_margin_5'] = grouped['margin_val'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(99)

                # Time Stats (Best Time per Distance)
                # This is tricky because we need to group by horse AND distance
                # A simple way is to calculate global best time for the horse per distance category
                if 'time_seconds' in df.columns and 'distance_category' in df.columns:
                    # We can't easily do this with simple transform on 'grouped' (which is by horse_id)
                    # We need to iterate or use a more complex groupby
                    # For now, let's skip complex "Best Time per Distance" in this block or implement a simplified version
                    pass

                # Distance Suitability (Avg Rank per Distance Category)
                if 'distance_category' in df.columns:
                    for cat in ['Sprint', 'Mile', 'Intermediate', 'Long']:
                        is_cat = df['distance_category'] == cat
                        df[f'rank_{cat}'] = df['rank'].where(is_cat)
                        # Cumulative mean
                        df[f'avg_rank_{cat}'] = grouped[f'rank_{cat}'].transform(lambda x: x.shift(1).expanding().mean()).fillna(99)
                        df.drop(columns=[f'rank_{cat}'], inplace=True)

                # Surface Suitability
                if 'surface' in df.columns:
                    for surf in ['芝', 'ダ']:
                        surf_name = 'turf' if surf == '芝' else 'dirt'
                        is_surf = df['surface'] == surf
                        df[f'rank_{surf_name}'] = df['rank'].where(is_surf)
                        df[f'avg_rank_{surf_name}'] = grouped[f'rank_{surf_name}'].transform(lambda x: x.shift(1).expanding().mean()).fillna(99)
                        df.drop(columns=[f'rank_{surf_name}'], inplace=True)
                        
                # Place Suitability (Avg Rank per Place)
                if 'place' in df.columns:
                    # Top 4 places: Tokyo, Nakayama, Kyoto, Hanshin
                    for p in ['Tokyo', 'Nakayama', 'Kyoto', 'Hanshin']:
                        is_place = df['place'] == p
                        df[f'rank_{p}'] = df['rank'].where(is_place)
                        df[f'avg_rank_{p}'] = grouped[f'rank_{p}'].transform(lambda x: x.shift(1).expanding().mean()).fillna(99)
                        df.drop(columns=[f'rank_{p}'], inplace=True)

            except Exception as e:
                logger.error(f"Failed to create lag features: {e}")

        # 3. Jockey/Trainer Recent Performance (Win/Place/Show Rate last 100)
        # This requires sorting by date globally, not by horse
        try:
            # Sort by race_id to ensure chronological order within the same day
            # race_id format: YYYY... so it works for sorting
            df = df.sort_values('race_id')
            
            if 'rank' in df.columns:
                df['is_win'] = (df['rank'] == 1).astype(int)
                df['is_place'] = (df['rank'] <= 2).astype(int)
                df['is_show'] = (df['rank'] <= 3).astype(int)
            
                # Jockey
                if 'jockey_id' in df.columns:
                    # Win Rate
                    df['jockey_win_rate_100'] = df.groupby('jockey_id')['is_win'].transform(
                        lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                    ).fillna(0)
                    # Place Rate
                    df['jockey_place_rate_100'] = df.groupby('jockey_id')['is_place'].transform(
                        lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                    ).fillna(0)
                    # Show Rate
                    df['jockey_show_rate_100'] = df.groupby('jockey_id')['is_show'].transform(
                        lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                    ).fillna(0)
                    
                # Trainer
                if 'trainer_id' in df.columns:
                    # Win Rate
                    df['trainer_win_rate_100'] = df.groupby('trainer_id')['is_win'].transform(
                        lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                    ).fillna(0)
                    # Place Rate
                    df['trainer_place_rate_100'] = df.groupby('trainer_id')['is_place'].transform(
                        lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                    ).fillna(0)
                    # Show Rate
                    df['trainer_show_rate_100'] = df.groupby('trainer_id')['is_show'].transform(
                        lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                    ).fillna(0)
                
                df.drop(columns=['is_win', 'is_place', 'is_show'], inplace=True)
                
        except Exception as e:
            logger.error(f"Failed to create jockey/trainer features: {e}")

        # 4. Relative Features (Z-Scores)
        if 'race_id' in df.columns:
            numeric_cols_for_z = ['horse_weight', 'age', 'odds', 'ewma_rank_5', 'interval', 'avg_margin_5']
            for col in numeric_cols_for_z:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    df[f'{col}_zscore'] = df.groupby('race_id')[col].transform(
                        lambda x: (x - x.mean()) / (x.std() + 1e-6)
                    ).fillna(0)

        # 5. Target Encoding Application (Global Map)
        for col in self.id_cols:
            if col in df.columns and col in self.target_encoders:
                encoder = self.target_encoders[col]
                mapping = encoder['map']
                global_mean = encoder['global_mean']
                df[f'{col}_target_enc'] = df[col].map(mapping).fillna(global_mean)

        # 6. Label Encoding Application
        for col in self.categorical_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                mapping = {label: i for i, label in enumerate(le.classes_)}
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        # 7. Cleanup
        numeric_cols = ['bracket', 'horse_num', 'age', 'odds', 'popularity', 'horse_weight', 'weight_change', 'impost',
                        'prev_rank', 'ewma_rank_5', 'log_prize', 'log_odds',
                        'rank_3races', 'rank_5races', 'rank_10races', 'rank_1000races',
                        'prize_3races', 'prize_5races', 'prize_10races', 'prize_1000races',
                        'rank_5_std', 'rank_5_min', 'rank_5_max', 'prize_5_std', 'prize_5_max',
                        'avg_margin_5', 'interval', 'month_sin', 'month_cos',
                        'avg_rank_Sprint', 'avg_rank_Mile', 'avg_rank_Intermediate', 'avg_rank_Long',
                        'avg_rank_turf', 'avg_rank_dirt',
                        'avg_rank_Tokyo', 'avg_rank_Nakayama', 'avg_rank_Kyoto', 'avg_rank_Hanshin',
                        'jockey_win_rate_100', 'jockey_place_rate_100', 'jockey_show_rate_100',
                        'trainer_win_rate_100', 'trainer_place_rate_100', 'trainer_show_rate_100']
        
        # Add z-scores
        numeric_cols += [c for c in df.columns if '_zscore' in c]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def fit_transform(self, df):
        """
        Fits and transforms using K-Fold Target Encoding for training data.
        """
        self.fit(df)
        df_transformed = self.transform(df)
        
        # K-Fold Target Encoding
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        if 'target' not in df.columns:
            return df_transformed

        for col in self.id_cols:
            if col in df.columns:
                df_transformed[f'{col}_target_enc'] = np.nan
                for train_idx, val_idx in kf.split(df):
                    X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
                    global_mean = X_train['target'].mean()
                    summary = X_train.groupby(col)['target'].agg(['mean', 'count'])
                    alpha = 10
                    smoothed_mean = (summary['mean'] * summary['count'] + global_mean * alpha) / (summary['count'] + alpha)
                    df_transformed.loc[val_idx, f'{col}_target_enc'] = X_val[col].map(smoothed_mean).fillna(global_mean)
                
                df_transformed[f'{col}_target_enc'] = df_transformed[f'{col}_target_enc'].fillna(df['target'].mean())

        # Drop winner_time if exists
        if 'winner_time' in df_transformed.columns:
            df_transformed.drop(columns=['winner_time'], inplace=True)
            
        return df_transformed
