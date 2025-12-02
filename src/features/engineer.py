import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.target_encoders = {}
        self.categorical_cols = ['weather', 'condition', 'gender', 'surface']
        # IDs are handled by target encoding
        self.id_cols = ['jockey_id', 'trainer_id', 'horse_id']

    def _convert_time_to_seconds(self, time_str):
        try:
            if pd.isna(time_str): return np.nan
            # Handle "1:35.5" or "58.5"
            parts = str(time_str).split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except:
            return np.nan

    def fit(self, df):
        """
        Fits encoders on the dataframe.
        """
        # 0. Pre-calculate Target if not present (for encoding)
        temp_df = df.copy()
        if 'target' not in temp_df.columns and 'rank' in temp_df.columns:
            temp_df['target'] = temp_df['rank'].apply(lambda x: 1 if x == 1 else 0)
        
        if 'target' not in temp_df.columns:
            logger.warning("Target column not found in fit. Skipping Target Encoding.")
            return self

        # 1. Target Encoding for IDs
        for col in self.id_cols:
            if col in temp_df.columns:
                # Calculate mean target for each category
                # Global mean for filling unknowns
                global_mean = temp_df['target'].mean()
                summary = temp_df.groupby(col)['target'].agg(['mean', 'count'])
                # Simple smoothing: (mean * count + global * alpha) / (count + alpha)
                alpha = 10
                smoothed_mean = (summary['mean'] * summary['count'] + global_mean * alpha) / (summary['count'] + alpha)
                self.target_encoders[col] = {
                    'map': smoothed_mean.to_dict(),
                    'global_mean': global_mean
                }

        # 2. Label Encoding for low cardinality categorical
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Convert to string to handle mixed types
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
        
        return self

    def transform(self, df):
        """
        Transforms the dataframe into features.
        """
        df = df.copy()
        
        # 1. Target Creation
        if 'rank' in df.columns:
            df['target'] = df['rank'].apply(lambda x: 1 if x == 1 else 0)

        # 2. Time Conversion
        if 'time' in df.columns:
            df['time_seconds'] = df['time'].apply(self._convert_time_to_seconds)

        # 3. Lag Features (Past Performance)
        # Sort by horse and date
        if 'date' in df.columns and 'horse_id' in df.columns:
            try:
                # Convert date to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
                else:
                    df['date_dt'] = df['date']
                    
                df = df.sort_values(['horse_id', 'date_dt'])
                
                # Group by horse
                grouped = df.groupby('horse_id')
                
                # Previous Rank
                df['prev_rank'] = grouped['rank'].shift(1).fillna(99)
                
                # Average Rank last 5
                df['avg_rank_5'] = grouped['rank'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(99)
                
                # Previous Time Seconds
                if 'time_seconds' in df.columns:
                    df['prev_time_seconds'] = grouped['time_seconds'].shift(1).fillna(0)
                
                # Previous Odds
                if 'odds' in df.columns:
                    df['prev_odds'] = grouped['odds'].shift(1).fillna(0)
                
                # Weight Change Average
                if 'weight_change' in df.columns:
                     df['avg_weight_change_5'] = grouped['weight_change'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)

            except Exception as e:
                logger.error(f"Failed to create lag features: {e}")

        # 4. Relative Features (Race Level)
        # Group by race_id
        if 'race_id' in df.columns:
            # Weight Z-Score
            if 'horse_weight' in df.columns:
                df['horse_weight'] = pd.to_numeric(df['horse_weight'], errors='coerce').fillna(0)
                df['weight_zscore'] = df.groupby('race_id')['horse_weight'].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                ).fillna(0)

            # Age Z-Score
            if 'age' in df.columns:
                df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)
                df['age_zscore'] = df.groupby('race_id')['age'].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                ).fillna(0)
            
            # Odds Z-Score
            if 'odds' in df.columns:
                df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(0)
                df['odds_zscore'] = df.groupby('race_id')['odds'].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-6)
                ).fillna(0)

        # 5. Target Encoding Application
        for col in self.id_cols:
            if col in df.columns and col in self.target_encoders:
                encoder = self.target_encoders[col]
                mapping = encoder['map']
                global_mean = encoder['global_mean']
                # Map, fill unknown with global_mean
                df[f'{col}_target_enc'] = df[col].map(mapping).fillna(global_mean)

        # 6. Label Encoding Application
        for col in self.categorical_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Vectorized mapping
                mapping = {label: i for i, label in enumerate(le.classes_)}
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        # 7. Numeric Conversion / Cleanup
        numeric_cols = ['bracket', 'horse_num', 'age', 'odds', 'popularity', 'horse_weight', 'weight_change', 
                        'prev_rank', 'avg_rank_5', 'prev_time_seconds', 'prev_odds', 'avg_weight_change_5',
                        'weight_zscore', 'age_zscore', 'odds_zscore']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
