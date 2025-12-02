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
        # Removed horse_id as it causes overfitting due to sparsity
        # Added jockey_trainer_pair for synergy
        self.id_cols = ['jockey_id', 'trainer_id', 'jockey_trainer_pair']

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
        
        # Create Synergy Feature
        if 'jockey_id' in temp_df.columns and 'trainer_id' in temp_df.columns:
            temp_df['jockey_trainer_pair'] = temp_df['jockey_id'].astype(str) + '_' + temp_df['trainer_id'].astype(str)

        # Time Conversion for Target Calculation
        if 'time' in temp_df.columns:
            temp_df['time_seconds'] = temp_df['time'].apply(self._convert_time_to_seconds)

        # Enhanced Target Definition (Expert Logic)
        # 1 if rank=1 OR (rank=2 AND time_diff=0.0)
        if 'target' not in temp_df.columns and 'rank' in temp_df.columns:
            # Default target
            temp_df['target'] = temp_df['rank'].apply(lambda x: 1 if x == 1 else 0)
            
            # Adjust for close 2nd place
            if 'time_seconds' in temp_df.columns and 'race_id' in temp_df.columns:
                # Calculate winner time per race
                # Handle Dead Heats (multiple winners): drop duplicates as times are identical
                winner_times = temp_df[temp_df['rank'] == 1].drop_duplicates(subset=['race_id']).set_index('race_id')['time_seconds']
                temp_df['winner_time'] = temp_df['race_id'].map(winner_times)
                
                # If rank 2 and time == winner_time, set target to 1
                mask_close_2nd = (temp_df['rank'] == 2) & (temp_df['time_seconds'] == temp_df['winner_time'])
                temp_df.loc[mask_close_2nd, 'target'] = 1
        
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
        
        # 0. Create Synergy Feature
        if 'jockey_id' in df.columns and 'trainer_id' in df.columns:
            df['jockey_trainer_pair'] = df['jockey_id'].astype(str) + '_' + df['trainer_id'].astype(str)

        # 1. Time Conversion
        if 'time' in df.columns:
            df['time_seconds'] = df['time'].apply(self._convert_time_to_seconds)

        # 2. Target Creation (Same logic as fit)
        if 'rank' in df.columns:
            df['target'] = df['rank'].apply(lambda x: 1 if x == 1 else 0)
            
            # Adjust for close 2nd place
            if 'time_seconds' in df.columns and 'race_id' in df.columns:
                # We need to be careful here. In transform (test time), we might not know the winner time if we are processing row by row?
                # But here we process batch.
                # However, for training data transform, we can use this.
                # For test data (future), 'rank' and 'time' are unknown!
                # So this block is ONLY for training/validation where we have ground truth.
                # If 'rank' is missing (inference), we skip target creation.
                
                # Calculate winner time per race
                # Note: This requires the dataframe to contain the winner.
                # Handle Dead Heats
                winner_times = df[df['rank'] == 1].drop_duplicates(subset=['race_id']).set_index('race_id')['time_seconds']
                # Map might produce NaNs if winner is not in this batch (unlikely for full race data)
                df['winner_time'] = df['race_id'].map(winner_times)
                
                mask_close_2nd = (df['rank'] == 2) & (df['time_seconds'] == df['winner_time'])
                df.loc[mask_close_2nd, 'target'] = 1

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
                
                # Average Rank last 5 (Simple)
                df['avg_rank_5'] = grouped['rank'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(99)
                
                # EWMA Rank (Time Decay) - span=5
                df['ewma_rank_5'] = grouped['rank'].transform(lambda x: x.shift(1).ewm(span=5).mean()).fillna(99)
                
                # Previous Prize
                if 'prize' in df.columns:
                    df['prev_prize'] = grouped['prize'].shift(1).fillna(0)
                    df['ewma_prize_5'] = grouped['prize'].transform(lambda x: x.shift(1).ewm(span=5).mean()).fillna(0)

                # Previous Time Seconds
                if 'time_seconds' in df.columns:
                    df['prev_time_seconds'] = grouped['time_seconds'].shift(1).fillna(0)
                
                # Previous Odds
                if 'odds' in df.columns:
                    df['prev_odds'] = grouped['odds'].shift(1).fillna(0)
                
                # Weight Change Average
                if 'weight_change' in df.columns:
                     df['avg_weight_change_5'] = grouped['weight_change'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(0)

                # --- Context Specific Features ---
                # Surface specific average rank
                if 'surface' in df.columns:
                    # Create dummy columns for surface
                    is_turf = df['surface'] == '芝'
                    is_dirt = df['surface'] == 'ダ'
                    
                    # Calculate cumulative mean of rank for Turf
                    # We need to shift(1) to avoid leakage
                    df['rank_turf'] = df['rank'].where(is_turf)
                    df['avg_rank_turf'] = grouped['rank_turf'].transform(lambda x: x.shift(1).expanding().mean()).fillna(99)
                    
                    # Calculate cumulative mean of rank for Dirt
                    df['rank_dirt'] = df['rank'].where(is_dirt)
                    df['avg_rank_dirt'] = grouped['rank_dirt'].transform(lambda x: x.shift(1).expanding().mean()).fillna(99)
                    
                    # Drop temporary columns
                    df.drop(columns=['rank_turf', 'rank_dirt'], inplace=True)

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
                
            # Prize Z-Score (New)
            if 'ewma_prize_5' in df.columns:
                df['ewma_prize_zscore'] = df.groupby('race_id')['ewma_prize_5'].transform(
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
                        'prev_rank', 'avg_rank_5', 'ewma_rank_5', 'prev_time_seconds', 'prev_odds', 'avg_weight_change_5',
                        'weight_zscore', 'age_zscore', 'odds_zscore', 'ewma_prize_5', 'ewma_prize_zscore', 'avg_rank_turf', 'avg_rank_dirt']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def fit_transform(self, df):
        """
        Fits and transforms using K-Fold Target Encoding for training data to prevent leakage.
        """
        self.fit(df) # Fit global encoders for future use
        
        # Transform with K-Fold for ID columns
        df_transformed = self.transform(df) # Get other features first
        
        # Overwrite ID target encodings with K-Fold values
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Ensure target exists
        if 'target' not in df.columns:
            return df_transformed

        for col in self.id_cols:
            if col in df.columns:
                # Initialize with NaNs
                df_transformed[f'{col}_target_enc'] = np.nan
                
                for train_idx, val_idx in kf.split(df):
                    # Split data
                    X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
                    
                    # Calculate mean on training fold
                    global_mean = X_train['target'].mean()
                    summary = X_train.groupby(col)['target'].agg(['mean', 'count'])
                    alpha = 10
                    smoothed_mean = (summary['mean'] * summary['count'] + global_mean * alpha) / (summary['count'] + alpha)
                    
                    # Map to validation fold
                    df_transformed.loc[val_idx, f'{col}_target_enc'] = X_val[col].map(smoothed_mean).fillna(global_mean)
                
                # Fill any remaining NaNs (if any) with global mean
                df_transformed[f'{col}_target_enc'] = df_transformed[f'{col}_target_enc'].fillna(df['target'].mean())

        # Drop winner_time if it exists (Leakage removal)
        if 'winner_time' in df_transformed.columns:
            df_transformed.drop(columns=['winner_time'], inplace=True)
            
        return df_transformed
