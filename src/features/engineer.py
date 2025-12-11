import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import setup_logger
from src.features.pedigree import PedigreeFeature
from src.features.connections import ConnectionsFeature
from src.features.history import HistoryFeature
from src.features.jravan_features import JraVanFeatures

logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self, use_jravan=True):
        self.label_encoders = {}
        self.target_encoders = {}
        self.categorical_cols = ['weather', 'condition', 'gender', 'surface', 'distance_category', 'weight_bin',
                                 'around', 'race_class', 'place', 'running_style',
                                 'sire_type', 'bms_type']  # JRA-VAN blood type features
        # Added pedigree related columns to id_cols for Target Encoding
        self.id_cols = ['jockey_id', 'trainer_id', 'jockey_trainer_pair',
                        'sire_id', 'bms_id', 'owner_id', 'breeder_id',
                        'sire_surface', 'sire_distance', 'bms_surface', 'bms_distance',
                        'jockey_place', 'jockey_surface', 'jockey_distance',
                        'trainer_place', 'trainer_surface', 'trainer_distance',
                        'owner_surface', 'horse_jockey',
                        'age_gender', 'class_distance', 'condition_surface']
        self.pedigree_engineer = PedigreeFeature()
        self.connections_engineer = ConnectionsFeature()
        self.history_engineer = HistoryFeature()
        
        # JRA-VAN Feature Integration
        self.use_jravan = use_jravan
        if use_jravan:
            try:
                self.jravan_engineer = JraVanFeatures()
            except Exception as e:
                logger.warning(f"Failed to initialize JraVanFeatures: {e}")
                self.use_jravan = False
                self.jravan_engineer = None
        else:
            self.jravan_engineer = None

    def _extract_place(self, race_id):
        try:
            # race_id: YYYYPP... (12 digits). Place code is at index 4,5
            return str(race_id)[4:6]
        except:
            return 'unknown'

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

    def fit(self, df, skip_feature_generation=False):
        """
        Fits encoders on the dataframe.
        """
        temp_df = df.copy()
        
        if not skip_feature_generation:
            # Merge Pedigree
            if 'sire_id' not in temp_df.columns:
                temp_df = self.pedigree_engineer.merge_pedigree(temp_df)
            
            # 0. Pre-processing for encoding
            if 'race_id' in temp_df.columns:
                temp_df['place'] = temp_df['race_id'].apply(self._extract_place)

            if 'distance' in temp_df.columns:
                temp_df['distance_category'] = temp_df['distance'].apply(self._categorize_distance)
            
            if 'horse_weight' in temp_df.columns:
                temp_df['weight_bin'] = temp_df['horse_weight'].apply(self._bin_weight)

            if 'passing_order' in temp_df.columns:
                temp_df['running_style'] = temp_df['passing_order'].apply(self._classify_running_style)

            if 'jockey_id' in temp_df.columns and 'trainer_id' in temp_df.columns:
                temp_df['jockey_trainer_pair'] = temp_df['jockey_id'].astype(str) + '_' + temp_df['trainer_id'].astype(str)
                
            # Create Pedigree Interactions (requires distance_category etc)
            temp_df = self.pedigree_engineer.create_interaction_features(temp_df)
            # Create Connection Interactions
            temp_df = self.connections_engineer.create_connection_features(temp_df)

            # Target Creation
            if 'target' not in temp_df.columns and 'rank' in temp_df.columns:
                temp_df['target'] = temp_df['rank'].apply(lambda x: 1 if x == 1 else 0)
                # (Close 2nd logic omitted for simplicity in fit, usually handled in transform or pre-calc)
            
            # 2. Lag Features & Domain Knowledge
            temp_df = self.history_engineer.create_history_features(temp_df)
        
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

    def transform(self, df, encoding_only=False):
        """
        Transforms the dataframe into features.
        If encoding_only=True, skips feature generation and only applies target encoding.
        """
        df = df.copy()
        
        if encoding_only:
            # 1. Apply Target Encoding for IDs (Global)
            # Just ensure basic columns like place/distance_category are present if needed for interaction encoding?
            # Interactions should already be present if generated before split.
            pass
        else:
            # Merge Pedigree
            if 'sire_id' not in df.columns:
                df = self.pedigree_engineer.merge_pedigree(df)
            
            # --- Basic Processing ---
            if 'time' in df.columns:
                df['time_seconds'] = df['time'].apply(self._convert_time_to_seconds)

            if 'race_id' in df.columns:
                df['place'] = df['race_id'].apply(self._extract_place)

            if 'distance' in df.columns:
                df['distance_category'] = df['distance'].apply(self._categorize_distance)
                
            if 'horse_weight' in df.columns:
                df['weight_bin'] = df['horse_weight'].apply(self._bin_weight)

            if 'passing_order' in df.columns:
                df['running_style'] = df['passing_order'].apply(self._classify_running_style)

            if 'jockey_id' in df.columns and 'trainer_id' in df.columns:
                df['jockey_trainer_pair'] = df['jockey_id'].astype(str) + '_' + df['trainer_id'].astype(str)
                
            # Create Pedigree Interactions
            df = self.pedigree_engineer.create_interaction_features(df)
            # Create Connection Interactions
            df = self.connections_engineer.create_connection_features(df)
            
            # Create Misc Interactions (Categorical Combos)
            if 'age' in df.columns and 'gender' in df.columns:
                df['age_gender'] = df['age'].astype(str) + '_' + df['gender'].astype(str)
            if 'race_class' in df.columns and 'distance_category' in df.columns:
                df['class_distance'] = df['race_class'].astype(str) + '_' + df['distance_category'].astype(str)
            if 'condition' in df.columns and 'surface' in df.columns:
                 df['condition_surface'] = df['condition'].astype(str) + '_' + df['surface'].astype(str)

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
            df = self.history_engineer.create_history_features(df)
            
            # 2.5 JRA-VAN Features Integration (Prev race PCI, running style, etc.)
            if self.use_jravan and self.jravan_engineer is not None:
                try:
                    logger.info("Merging JRA-VAN features...")
                    df = self.jravan_engineer.merge_all_features(df)
                except Exception as e:
                    logger.warning(f"Failed to merge JRA-VAN features: {e}")
            
            # 3. Interaction Features (Ratios)
            if 'horse_weight' in df.columns and 'weight' in df.columns:
                try:
                    df['impost_ratio'] = df['weight'] / df['horse_weight'].replace(0, np.nan)
                except:
                    pass

            if 'weight_change' in df.columns and 'horse_weight' in df.columns:
                try:
                    df['weight_change_ratio'] = df['weight_change'] / df['horse_weight'].replace(0, np.nan)
                except:
                    pass

            # 3. Jockey/Trainer Recent Performance (Win/Place/Show Rate last 100)
            # Note: Using shift(1) approach with proper date sorting to minimize leakage
            try:
                if 'rank' in df.columns and 'date_dt' in df.columns:
                    # Sort by date and race_id to ensure chronological order
                    df = df.sort_values(['date_dt', 'race_id']).reset_index(drop=True)
                    
                    df['is_win'] = (df['rank'] == 1).astype(int)
                    df['is_place'] = (df['rank'] <= 2).astype(int)
                    df['is_show'] = (df['rank'] <= 3).astype(int)
                
                    # Jockey
                    if 'jockey_id' in df.columns:
                        # Win Rate
                        df['jockey_win_rate_100'] = df.groupby('jockey_id')['is_win'].transform(
                            lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                        )
                        # Place Rate
                        df['jockey_place_rate_100'] = df.groupby('jockey_id')['is_place'].transform(
                            lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                        )
                        # Show Rate
                        df['jockey_show_rate_100'] = df.groupby('jockey_id')['is_show'].transform(
                            lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                        )
                        
                    # Trainer
                    if 'trainer_id' in df.columns:
                        # Win Rate
                        df['trainer_win_rate_100'] = df.groupby('trainer_id')['is_win'].transform(
                            lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                        )
                        # Place Rate
                        df['trainer_place_rate_100'] = df.groupby('trainer_id')['is_place'].transform(
                            lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                        )
                        # Show Rate
                        df['trainer_show_rate_100'] = df.groupby('trainer_id')['is_show'].transform(
                            lambda x: x.shift(1).rolling(100, min_periods=10).mean()
                        )
                    
                    df.drop(columns=['is_win', 'is_place', 'is_show'], inplace=True)
                    
            except Exception as e:
                logger.error(f"Failed to create jockey/trainer recent performance: {e}")

            # 3.5 Pace Prediction Features (Race Level Aggregation)
            if 'race_id' in df.columns:
                try:
                    # Expected number of horses for each style (Sum of rates)
                    # rate_Nigeru_5 is 0.0-1.0 probability of being Nigeru based on last 5 races
                    for style in ['Nigeru', 'Senko', 'Oikomi']:
                        col = f'rate_{style}_5'
                        if col in df.columns:
                            # Total expected count in the race
                            df[f'race_expected_{style}_count'] = df.groupby('race_id')[col].transform('sum')
                            
                            # Presence of pure style (Max probability)
                            df[f'race_max_{style}_prob'] = df.groupby('race_id')[col].transform('max')
                            
                    # Pace Index: (Nigeru + Senko) - Oikomi ? High value = High Pace
                    if 'race_expected_Nigeru_count' in df.columns and 'race_expected_Senko_count' in df.columns:
                        df['race_pace_index'] = df['race_expected_Nigeru_count'] * 1.5 + df['race_expected_Senko_count']
                        
                except Exception as e:
                    logger.error(f"Error creating pace features: {e}")

            # 4. Relative Features (Z-Scores, Deviation, Ratio)
            if 'race_id' in df.columns:
                numeric_cols_for_relative = ['horse_weight', 'age', 'weight', 
                                             'prev_rank', 'prev_prize', 'prev_agari_3f', 
                                             'interval', 'rank_5_mean', 'prize_5_mean',
                                             'agari_3f_5_mean', 'position_gain_5_mean']
                
                for col in numeric_cols_for_relative:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            
                            # Calculate stats per race
                            agg_funcs = ['mean', 'std', 'min', 'max']
                            race_stats = df.groupby('race_id')[col].agg(agg_funcs)
                            
                            # Flatten columns
                            race_stats.columns = [f'{col}_race_{stat}' for stat in agg_funcs]
                            
                            # Merge back (left join on race_id)
                            df = df.merge(race_stats, on='race_id', how='left')
                            
                            # Calculate relative features
                            mean_col = f'{col}_race_mean'
                            std_col = f'{col}_race_std'
                            min_col = f'{col}_race_min'
                            max_col = f'{col}_race_max'
                            
                            # Deviation
                            df[f'{col}_diff_race_mean'] = df[col] - df[mean_col]
                            
                            # Z-Score
                            df[f'{col}_zscore'] = (df[col] - df[mean_col]) / (df[std_col] + 1e-6)
                            
                            # Ratio
                            df[f'{col}_ratio_race_mean'] = df[col] / (df[mean_col] + 1e-6)
                            
                            # Min/Max diff
                            df[f'{col}_diff_race_min'] = df[col] - df[min_col]
                            df[f'{col}_diff_race_max'] = df[col] - df[max_col]
                            
                        except Exception as e:
                            logger.error(f"Error calculating relative features for {col}: {e}")

        # --- Apply Encoding (Always done if exists) --- 
        
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
        
        # Add z-scores and other relative stats
        numeric_cols += [c for c in df.columns if '_zscore' in c]
        numeric_cols += [c for c in df.columns if '_ratio_race_mean' in c]
        numeric_cols += [c for c in df.columns if '_diff_race_mean' in c]
        numeric_cols += [c for c in df.columns if '_diff_race_min' in c]
        numeric_cols += [c for c in df.columns if '_diff_race_max' in c]
        
        # Add new lags explicitly
        numeric_cols += ['prev_rank_2', 'prev_rank_3', 'prev_prize_2', 'prev_prize_3',
                         'prev_odds_2', 'prev_odds_3']
                         
        # Add new Pace/Style features
        numeric_cols += [c for c in df.columns if 'rate_' in c] # rate_Nigeru_5, etc.
        numeric_cols += [c for c in df.columns if 'race_expected_' in c]
        numeric_cols += [c for c in df.columns if 'race_max_' in c or 'race_pace_' in c]
        
        # JRA-VAN derived features (prev race data - no leakage)
        jravan_numeric_cols = [
            'prev_pci', 'prev_rpci', 'prev_pci3',
            'prev_agari3f_rank', 'prev_ave3f', 'prev_3f_diff',
            'prev_running_style_jv',
            'prev_corner1', 'prev_corner2', 'prev_corner3', 'prev_corner4',
            'prev_rank_jv',
            'weight_change_jv'
        ]
        numeric_cols += jravan_numeric_cols
        
        # Define feature type-specific fillna values
        rate_cols = ['jockey_win_rate_100', 'jockey_place_rate_100', 'jockey_show_rate_100',
                     'trainer_win_rate_100', 'trainer_place_rate_100', 'trainer_show_rate_100']
        ratio_cols = ['impost_ratio', 'weight_change_ratio']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if col in rate_cols:
                    # Rate columns: fill with global mean (more informative than 0)
                    col_mean = df[col].mean()
                    df[col] = df[col].fillna(col_mean if not pd.isna(col_mean) else 0)
                elif col in ratio_cols:
                    # Ratio columns: fill with 1.0 (neutral baseline)
                    df[col] = df[col].fillna(1.0)
                elif col == 'popularity' or col == 'prev_popularity':
                    # Popularity: fill with 99 (least popular) instead of 0 (which model might see as better than 1)
                    df[col] = df[col].fillna(99)
                else:
                    # Default: fill with 0
                    df[col] = df[col].fillna(0)

        return df

    def fit_transform(self, df, encoding_only=False, skip_feature_generation=False):
        """
        Fits and transforms using K-Fold Target Encoding for training data.
        """
        self.fit(df, skip_feature_generation=skip_feature_generation)
        df_transformed = self.transform(df, encoding_only=encoding_only)
        
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
