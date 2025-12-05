import pandas as pd
import os
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class PedigreeFeature:
    def __init__(self, horse_data_path='data/common/raw_data/horses.csv'):
        self.horse_data_path = horse_data_path
        self.horse_df = None
        self._load_horse_data()

    def _load_horse_data(self):
        """
        Load horse data from CSV.
        """
        if os.path.exists(self.horse_data_path):
            self.horse_df = pd.read_csv(self.horse_data_path)
            # Ensure horse_id is string/object to match results
            self.horse_df['horse_id'] = self.horse_df['horse_id'].astype(str)
            logger.info(f"Loaded horse data: {len(self.horse_df)} records")
        else:
            logger.warning(f"Horse data file not found: {self.horse_data_path}")
            self.horse_df = pd.DataFrame()

    def merge_pedigree(self, results_df):
        """
        Merge pedigree data into results DataFrame.
        """
        if self.horse_df.empty:
            return results_df

        # Ensure horse_id in results is string
        results_df['horse_id'] = results_df['horse_id'].astype(str)
        
        # Merge
        # We only need specific columns
        cols_to_use = [
            'horse_id', 
            'sire_name', 'sire_id', 
            'dam_name', 'dam_id', 
            'bms_name', 'bms_id',
            'owner_name', 'owner_id',
            'breeder_name', 'breeder_id',
            'production_area'
        ]
        
        # Filter cols that exist
        cols_to_use = [c for c in cols_to_use if c in self.horse_df.columns]
        
        merged_df = pd.merge(results_df, self.horse_df[cols_to_use], on='horse_id', how='left')
        
        return merged_df

    def create_interaction_features(self, df):
        """
        Create interaction features for Target Encoding.
        e.g., Sire x Surface, BMS x Distance Category
        """
        # Fill NaNs for interaction
        df['sire_id'] = df['sire_id'].fillna('unknown')
        df['bms_id'] = df['bms_id'].fillna('unknown')
        df['surface'] = df['surface'].fillna('unknown')
        df['distance_category'] = df['distance_category'].fillna('unknown')
        
        # Sire x Surface (e.g., Deep Impact_Turf)
        df['sire_surface'] = df['sire_id'].astype(str) + '_' + df['surface'].astype(str)
        
        # Sire x Distance (e.g., Deep Impact_Long)
        df['sire_distance'] = df['sire_id'].astype(str) + '_' + df['distance_category'].astype(str)
        
        # BMS x Surface
        df['bms_surface'] = df['bms_id'].astype(str) + '_' + df['surface'].astype(str)
        
        # BMS x Distance
        df['bms_distance'] = df['bms_id'].astype(str) + '_' + df['distance_category'].astype(str)
        
        return df
