import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ConnectionsFeature:
    def __init__(self):
        pass

    def create_connection_features(self, df):
        """
        Create interaction features for Jockeys, Trainers, and Owners.
        """
        df = df.copy()
        
        # Fill NaNs for interaction
        cols_to_fill = ['jockey_id', 'trainer_id', 'owner_id', 'place', 'surface', 'distance_category', 'horse_id']
        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
            else:
                # If column missing, create empty series to avoid errors
                df[col] = 'unknown'

        # --- Jockey Interactions ---
        # Jockey x Place (e.g., Lemaire @ Tokyo)
        df['jockey_place'] = df['jockey_id'].astype(str) + '_' + df['place'].astype(str)
        
        # Jockey x Surface (e.g., Lemaire @ Turf)
        df['jockey_surface'] = df['jockey_id'].astype(str) + '_' + df['surface'].astype(str)
        
        # Jockey x Distance (e.g., Lemaire @ Long)
        df['jockey_distance'] = df['jockey_id'].astype(str) + '_' + df['distance_category'].astype(str)

        # --- Trainer Interactions ---
        # Trainer x Place
        df['trainer_place'] = df['trainer_id'].astype(str) + '_' + df['place'].astype(str)
        
        # Trainer x Surface
        df['trainer_surface'] = df['trainer_id'].astype(str) + '_' + df['surface'].astype(str)
        
        # Trainer x Distance
        df['trainer_distance'] = df['trainer_id'].astype(str) + '_' + df['distance_category'].astype(str)

        # --- Owner Interactions ---
        # Owner x Surface (Owners often specialize)
        if 'owner_id' in df.columns:
            df['owner_surface'] = df['owner_id'].astype(str) + '_' + df['surface'].astype(str)

        # --- Synergy ---
        # Horse x Jockey (Compatibility)
        df['horse_jockey'] = df['horse_id'].astype(str) + '_' + df['jockey_id'].astype(str)
        
        return df
