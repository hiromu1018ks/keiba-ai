import pandas as pd
import os
import glob
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class JraVanLoader:
    def __init__(self, data_dir='data/jra_van'):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_csv(self, filename_pattern):
        """
        Loads CSV files matching the pattern from the data directory.
        Returns a concatenated DataFrame.
        """
        search_path = os.path.join(self.data_dir, filename_pattern)
        files = glob.glob(search_path)
        
        if not files:
            logger.warning(f"No files found matching pattern: {search_path}")
            return pd.DataFrame()

        dfs = []
        for file in files:
            try:
                # JRA-VAN CSVs are often Shift-JIS
                df = pd.read_csv(file, encoding='shift_jis')
                dfs.append(df)
                logger.info(f"Loaded {file}")
            except Exception as e:
                logger.error(f"Failed to load {file}: {e}")
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

if __name__ == "__main__":
    # Simple test
    loader = JraVanLoader()
    # Assuming a file named 'sample.csv' exists for testing
    df = loader.load_csv('*.csv')
    print(df.head())
