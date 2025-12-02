import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.categorical_cols = ['weather', 'condition', 'gender', 'jockey', 'horse_name', 'surface']

    def fit(self, df):
        """
        Fits label encoders on the dataframe.
        """
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Convert to string to handle mixed types or NaNs
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
        return self

    def transform(self, df):
        """
        Transforms the dataframe into features.
        """
        df = df.copy()
        
        # 1. Target Creation
        # 1st place = 1, others = 0
        if 'rank' in df.columns:
            df['target'] = df['rank'].apply(lambda x: 1 if x == 1 else 0)

        # 2. Categorical Encoding
        for col in self.categorical_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Create a mapping dictionary for faster transformation
                mapping = {label: i for i, label in enumerate(le.classes_)}
                # Map values, filling unknown with -1
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)

        # 3. Numeric Conversion / Cleanup
        # Ensure numeric columns are actually numeric
        numeric_cols = ['bracket', 'horse_num', 'age', 'odds', 'popularity', 'distance']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 4. Select Features for Model
        # We drop non-feature columns like 'date', 'time', 'race_id' (unless used for grouping)
        # For now, we keep 'race_id' and 'date' for splitting, but model shouldn't see them.
        
        return df

    def fit_transform(self, df):
        return self.fit(df).transform(df)
