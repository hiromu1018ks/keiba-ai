import pandas as pd
import lightgbm as lgb
from src.features.engineer import FeatureEngineer
import os

def check_importance():
    data_path = 'data/common/raw_data/results.csv'
    if not os.path.exists(data_path):
        print("Data file not found.")
        return

    df = pd.read_csv(data_path)
    
    # Preprocess dates
    df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
    df['year'] = df['date_dt'].dt.year
    
    # Use data up to 2023 for training
    train_df = df[df['year'] <= 2023]
    
    if train_df.empty:
        print("No training data found.")
        return

    print(f"Training data shape: {train_df.shape}")

    # Feature Engineering
    fe = FeatureEngineer()
    train_X_all = fe.fit_transform(train_df)
    
    # Define features
    exclude_cols = ['race_id', 'date', 'time', 'rank', 'target', 'date_dt', 'year',
                    'horse_name', 'jockey', 'trainer', 'horse_id', 'jockey_id', 'trainer_id', 'time_seconds',
                    'prize', 'surface', 'distance', 'weather', 'condition', 'jockey_trainer_pair']
    feature_cols = [c for c in train_X_all.columns if c not in exclude_cols]
    
    X_train = train_X_all[feature_cols]
    y_train = train_X_all['target']
    
    print(f"Features: {feature_cols}")
    
    # Train Model
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Feature Importance:")
    print(importance.head(20))
    
    # Check for leakage (suspiciously high importance)
    top_feature = importance.iloc[0]
    if top_feature['importance'] > len(feature_cols) * 100: # Heuristic
        print(f"\nWARNING: Feature '{top_feature['feature']}' has suspiciously high importance!")

if __name__ == "__main__":
    check_importance()
