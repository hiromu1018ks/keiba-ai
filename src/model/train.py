import pandas as pd
import lightgbm as lgb
import os
import pickle
from sklearn.metrics import accuracy_score, log_loss
from src.features.engineer import FeatureEngineer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def train_model(data_path='data/common/raw_data/results.csv', model_dir='models'):
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. Load Data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data with {len(df)} rows.")

    # 2. Feature Engineering
    fe = FeatureEngineer()
    df_processed = fe.fit_transform(df)
    
    # Define features and target
    # Exclude non-feature columns
    # Exclude non-feature columns
    exclude_cols = ['race_id', 'date', 'time', 'rank', 'target', 'date_dt', 
                    'horse_name', 'jockey', 'trainer', 'horse_id', 'jockey_id', 'trainer_id', 'time_seconds']
    feature_cols = [c for c in df_processed.columns if c not in exclude_cols]
    
    X = df_processed[feature_cols]
    y = df_processed['target']
    
    logger.info(f"Training with features: {feature_cols}")

    # 3. Split Data (Simple split for now as we have very little data)
    # In production, use time-series split based on 'date'
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Train LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=100,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)]
    )

    # 5. Evaluate
    y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]
    
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_prob)
    
    logger.info(f"Evaluation Results - Accuracy: {acc:.4f}, LogLoss: {loss:.4f}")

    # 6. Save Model
    model_path = os.path.join(model_dir, 'lgbm_baseline.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")

if __name__ == "__main__":
    train_model()
