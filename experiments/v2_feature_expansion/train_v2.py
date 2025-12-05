import pandas as pd
import lightgbm as lgb
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from experiments.v2_feature_expansion.v2_src.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns

logger = setup_logger(__name__)

def train_model(data_path='data/common/raw_data/results.csv', model_dir='experiments/v2_feature_expansion/models'):
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. Load Data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded data with {len(df)} rows.")

    # Sort by date for Time Series Split logic
    # Parse date if needed
    try:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
        else:
            df['date_dt'] = df['date']
    except Exception as e:
        logger.warning(f"Date parsing failed: {e}. Using original order.")
        df['date_dt'] = pd.to_datetime('1900-01-01') # Dummy

    df = df.sort_values(['date_dt', 'race_id'])

    # 2. Feature Engineering (Generation Phase)
    logger.info("Starting Feature Engineering (Generation)...")
    fe = FeatureEngineer()
    # Step 1: Generate Features on Global Data (History, Interactions)
    # encoding_only=False, but since fit() hasn't been called, no encoding is applied yet.
    df_generated = fe.transform(df, encoding_only=False)
    
    logger.info(f"Generated columns: {len(df_generated.columns)}")
    
    # 3. Time Series Split
    # Use last 20% for testing
    split_idx = int(len(df_generated) * 0.8)
    
    # Features already include 'target' due to transform logic (it creates it)
    # But usually we want to separate target.
    
    df_train = df_generated.iloc[:split_idx].copy()
    df_test = df_generated.iloc[split_idx:].copy()
    
    logger.info(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
    
    # 4. Target Encoding (Fit on Train, Apply to Both)
    logger.info("Fitting Target Encoders on Training data...")
    fe.fit(df_train)
    
    logger.info("Applying Target Encoding...")
    df_train_enc = fe.transform(df_train, encoding_only=True)
    df_test_enc = fe.transform(df_test, encoding_only=True)
    
    # Combine back for column selection (or just use df_train_enc)
    # We need to define features based on one of them
    df_processed = df_train_enc

    # 5. Define Features and Target (Leakage Prevention)
    # Columns to exclude (IDs, Targets, Future Info)
    future_info = [
        # Targets
        'rank', 'time', 'time_seconds', 'target', 'prize', 'log_prize',
        # Post-race status
        'passing_order', 'agari_3f', 'margin', 'margin_val',
        'corner_1コーナー', 'corner_2コーナー', 'corner_3コーナー', 'corner_4コーナー', 'race_laps',
        'running_style', # Derived from passing_order
        'position_gain', # Derived from passing_order
        # Odds/Popularity (Post-race or closing for strict model)
        'odds', 'popularity', 'log_odds', 
        # Derived future
        'is_win', 'is_place', 'is_show'
    ]
    
    meta_cols = [
        'race_id', 'date', 'date_dt', 'horse_name', 'jockey', 'trainer', 
        'horse_id', 'jockey_id', 'trainer_id', 'owner', 'owner_id', 
        'sire_id', 'dam_id', 'bms_id', 'breeder_id',
        'jockey_trainer_pair', 'sire_name', 'dam_name', 'bms_name', 'owner_name', 'breeder_name'
    ]
    
    exclude_cols = set(future_info + meta_cols)
    
    feature_cols = [c for c in df_processed.columns if c not in exclude_cols]
    
    # Raw interaction strings to drop (since we have target encodings now)
    raw_interactions = [
        'sire_surface', 'sire_distance', 'bms_surface', 'bms_distance',
        'jockey_place', 'jockey_surface', 'jockey_distance',
        'trainer_place', 'trainer_surface', 'trainer_distance',
        'owner_surface', 'horse_jockey',
        'age_gender', 'class_distance', 'condition_surface'
    ]
    feature_cols = [c for c in feature_cols if c not in raw_interactions]

    # Additional cleanup
    final_feature_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df_processed[c]):
            final_feature_cols.append(c)
        else:
            # Check if it was supposed to be numeric
            if 'rate' in c or 'score' in c or 'ratio' in c:
                 # Try convert
                 pass
            # logger.warning(f"Dropping non-numeric feature: {c} ({df_processed[c].dtype})") # verbose

    logger.info(f"Final Feature Count: {len(final_feature_cols)}")
    logger.info(f"Features (Head): {final_feature_cols[:10]}")

    X_train = df_train_enc[final_feature_cols]
    y_train = df_train_enc['target']
    X_test = df_test_enc[final_feature_cols]
    y_test = df_test_enc['target']

    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 5. Train LightGBM
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    
    logger.info("Training LightGBM...")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True), lgb.log_evaluation(100)]
    )

    # 6. Evaluate
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]
    
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    logger.info(f"Evaluation Results - Accuracy: {acc:.4f}, LogLoss: {loss:.4f}, AUC: {auc:.4f}")
    
    # 7. Feature Importance
    importance = pd.DataFrame({
        'feature': final_feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Important Features:")
    print(importance.head(20))
    
    # Save importance
    importance.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)

    # 8. Save Model
    model_path = os.path.join(model_dir, 'lgbm_v2.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")

if __name__ == "__main__":
    train_model()
