import pandas as pd
import lightgbm as lgb
import os
import pickle
import numpy as np
import optuna
import json
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
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

    # Sort by date for Time Series Split logic
    try:
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
        else:
            df['date_dt'] = df['date']
    except Exception as e:
        logger.warning(f"Date parsing failed: {e}. Using original order.")
        df['date_dt'] = pd.to_datetime('1900-01-01')

    df = df.sort_values(['date_dt', 'race_id'])

    # 2. Feature Engineering (Generation)
    logger.info("Starting Feature Engineering (Generation)...")
    fe = FeatureEngineer()
    df_generated = fe.transform(df, encoding_only=False)
    
    logger.info(f"Generated columns: {len(df_generated.columns)}")

    # 3. Time Series Split
    split_idx = int(len(df_generated) * 0.8)
    
    df_train = df_generated.iloc[:split_idx].copy()
    df_test = df_generated.iloc[split_idx:].copy()
    
    logger.info(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

    # 4. Target Encoding
    logger.info("Fitting and Applying Target Encoders (Leakage-Free)...")
    
    # Fit and Apply K-Fold TE on Train
    df_train_enc = fe.fit_transform(df_train, encoding_only=True)
    
    # Apply Global TE on Test
    df_test_enc = fe.transform(df_test, encoding_only=True)

    # 5. Define Features and Target
    # Exclude columns that are not features or are raw strings replaced by encoding
    future_info = [
        'rank', 'time', 'time_seconds', 'target', 'prize', 'log_prize',
        'passing_order', 'agari_3f', 'margin', 'margin_val',
        'corner_1コーナー', 'corner_2コーナー', 'corner_3コーナー', 'corner_4コーナー', 'race_laps',
        'running_style', 'position_gain',
        'odds', 'popularity', 'log_odds', 
        'is_win', 'is_place', 'is_show'
    ]
    
    meta_cols = [
        'race_id', 'date', 'date_dt', 'horse_name', 'jockey', 'trainer', 
        'horse_id', 'jockey_id', 'trainer_id', 'owner', 'owner_id', 
        'sire_id', 'dam_id', 'bms_id', 'breeder_id',
        'jockey_trainer_pair', 'sire_name', 'dam_name', 'bms_name', 'owner_name', 'breeder_name'
    ]
    
    raw_interactions = [
        'sire_surface', 'sire_distance', 'bms_surface', 'bms_distance',
        'jockey_place', 'jockey_surface', 'jockey_distance',
        'trainer_place', 'trainer_surface', 'trainer_distance',
        'owner_surface', 'horse_jockey',
        'age_gender', 'class_distance', 'condition_surface'
    ]
    
    exclude_cols = set(future_info + meta_cols + raw_interactions)
    feature_cols = [c for c in df_train_enc.columns if c not in exclude_cols]
    
    # Additional numeric check
    final_feature_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df_train_enc[c]):
            final_feature_cols.append(c)
            
    logger.info(f"Final Feature Count: {len(final_feature_cols)}")
    
    X_train = df_train_enc[final_feature_cols]
    y_train = df_train_enc['target']
    X_test = df_test_enc[final_feature_cols]
    y_test = df_test_enc['target']

    # 4. Hyperparameter Tuning with Optuna (Using subset of Train)
    # Split Train further for Optuna
    split_opt_idx = int(len(X_train) * 0.8)
    X_opt_train = X_train.iloc[:split_opt_idx]
    y_opt_train = y_train.iloc[:split_opt_idx]
    X_opt_val = X_train.iloc[split_opt_idx:]
    y_opt_val = y_train.iloc[split_opt_idx:]

    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        model = LGBMClassifier(**param)
        model.fit(
            X_opt_train, y_opt_train,
            eval_set=[(X_opt_val, y_opt_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        preds = model.predict_proba(X_opt_val)[:, 1]
        return log_loss(y_opt_val, preds)

    logger.info("Starting Optuna optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) 
    
    best_params = study.best_params
    logger.info(f"Best params: {best_params}")
    
    # Save best params
    with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)

    # 5. Train Final Calibrated Model with Best Params on FULL Train
    final_params = best_params.copy()
    final_params['objective'] = 'binary'
    final_params['metric'] = 'binary_logloss'
    final_params['boosting_type'] = 'gbdt'
    final_params['n_estimators'] = 1000
    final_params['verbose'] = -1
    
    base_model = LGBMClassifier(**final_params)

    # Calibrated Classifier (Isotonic Regression)
    # Using CV=5 for calibration. Note: This does its own internal splitting of X_train.
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    
    logger.info("Training Final Calibrated Model (Isotonic) with Best Params...")
    calibrated_model.fit(X_train, y_train)
    
    model = calibrated_model

    # 6. Evaluate
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]
    
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    logger.info(f"Evaluation Results - Accuracy: {acc:.4f}, LogLoss: {loss:.4f}, AUC: {auc:.4f}")

    # 7. Save Model
    model_path = os.path.join(model_dir, 'lgbm_calibrated.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")
    
    # Also save the FeatureEngineer (with fitted encoders)
    fe_path = os.path.join(model_dir, 'feature_engineer.pkl')
    with open(fe_path, 'wb') as f:
        pickle.dump(fe, f)
    logger.info(f"Saved FeatureEngineer to {fe_path}")

if __name__ == "__main__":
    train_model()
