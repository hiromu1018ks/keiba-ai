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

    # 1. Load Data / Features
    cache_path = 'data/cache/features_sim.pkl'
    
    if os.path.exists(cache_path):
        logger.info(f"Loading features from cache: {cache_path}")
        df_generated = pd.read_pickle(cache_path)
    else:
        logger.info("Cache not found. Loading raw data and generating features...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with {len(df)} rows.")

        # Date Parsing
        try:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日', errors='coerce')
            else:
                df['date_dt'] = df['date']
            df['year'] = df['date_dt'].dt.year
        except Exception as e:
            logger.warning(f"Date parsing failed: {e}.")
            return

        # JRA Filter
        jra_places = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        df['place_code'] = df['race_id'].astype(str).str[4:6]
        df = df[df['place_code'].isin(jra_places)].copy()
        logger.info(f"Filtered to JRA only: {len(df)} rows")

        # Sort
        df = df.sort_values(['date_dt', 'race_id'])

        # Feature Generation
        logger.info("Starting Feature Engineering (Generation)...")
        fe = FeatureEngineer()
        df_generated = fe.transform(df, encoding_only=False)
        
        # Save to cache for future consistency
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_generated.to_pickle(cache_path)
    
    if 'year' not in df_generated.columns:
         # Try to recover year if missing from cache (though simulations ensure it)
         if 'date_dt' in df_generated.columns:
             df_generated['year'] = df_generated['date_dt'].dt.year
         else:
             logger.error("Year column missing in features.")
             return

    logger.info(f"Total features generated: {len(df_generated.columns)}")

    # 3. Time Series Split (Year Based)
    # Use data up to 2023 for Training, 2024+ for Testing/Evaluation
    # Or optimize hyperparameters using 2023, evaluate on 2024.
    # Given we are in Dec 2025, let's use < 2024 for Train, >= 2024 for Test.
    
    train_mask = df_generated['year'] < 2024
    test_mask = df_generated['year'] >= 2024
    
    df_train = df_generated[train_mask].copy()
    df_test = df_generated[test_mask].copy()
    
    logger.info(f"Train size (Available < 2024): {len(df_train)}, Test size (>= 2024): {len(df_test)}")

    # 4. Target Encoding
    logger.info("Fitting and Applying Target Encoders (Leakage-Free)...")
    fe = FeatureEngineer()
    
    # Fit and Apply K-Fold TE on Train (skip generation as we have it)
    df_train_enc = fe.fit_transform(df_train, encoding_only=True, skip_feature_generation=True)
    
    # Apply Global TE on Test
    df_test_enc = fe.transform(df_test, encoding_only=True)

    # 5. Define Features and Target
    future_info = [
        'rank', 'time', 'time_seconds', 'target', 'prize', 'log_prize',
        'passing_order', 'agari_3f', 'margin', 'margin_val',
        'corner_1コーナー', 'corner_2コーナー', 'corner_3コーナー', 'corner_4コーナー', 'race_laps',
        'running_style', 'position_gain', 'first_corner_pos', 'final_corner_pos',
        'odds', 'popularity', 'log_odds', 
        'is_win', 'is_place', 'is_show'
    ]
    
    meta_cols = [
        'race_id', 'date', 'date_dt', 'horse_name', 'jockey', 'trainer', 
        'horse_id', 'jockey_id', 'trainer_id', 'owner', 'owner_id', 
        'sire_id', 'dam_id', 'bms_id', 'breeder_id', 'year', 'place_code',
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
    
    # Final numeric check
    final_feature_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df_train_enc[c]):
            final_feature_cols.append(c)
            
    logger.info(f"Final Feature Count: {len(final_feature_cols)}")
    
    X_train = df_train_enc[final_feature_cols]
    y_train = df_train_enc['target']
    X_test = df_test_enc[final_feature_cols]
    y_test = df_test_enc['target']

    # 4. Hyperparameter Tuning with Optuna
    # Check if best_params already exist
    best_params_path = os.path.join(model_dir, 'best_params.json')
    
    if os.path.exists(best_params_path):
        logger.info(f"Loading best params from {best_params_path}...")
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
    else:
        # Use subset of Train for speed
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
        with open(best_params_path, 'w') as f:
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
    
    logger.info(f"Evaluation Results (2024+) - Accuracy: {acc:.4f}, LogLoss: {loss:.4f}, AUC: {auc:.4f}")

    # 6.5 Visualize Calibration
    from src.utils.visualize import plot_calibration_curve
    plot_calibration_curve(y_test, y_pred_prob, output_path='reports/figures/calibration_curve.png')

    # 7. Retrain on FULL Dataset for Production
    logger.info("=== Retraining on FULL Dataset (All Years) for Production Model ===")
    
    # We must apply K-Fold TE to the entire dataset to use it all for training without leakage
    fe_full = FeatureEngineer()
    # skip_feature_generation=True because df_generated has raw features
    df_full_enc = fe_full.fit_transform(df_generated, encoding_only=True, skip_feature_generation=True)
    
    X_full = df_full_enc[final_feature_cols]
    y_full = df_full_enc['target']
    
    final_production_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
    final_production_model.fit(X_full, y_full)
    
    model = final_production_model # Update model reference to the full one

    # 8. Save Model
    model_path = os.path.join(model_dir, 'lgbm_calibrated.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Saved model to {model_path}")
    
    # Also save the FeatureEngineer (fitted on full data)
    fe_path = os.path.join(model_dir, 'feature_engineer.pkl')
    with open(fe_path, 'wb') as f:
        pickle.dump(fe_full, f)
    logger.info(f"Saved FeatureEngineer to {fe_path}")

if __name__ == "__main__":
    train_model()
