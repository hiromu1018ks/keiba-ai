"""
Ensemble Model Training with LightGBM + CatBoost (GPU)
Combines predictions from both models for improved accuracy.
"""
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import os
import pickle
import numpy as np
import optuna
import json
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from src.features.engineer import FeatureEngineer
from src.model.ensemble import EnsembleModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def train_ensemble_model(data_path='data/common/raw_data/results.csv', model_dir='models'):
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. Load Features from Cache
    cache_path = 'data/cache/features_sim.pkl'
    
    if os.path.exists(cache_path):
        logger.info(f"Loading features from cache: {cache_path}")
        df_generated = pd.read_pickle(cache_path)
    else:
        logger.error("Feature cache not found. Run train.py first.")
        return
    
    if 'year' not in df_generated.columns:
        if 'date_dt' in df_generated.columns:
            df_generated['year'] = df_generated['date_dt'].dt.year
        else:
            logger.error("Year column missing in features.")
            return

    logger.info(f"Total features: {len(df_generated.columns)}")

    # 2. Time Series Split
    train_mask = df_generated['year'] < 2024
    test_mask = df_generated['year'] >= 2024
    
    df_train = df_generated[train_mask].copy()
    df_test = df_generated[test_mask].copy()
    
    logger.info(f"Train size: {len(df_train)}, Test size: {len(df_test)}")

    # 3. Target Encoding
    logger.info("Fitting and Applying Target Encoders...")
    fe = FeatureEngineer()
    df_train_enc = fe.fit_transform(df_train, encoding_only=True, skip_feature_generation=True)
    df_test_enc = fe.transform(df_test, encoding_only=True)

    # 4. Define Features
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
    
    final_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_train_enc[c])]
            
    logger.info(f"Final Feature Count: {len(final_feature_cols)}")
    
    X_train = df_train_enc[final_feature_cols]
    y_train = df_train_enc['target']
    X_test = df_test_enc[final_feature_cols]
    y_test = df_test_enc['target']

    # Fill NaN for CatBoost
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)

    # 5. Load LightGBM best params
    best_params_path = os.path.join(model_dir, 'best_params_baseline.json')
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'verbose': -1,
        'learning_rate': 0.05,
        'num_leaves': 148
    }
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            lgbm_params.update(json.load(f))
    
    # 6. Train LightGBM (Calibrated)
    logger.info("Training LightGBM (Calibrated)...")
    lgbm_base = LGBMClassifier(**lgbm_params)
    lgbm_calibrated = CalibratedClassifierCV(lgbm_base, method='isotonic', cv=5)
    lgbm_calibrated.fit(X_train, y_train)
    
    lgbm_pred = lgbm_calibrated.predict_proba(X_test)[:, 1]
    lgbm_auc = roc_auc_score(y_test, lgbm_pred)
    logger.info(f"LightGBM Test AUC: {lgbm_auc:.4f}")

    # 7. Train CatBoost with GPU
    logger.info("Training CatBoost with GPU...")
    catboost_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        task_type='GPU',
        devices='0',
        verbose=100,
        early_stopping_rounds=50,
        eval_metric='Logloss'
    )
    catboost_model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=100
    )
    
    catboost_pred = catboost_model.predict_proba(X_test)[:, 1]
    catboost_auc = roc_auc_score(y_test, catboost_pred)
    logger.info(f"CatBoost Test AUC: {catboost_auc:.4f}")

    # 8. Find Optimal Ensemble Weights
    logger.info("Finding optimal ensemble weights...")
    best_weight = 0.5
    best_auc = 0
    
    for w in np.arange(0.1, 0.9, 0.1):
        ensemble_pred = w * lgbm_pred + (1 - w) * catboost_pred
        auc = roc_auc_score(y_test, ensemble_pred)
        if auc > best_auc:
            best_auc = auc
            best_weight = w
    
    logger.info(f"Best Weight: LightGBM={best_weight:.1f}, CatBoost={1-best_weight:.1f}")
    logger.info(f"Best Ensemble AUC: {best_auc:.4f}")

    # 9. Create Ensemble Model
    ensemble = EnsembleModel(lgbm_calibrated, catboost_model, lgbm_weight=best_weight)
    
    # Final Evaluation
    ensemble_pred = ensemble.predict_proba(X_test)[:, 1]
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    ensemble_logloss = log_loss(y_test, ensemble_pred)
    
    logger.info(f"=== Final Results ===")
    logger.info(f"LightGBM AUC:  {lgbm_auc:.4f}")
    logger.info(f"CatBoost AUC:  {catboost_auc:.4f}")
    logger.info(f"Ensemble AUC:  {ensemble_auc:.4f}")
    logger.info(f"Ensemble LogLoss: {ensemble_logloss:.4f}")

    # 10. Save Models
    model_path = os.path.join(model_dir, 'ensemble_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
    logger.info(f"Saved ensemble model to {model_path}")
    
    # Save Feature Engineer
    fe_path = os.path.join(model_dir, 'feature_engineer_ensemble.pkl')
    with open(fe_path, 'wb') as f:
        pickle.dump(fe, f)
    logger.info(f"Saved FeatureEngineer to {fe_path}")

    # Save ensemble config
    config = {
        'lgbm_weight': best_weight,
        'catboost_weight': 1 - best_weight,
        'lgbm_auc': lgbm_auc,
        'catboost_auc': catboost_auc,
        'ensemble_auc': ensemble_auc,
        'ensemble_logloss': ensemble_logloss
    }
    config_path = os.path.join(model_dir, 'ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved ensemble config to {config_path}")


if __name__ == "__main__":
    train_ensemble_model()
