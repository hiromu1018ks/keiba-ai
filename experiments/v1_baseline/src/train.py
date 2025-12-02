import pandas as pd
import lightgbm as lgb
import os
import pickle
from sklearn.metrics import accuracy_score, log_loss
from experiments.v1_baseline.src.feature_engineering import FeatureEngineer
from common.src.utils.logger import setup_logger

logger = setup_logger(__name__)

def train_model(data_path='common/data/rawdf/results.csv', model_dir='models'):
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

    # 4. Hyperparameter Tuning with Optuna
    import optuna
    import json
    from lightgbm import LGBMClassifier
    
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
        
        # Use simple train/test split for tuning speed
        # In production, use CV
        model = LGBMClassifier(**param)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        preds = model.predict_proba(X_test)[:, 1]
        return log_loss(y_test, preds)

    logger.info("Starting Optuna optimization...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) # 20 trials for demo speed
    
    best_params = study.best_params
    logger.info(f"Best params: {best_params}")
    
    # Save best params
    with open(os.path.join(model_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)

    # 5. Train Final Calibrated Model with Best Params
    from sklearn.calibration import CalibratedClassifierCV
    from lightgbm import LGBMClassifier

    # Base LightGBM Model with Best Params
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

    # 5. Evaluate
    # CalibratedClassifierCV returns probabilities for both classes [prob_0, prob_1]
    y_pred_prob = model.predict_proba(X_test)[:, 1]
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
