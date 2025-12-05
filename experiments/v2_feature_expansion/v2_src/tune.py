import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from experiments.v2_feature_expansion.src.feature_engineering import FeatureEngineer
from common.src.utils.logger import setup_logger
import json
import os

logger = setup_logger(__name__)

def objective(trial):
    data_path = 'common/data/rawdf/results.csv'
    df = pd.read_csv(data_path)
    
    # Simple preprocessing for tuning (use a subset or full data?)
    # Let's use data < 2024 for tuning to avoid overfitting to the test set (2024-2025)
    # Or use a rolling validation.
    
    try:
        df['date_dt'] = pd.to_datetime(df['date'], format='%Y年%m月%d日')
        df['year'] = df['date_dt'].dt.year
    except:
        return 9999

    # Train on 2021-2022, Val on 2023
    train_df = df[df['year'].isin([2021, 2022])]
    val_df = df[df['year'] == 2023]
    
    if len(train_df) == 0 or len(val_df) == 0:
        return 9999

    fe = FeatureEngineer()
    train_X_all = fe.fit_transform(train_df)
    val_X_all = fe.transform(val_df)
    
    exclude_cols = ['race_id', 'date', 'time', 'rank', 'target', 'date_dt', 'year',
                    'horse_name', 'jockey', 'trainer', 'horse_id', 'jockey_id', 'trainer_id', 'time_seconds',
                    'prize', 'surface', 'distance', 'weather', 'condition', 'jockey_trainer_pair', 'winner_time',
                    'around', 'race_class', 'place', 'passing_order']
    
    feature_cols = [c for c in train_X_all.columns if c not in exclude_cols]
    
    X_train = train_X_all[feature_cols]
    y_train = train_X_all['target']
    X_val = val_X_all[feature_cols]
    y_val = val_X_all['target']
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': 1000
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=50)])
    
    preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)
    return loss

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    logger.info(f"Best trial: {study.best_trial.params}")
    
    # Save best params
    os.makedirs('models', exist_ok=True)
    with open('models/best_params.json', 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)
