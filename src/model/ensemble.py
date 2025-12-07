"""
Ensemble Model class - shared between train and simulate
"""

class EnsembleModel:
    """Ensemble of LightGBM and CatBoost with weighted averaging."""
    
    def __init__(self, lgbm_model, catboost_model, lgbm_weight=0.5):
        self.lgbm_model = lgbm_model
        self.catboost_model = catboost_model
        self.lgbm_weight = lgbm_weight
        self.catboost_weight = 1 - lgbm_weight
    
    def predict_proba(self, X):
        lgbm_proba = self.lgbm_model.predict_proba(X)
        catboost_proba = self.catboost_model.predict_proba(X)
        
        # Weighted average
        ensemble_proba = (
            self.lgbm_weight * lgbm_proba + 
            self.catboost_weight * catboost_proba
        )
        return ensemble_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
