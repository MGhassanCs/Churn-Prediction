# ---------- src/adv_models.py ----------
"""
adv_models.py
Defines advanced models (XGBoost, LightGBM, CatBoost) for classification
"""

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_all_models():
    """Returns a dictionary of model name -> instantiated model."""
    return {
        "logistic_regression": LogisticRegression(max_iter=500),
        "random_forest": RandomForestClassifier(),
        "xgboost": XGBClassifier(eval_metric='logloss'),
        "lightgbm": LGBMClassifier(),
        "catboost": CatBoostClassifier(verbose=0)
    }