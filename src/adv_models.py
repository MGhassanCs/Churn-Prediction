# src/adv_models.py
"""
Defines advanced machine learning models (XGBoost, LightGBM, CatBoost, etc.)
and provides functions to train them with hyperparameter tuning.

This module is used by the main training pipeline to instantiate and tune models.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np # Added for type hinting and potential future use
import logging

# Import configuration from src.config
from src.config import RANDOM_STATE, PARAMS, CV # Ensure these are defined in src/config.py

def get_all_base_models() -> dict:
    """
    Returns a dictionary mapping model names to their base (un-tuned) scikit-learn compatible model objects.
    This makes it easy to instantiate all supported models in a consistent way.
    """
    return {
        "logistic_regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=500),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "xgboost": XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
        "lightgbm": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        "catboost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    }

def train_model_with_tuning(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Trains a specified model using GridSearchCV for hyperparameter tuning.

    Args:
        model_name (str): The name of the model to train (must be a key in PARAMS).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        tuple: (best_estimator, best_score) where best_estimator is the trained model and best_score is the best ROC AUC.
    Raises:
        ValueError: If an unsupported model name is provided or parameters are missing.
    """
    base_models = get_all_base_models()
    if model_name not in base_models:
        raise ValueError(f"Unsupported model name: {model_name}")
    if model_name not in PARAMS:
        raise ValueError(f"No tuning parameters defined for model: {model_name} in src/config.py")
    model = base_models[model_name]
    param_grid = PARAMS[model_name]
    # Use GridSearchCV to find the best hyperparameters for this model
    logging.info(f"Starting GridSearchCV for {model_name}...")
    grid = GridSearchCV(model, param_grid, cv=CV, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    logging.info(f"Best parameters for {model_name}: {grid.best_params_}")
    return grid.best_estimator_, grid.best_score_