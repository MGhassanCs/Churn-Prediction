"""
Provides functions to compare multiple machine learning models and select the best one based on ROC AUC score.

This module is useful for benchmarking different algorithms and hyperparameters.
"""

from src.hyperparameter_tuning import (
    tune_logistic_regression,
    tune_random_forest,
    tune_xgboost,
    tune_lightgbm,
    tune_catboost,
)

from sklearn.metrics import roc_auc_score
import pandas as pd
import logging


def compare_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> list:
    """
    Trains and evaluates multiple models with tuned hyperparameters.
    Returns a list of results for each model, including its name, ROC AUC score, best parameters, and the trained model object.

    Args:
        X_train, X_test: Training and test features.
        y_train, y_test: Training and test labels.
    Returns:
        list: Each item is a dict with model details and performance.
    """
    results = []
    # Dictionary mapping model names to their tuning functions
    models = {
        "Logistic Regression": tune_logistic_regression,
        "Random Forest": tune_random_forest,
        "XGBoost": tune_xgboost,
        "LightGBM": tune_lightgbm,
        "CatBoost": tune_catboost
    }
    for name, tuner_func in models.items():
        logging.info(f"🔍 Tuning {name}...")
        model, best_params = tuner_func(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        logging.info(f"✅ {name} AUC: {auc:.4f}")
        results.append({
            "Model": name,
            "ROC AUC": auc,
            "Best Params": best_params,
            "Trained Model": model
        })
    return results


def get_best_model(results: list) -> dict:
    """
    Selects the model with the highest ROC AUC score from the results.

    Args:
        results (list): List of dicts returned by compare_models()
    Returns:
        dict: Best model's full details (name, ROC AUC, params, model object)
    """
    best = max(results, key=lambda x: x["ROC AUC"])
    logging.info(f"🏆 Best model: {best['Model']} with AUC {best['ROC AUC']:.4f}")
    return best
