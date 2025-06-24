# ---------- src/hyperparameter_tuning.py ----------
"""
Provides a helper function for hyperparameter tuning of a single model using GridSearchCV.

This module is useful for manual or custom tuning outside the main pipeline.
"""

from sklearn.model_selection import GridSearchCV
import logging

def tune_model(model, param_grid: dict, X, y, cv: int = 5) -> tuple:
    """
    Performs grid search cross-validation to find the best hyperparameters for a given model.
    Args:
        model: The scikit-learn compatible model to tune.
        param_grid (dict): Dictionary of hyperparameters to search.
        X: Training features (array or DataFrame).
        y: Training labels (array or Series).
        cv (int): Number of cross-validation folds (default: 5).
    Returns:
        tuple: (best_estimator, best_params) where best_estimator is the trained model and best_params is the best parameter set found.
    """
    search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_