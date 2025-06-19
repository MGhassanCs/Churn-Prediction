# ---------- src/hyperparameter_tuning.py ----------
"""
hyperparameter_tuning.py
Helper for tuning a single model manually
"""

from sklearn.model_selection import GridSearchCV

def tune_model(model, param_grid, X, y, cv=5):
    """
    Returns best model after performing grid search
    """
    search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_