# ---------- src/model_selection.py ----------
"""
model_selection.py
Trains and compares all models, returning the best model
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from src.adv_models import get_all_models
from src.config import PARAMS, CV

def select_best_model(X_train, y_train, X_test, y_test):
    """
    Performs GridSearchCV for each model and selects the best based on ROC AUC.
    Returns the best trained model and its name.
    """
    best_score = 0
    best_model = None
    best_name = ""

    for name, model in get_all_models().items():
        print(f"\n🔍 Training {name}...")
        grid = GridSearchCV(model, PARAMS[name], cv=CV, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_prob = grid.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_prob)
        print(f"✅ {name} ROC AUC: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_name = name

    print(f"\n🏆 Best Model: {best_name} with ROC AUC: {best_score:.4f}")
    return best_model, best_name
