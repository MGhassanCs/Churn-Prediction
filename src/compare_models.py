from src.hyperparameter_tuning import (
    tune_logistic_regression,
    tune_random_forest,
    tune_xgboost,
    tune_lightgbm,
    tune_catboost,
)

from sklearn.metrics import roc_auc_score
import pandas as pd


def compare_models(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates multiple models with tuned hyperparameters.
    Returns the best model based on ROC AUC score.

    Returns:
        dict: Contains each model name, AUC score, best params, and the model object.
    """
    results = []

    models = {
        "Logistic Regression": tune_logistic_regression,
        "Random Forest": tune_random_forest,
        "XGBoost": tune_xgboost,
        "LightGBM": tune_lightgbm,
        "CatBoost": tune_catboost
    }

    for name, tuner_func in models.items():
        print(f"\n🔍 Tuning {name}...")
        model, best_params = tuner_func(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"✅ {name} AUC: {auc:.4f}")

        results.append({
            "Model": name,
            "ROC AUC": auc,
            "Best Params": best_params,
            "Trained Model": model
        })

    return results


def get_best_model(results):
    """
    Selects the model with the highest ROC AUC score from the results.

    Args:
        results (list): List of dicts returned by compare_models()

    Returns:
        dict: Best model's full details
    """
    best = max(results, key=lambda x: x["ROC AUC"])
    print(f"\n🏆 Best model: {best['Model']} with AUC {best['ROC AUC']:.4f}")
    return best
