# ---------- src/config.py ----------
"""
config.py
Global configuration file storing hyperparameters and model settings
"""

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV = 5  # Cross-validation folds

PARAMS = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10],
        "penalty": ["l2"],
        "solver": ["liblinear"]
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "class_weight": ["balanced"]
    },
    "xgboost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
        "learning_rate": [0.1, 0.3]
    },
   "lightgbm": {
    "n_estimators": [100, 200],
    "num_leaves": [31, 50],
    "learning_rate": [0.1, 0.3],
    "verbose": [-1],
    "force_row_wise": [True]
    },
    "catboost": {
        "iterations": [100, 200],
        "depth": [3, 6],
        "learning_rate": [0.1, 0.3]
    }
}