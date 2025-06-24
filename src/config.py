# src/config.py
"""
Stores global settings and hyperparameters for all models used in the churn prediction project.

This file centralizes configuration so you can easily change random seeds, test size, or model hyperparameters in one place.
"""

# Random seed for reproducibility
RANDOM_STATE = 42

# Fraction of data to use for the test set
TEST_SIZE = 0.2

# Number of cross-validation folds
CV = 5

# Hyperparameter grids for each model (used in GridSearchCV)
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
        "learning_rate": [0.1, 0.3]
    },
    "catboost": {
        "iterations": [100, 200],
        "depth": [3, 6],
        "learning_rate": [0.1, 0.3]
    }
}