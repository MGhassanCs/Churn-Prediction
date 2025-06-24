# src/model_selection.py
"""
Trains multiple models, selects and saves the best, and generates performance reports.

This module is the main orchestrator for the training pipeline:
- Loads and preprocesses the data
- Trains all supported models with hyperparameter tuning
- Selects the best model based on ROC AUC
- Saves the best model and generates a detailed report
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.adv_models import train_model_with_tuning
from src.evaluate import evaluate_model, generate_full_report
from src.data_preprocessing import clean_data, encode_features, load_data
from src.config import RANDOM_STATE, TEST_SIZE
import logging
from typing import Tuple

def train_and_select_best_model(filepath: str) -> None:
    """
    Loads data, preprocesses it, trains multiple models with tuning,
    selects the best model, saves it, and generates a performance report.

    Args:
        filepath (str): Path to the input CSV data file.
    Returns:
        None
    """
    # Load and preprocess the data
    df = load_data(filepath)
    df = clean_data(df)
    df_processed = encode_features(df) # Ensure this returns a DataFrame with 'Churn'
    X = df_processed.drop(columns=['Churn'])
    y = df_processed['Churn']
    # Split data into train and test sets
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    best_model = None
    best_score = -1.0 # Initialize with a value lower than any possible ROC AUC
    best_name = ""
    best_metrics_for_report = {} # To store detailed metrics for the best model
    # List of model names to iterate through (must match keys in PARAMS in config.py)
    model_names_to_train = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "catboost"]
    for model_name in model_names_to_train:
        logging.info(f"Training {model_name}...")
        try:
            # Train the model with hyperparameter tuning
            model, score = train_model_with_tuning(model_name, X_train, y_train)
            logging.info(f"{model_name} ROC AUC (tuned): {score:.4f}")
            # If this model is the best so far, update best_model
            if score > best_score:
                best_score = score
                best_model = model
                best_name = model_name
                # Evaluate this current best model to get all its metrics
                current_metrics = evaluate_model(best_model, X_test, y_test)
                best_metrics_for_report = current_metrics # Store for the final report
                best_metrics_for_report['y_test'] = y_test # Ensure y_test is explicitly available
        except Exception as e:
            logging.error(f"Error training {model_name}: {e}")
            continue # Continue to the next model if one fails
    if best_model:
        logging.info(f"Best Model Overall: {best_name} with ROC AUC: {best_score:.4f}")
        model_path = f"models/{best_name}_model.joblib"
        joblib.dump(best_model, model_path)
        logging.info(f"Saved best model to {model_path}")
        # Generate the full report for the best model
        generate_full_report(best_name, best_metrics_for_report)
    else:
        logging.error("No models were successfully trained or selected.")