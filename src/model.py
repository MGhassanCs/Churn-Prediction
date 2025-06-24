# src/model.py
"""
model.py
Loads and runs the best saved model (after selection)
"""
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd # Added for type hinting
from src.config import RANDOM_STATE, TEST_SIZE # Added TEST_SIZE
from src.data_preprocessing import clean_data, encode_features, load_data
from src.evaluate import evaluate_model
import logging

def run_best_model(filepath: str, model_path: str) -> None:
    """
    Loads data, preprocesses it, loads a saved model, and evaluates it.

    Args:
        filepath (str): Path to the input CSV data file.
        model_path (str): Path to the saved model (.joblib) file.
    Returns:
        None
    """
    df = load_data(filepath)
    df = clean_data(df)
    
    # IMPORTANT: The current encode_features fits a new scaler.
    # For robust production use, you would need to save the scaler
    # from the training process and use its .transform() method here.
    # For this project's current scope (re-evaluating on the original test set),
    # this will work, but it's a simplification for deployment.
    df_processed = encode_features(df)

    X = df_processed.drop(columns=['Churn'])
    y = df_processed['Churn']

    # Splitting data again to get the same test set as during training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE, stratify=y)

    model = joblib.load(model_path)
    logging.info(f"Loaded model from: {model_path}")
    evaluate_model(model, X_test, y_test)