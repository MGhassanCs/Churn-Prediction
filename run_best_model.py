"""
run_best_model.py
Loads the saved best model and evaluates it on the specified dataset.
"""

import joblib
from src.data_preprocessing import load_data, clean_data, encode_features
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve

def run_best_model(data_path, model_path):
    """
    Load data, preprocess it, load a saved model, and evaluate the model.

    Args:
        data_path (str): Path to the CSV dataset.
        model_path (str): Path to the saved model file (.joblib).

    Returns:
        None
    """
    # Load and preprocess dataset
    df = load_data(data_path)
    df = clean_data(df)
    df = encode_features(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Load saved model
    model = joblib.load(model_path)

    # Evaluate the model on the data
    y_pred, y_prob = evaluate_model(model, X, y)

    # Plot evaluation visuals
    plot_confusion_matrix(y, y_pred)
    plot_roc_curve(y, y_prob)

if __name__ == "__main__":
    # Make sure this matches the model you saved during training
    run_best_model("data/telco.csv", "models/catboost_model.joblib")
