"""
main.py
Train and select the best model, save it for future use.
"""

from src.data_preprocessing import load_data, clean_data, encode_features
from src.model_selection import select_best_model
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve
from src.config import RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split
import joblib
import os

def main():
    df = load_data("data/telco.csv")
    df = clean_data(df)
    df = encode_features(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # Train and select best model
    best_model, name = select_best_model(X_train, y_train, X_test, y_test)

    # Save the best model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{name}_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Saved best model to {model_path}")

    # Evaluate on test set
    y_pred, y_prob = evaluate_model(best_model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)

if __name__ == "__main__":
    main()
