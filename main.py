# main.py
"""
Entrypoint for the Churn Prediction project.

This script provides a command-line interface (CLI) to:
- Train and select the best model (from several ML algorithms)
- Evaluate a previously saved model (without retraining)

Usage examples:
    python main.py train --data data/telco.csv
    python main.py evaluate --data data/telco.csv --model models/catboost_model.joblib

If --model is omitted in 'evaluate', the most recent model in 'models/' will be used.
"""

import logging
import argparse
import os
from src.model_selection import train_and_select_best_model
from src.model import run_best_model

if __name__ == "__main__":
    # Set up logging for informative output
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set up the command-line interface
    parser = argparse.ArgumentParser(
        description="Churn Prediction Project CLI: Train or evaluate churn prediction models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for the 'train' command
    train_parser = subparsers.add_parser(
        "train",
        help="Train and select the best model. This will train all supported models, select the best based on ROC AUC, and save it."
    )
    train_parser.add_argument(
        "--data",
        type=str,
        default="data/telco.csv",
        help="Path to the data CSV file (default: data/telco.csv)"
    )

    # Subparser for the 'evaluate' command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved model on the test set. No retraining occurs."
    )
    eval_parser.add_argument(
        "--data",
        type=str,
        default="data/telco.csv",
        help="Path to the data CSV file (default: data/telco.csv)"
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the saved model (.joblib). If omitted, the most recent model in 'models/' will be used."
    )

    args = parser.parse_args()

    # Ensure the data and models directories exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logging.info(f"Created directory: {data_dir}")
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logging.info(f"Created directory: {models_dir}")

    # Handle the 'train' command
    if args.command == "train":
        # This will train all models, select the best, save it, and generate a report
        train_and_select_best_model(args.data)

    # Handle the 'evaluate' command
    elif args.command == "evaluate":
        # This will load a saved model and evaluate it on the test set (no retraining)
        model_path = args.model
        if model_path is None:
            # If no model is specified, use the most recent model in 'models/'
            import glob
            model_files = sorted(
                glob.glob(os.path.join(models_dir, "*_model.joblib")),
                key=os.path.getmtime,
                reverse=True
            )
            if not model_files:
                logging.error("No model files found in models/. Please train a model first or specify --model.")
                exit(1)
            model_path = model_files[0]
            logging.info(f"Using most recent model: {model_path}")
        run_best_model(args.data, model_path)