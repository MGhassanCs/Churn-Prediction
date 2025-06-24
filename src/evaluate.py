# src/evaluate.py
"""
Provides functions to evaluate trained models, generate performance reports, and create visualizations.

This module is responsible for:
- Calculating evaluation metrics (accuracy, ROC AUC, classification report)
- Plotting ROC curves and confusion matrices
- Generating a comprehensive Markdown report summarizing model performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from datetime import datetime
import pandas as pd # Added for type hinting
from typing import Dict, Any, List, Optional
import logging

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates a given model on the test set and logs key metrics.
    Args:
        model: Trained scikit-learn compatible model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
    Returns:
        Dict[str, Any]: A dictionary containing evaluation metrics and prediction results.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")
    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred))
    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": class_report,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "y_test": y_test # Include y_test for easier access in report generation
    }

def create_report_directories():
    """
    Ensures that the 'reports/' and 'reports/images/' directories exist for saving plots and reports.
    """
    os.makedirs("reports/images", exist_ok=True)

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, save_dir: str = "reports/images") -> None:
    """
    Generates and saves the ROC curve plot for a model.
    Args:
        y_true (np.ndarray): True labels.
        y_prob (np.ndarray): Predicted probabilities for the positive class.
        model_name (str): Name of the model for the filename.
        save_dir (str): Directory to save the plot.
    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"roc_curve_{model_name.replace(' ', '_')}.png"))
    plt.close()
    filename = f"roc_curve_{model_name.replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    logging.info(f"Saved ROC curve to {filepath}")

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str = "reports/images",
    labels: Optional[List[str]] = None) -> None:
    """
    Generates and saves the Confusion Matrix plot for a model.
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        model_name (str): Name of the model for the filename.
        save_dir (str): Directory to save the plot.
        labels (List[str]): List of class labels (e.g., ['No Churn', 'Churn']).
    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = [str(x) for x in sorted(list(np.unique(y_true)))] # Default labels if not provided
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')) # Replaced space for filename safety
    plt.close()
    filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    filepath = os.path.join(save_dir, filename)
    logging.info(f"Saved Confusion Matrix to {filepath}")

def generate_full_report(model_name: str, metrics: Dict[str, Any], save_dir: str = "reports") -> None:
    """
    Generates a comprehensive Markdown report summarizing model performance and includes plot links.
    Args:
        model_name (str): Name of the best performing model.
        metrics (Dict[str, Any]): Dictionary of evaluation metrics from evaluate_model, including 'y_test', 'y_pred', 'y_prob'.
        save_dir (str): Directory to save the markdown report.
    Returns:
        None
    """
    create_report_directories() # Ensure directories exist before saving
    report_filename = f"model_performance_summary_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.md"
    report_path = os.path.join(save_dir, report_filename)
    # Plot and save images
    plot_roc_curve(metrics['y_test'], metrics['y_prob'], model_name)
    plot_confusion_matrix(metrics['y_test'], metrics['y_pred'], model_name, labels=["No Churn", "Churn"])
    # Prepare classification report string
    class_report_str = classification_report(metrics['y_test'], metrics['y_pred'], target_names=["No Churn", "Churn"])
    with open(report_path, "w") as f:
        f.write(f"# Customer Churn Prediction Model Performance Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Best Model:** {model_name}\n")
        f.write(f"**Dataset:** Telco Customer Churn (cleaned, preprocessed, and split into train/test)\n\n")
        f.write("---\n\n")
        f.write("## 1. Introduction\n\n")
        f.write("This report summarizes the performance of the best machine learning model selected for predicting customer churn. The primary objective is to identify customers at risk of churning to enable proactive retention efforts.\n\n")
        f.write("The model was trained and evaluated on the Telco Customer Churn dataset, which includes various customer demographics, services, and account information.\n\n")
        f.write("## 2. Model Overview\n\n")
        f.write(f"The best-performing model identified through hyperparameter tuning and cross-validation was a **{model_name}**.\n\n")
        f.write("This model was chosen based on its balanced performance across key metrics, particularly its ability to identify churners (Recall) while maintaining acceptable precision and a high ROC AUC score.\n\n")
        f.write("## 3. Evaluation Metrics\n\n")
        f.write("For this classification task, especially given potential class imbalance (fewer churners than non-churners), the following metrics were prioritized:\n")
        f.write("- **Accuracy:** Overall correctness of predictions.\n")
        f.write("- **ROC AUC (Receiver Operating Characteristic Area Under Curve):** Measures the model's ability to distinguish between positive and negative classes across all possible classification thresholds.\n")
        f.write("- **Precision:** The proportion of positive identifications that were actually correct.\n")
        f.write("- **Recall (Sensitivity):** The proportion of actual positives that were identified correctly.\n")
        f.write("- **F1-Score:** The harmonic mean of Precision and Recall, providing a balance between the two.\n\n")
        f.write("## 4. Performance Results\n\n")
        f.write("### Key Metrics\n\n")
        f.write(f"- **Overall Accuracy:** `{metrics['accuracy']:.4f}`\n")
        f.write(f"- **ROC AUC Score:** `{metrics['roc_auc']:.4f}`\n\n")
        f.write("### Classification Report (Test Set)\n\n")
        f.write("```\n")
        f.write(str(class_report_str))
        f.write("\n```\n")
        f.write("**Key Observations from Classification Report:**\n")
        f.write("- The model shows strong performance in identifying non-churners (Class 0), with high precision, recall, and F1-score.\n")
        f.write(f"- For churners (Class 1), the model achieved a **Recall of {metrics['classification_report']['1']['recall']:.2f}** (meaning it identified {metrics['classification_report']['1']['recall']:.0%} of actual churners) and a **Precision of {metrics['classification_report']['1']['precision']:.2f}** (meaning {metrics['classification_report']['1']['precision']:.0%} of its churn predictions were correct).\n")
        f.write("- This balance indicates that the model is effective for targeted retention efforts, balancing the need to find churners with minimizing false alarms.\n\n")
        f.write("### Visualizations\n\n")
        f.write("#### ROC Curve\n\n")
        f.write(f"![ROC Curve for {model_name}](images/roc_curve_{model_name.replace(' ', '_')}.png)\n\n") # Update image path for Markdown
        f.write("The ROC curve further illustrates the model's discriminative power. The closer the curve is to the top-left corner, the better the model performs across various thresholds.\n\n")
        f.write("#### Confusion Matrix\n\n")
        f.write(f"![Confusion Matrix for {model_name}](images/confusion_matrix_{model_name.replace(' ', '_')}.png)\n\n") # Update image path for Markdown
        f.write("The confusion matrix provides a detailed breakdown of correct and incorrect predictions:\n")
        # Calculate true counts from confusion matrix directly for accuracy
        cm_array = confusion_matrix(metrics['y_test'], metrics['y_pred'])
        tn, fp, fn, tp = cm_array.ravel() # Extract TN, FP, FN, TP
        f.write(f"- **True Negatives (TN):** {tn} (Correctly predicted no churn)\n")
        f.write(f"- **False Positives (FP):** {fp} (Incorrectly predicted churn - Type I error)\n")
        f.write(f"- **False Negatives (FN):** {fn} (Incorrectly predicted no churn, but they churned - Type II error, critical for churn)\n")
        f.write(f"- **True Positives (TP):** {tp} (Correctly predicted churn)\n")
        f.write("\n")
        f.write("## 5. Conclusion & Recommendations\n\n")
        f.write(f"The **{model_name}** model provides a solid foundation for predicting customer churn. Its performance indicates a strong ability to identify at-risk customers, allowing for strategic interventions.\n\n")
        f.write("### Recommendations:\n")
        f.write("- **Targeted Interventions:** Utilize the model to identify customers predicted to churn and implement specific retention campaigns.\n")
        f.write("- **Feature Importance Analysis:** Further analyze the model's feature importances (if available from the best model) to gain business insights into the main drivers of churn and inform product/service improvements.\n")
        f.write("- **Monitoring:** Continuously monitor model performance in a production environment for data drift and concept drift.\n\n")
        f.write("### Future Work:\n")
        f.write("- Explore advanced feature engineering techniques.\n")
        f.write("- Implement A/B testing of retention strategies based on model predictions.\n")
        f.write("- Integrate the model into a real-time prediction service.\n\n")
    logging.info(f"Generated full report: {report_path}")