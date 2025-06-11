#evaluate
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the classification model on test data.

    Args:
        model: Trained classification model with predict and predict_proba methods.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for test set.

    Returns:
        tuple: (predicted labels, predicted probabilities for positive class)
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(report)

    return y_pred, y_prob

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot confusion matrix heatmap.

    Args:
        y_test (pd.Series): True labels.
        y_pred (np.array): Predicted labels.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(model, X_test, y_test):
    """
    Plot the ROC curve and display AUC.

    Args:
        model: Trained classification model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels.
    """
    from sklearn.metrics import roc_curve, auc

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
