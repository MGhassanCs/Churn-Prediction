# ---------- src/model.py ----------
"""
model.py
Defines the final best model for consistent use
"""

from sklearn.linear_model import LogisticRegression

def train_final_model(X_train, y_train):
    """Trains the final model selected as best performer."""
    model = LogisticRegression(max_iter=500, C=1.0, penalty="l2", solver="liblinear")
    model.fit(X_train, y_train)
    return model