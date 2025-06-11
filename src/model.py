#model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def split_data(df, target='Churn'):
    """
    Split the dataframe into train and test sets.
    
    Parameters:
    - df: pandas DataFrame with features and target
    - target: name of the target column
    
    Returns:
    - X_train, X_test, y_train, y_test: split datasets
    """
    X = df.drop(columns=[target])
    y = df[target]  # encode target
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_model(X_train, y_train):
    """
    Train a logistic regression model with increased max iterations.
    
    Parameters:
    - X_train: training features
    - y_train: training labels
    
    Returns:
    - trained logistic regression model
    """
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath='model.joblib'):
    """
    Save the trained model to disk.
    
    Parameters:
    - model: trained sklearn model
    - filepath: path to save the model file
    """
    joblib.dump(model, filepath)
