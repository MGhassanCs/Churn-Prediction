# ---------- src/data_preprocessing.py ----------
"""
data_preprocessing.py
Module for loading, cleaning, encoding the Telco dataset
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Loads CSV data from the given path."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df

def clean_data(df):
    """Cleans dataset by handling missing values and converting types."""
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def encode_features(df):
    """Encodes categorical features and scales numerical ones."""
    df = df.copy()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'Churn' in num_cols:
        num_cols.remove('Churn')

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df