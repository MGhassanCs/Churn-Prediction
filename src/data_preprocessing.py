# src/data_preprocessing.py
"""
Handles loading, cleaning, and feature engineering for the churn prediction dataset.

This module provides functions to:
- Load the raw CSV data
- Clean and preprocess the data (handle missing values, drop unnecessary columns, encode target)
- Encode categorical features and scale numerical features
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import logging

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(filepath)
    logging.info(f"Loaded dataset with shape: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input DataFrame by:
    - Converting 'TotalCharges' to numeric (coercing errors to NaN)
    - Dropping rows with missing 'TotalCharges'
    - Dropping the 'customerID' column (not useful for modeling)
    - Encoding the 'Churn' column as 1 (Yes) and 0 (No)
    Args:
        df (pd.DataFrame): Raw data.
    Returns:
        pd.DataFrame: Cleaned data.
    """
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df = df.drop(columns=['customerID'], errors='ignore')
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features using one-hot encoding and scales numerical features.
    - Categorical columns are converted to dummy/indicator variables.
    - Numerical columns are standardized (mean=0, std=1), except for the target 'Churn'.
    Args:
        df (pd.DataFrame): Cleaned data.
    Returns:
        pd.DataFrame: Processed data with encoded and scaled features.
    """
    df = df.copy()
    # Identify categorical and numerical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Churn' in num_cols:
        num_cols.remove('Churn') # Ensure 'Churn' is not scaled
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    scaler = StandardScaler()
    # Only scale if there are numeric columns
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        # Clean up column names for compatibility
        df.columns = [col.replace(' ', '_').replace('(', '').replace(')', '').replace('<', '_less_than_') for col in df.columns]
    return df