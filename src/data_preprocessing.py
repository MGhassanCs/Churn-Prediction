#data_preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset with shape: {df.shape}")
    return df

def clean_data(df):
    """
    Perform initial cleaning:
    - Convert 'TotalCharges' to numeric, coercing errors (turn invalid to NaN).
    - Drop rows where 'TotalCharges' is missing.
    - Drop 'customerID' column as it's an identifier and not useful for modeling.
    - Convert 'Churn' from categorical Yes/No to binary 1/0.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Convert TotalCharges to numeric, set errors to NaN for cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # Drop missing TotalCharges rows
    df = df.dropna(subset=['TotalCharges'])
    # Drop unique identifier column
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def encode_features(df):
    df = df.copy()

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # ⚠️ Remove target column 'Churn' from numerical columns before scaling
    if 'Churn' in numerical_cols:
        numerical_cols.remove('Churn')

    # Scale remaining numerical features
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df
