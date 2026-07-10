"""Preprocessing utilities for the fraud detection pipeline."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def scale_amount_column(df: pd.DataFrame) -> pd.DataFrame:
    """Scale the 'Amount' column using standardization.

    The PCA-transformed columns (V1-V28) are already scaled by design,
    but 'Amount' and 'Time' are raw and need scaling for models that are
    sensitive to feature magnitude (e.g. Logistic Regression).

    Args:
        df: DataFrame containing an 'Amount' column.

    Returns:
        A copy of the DataFrame with a new 'scaled_amount' column.
    """
    df = df.copy()
    scaler = StandardScaler()
    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])
    return df


def scale_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """Scale the 'Time' column using standardization.

    Args:
        df: DataFrame containing a 'Time' column.

    Returns:
        A copy of the DataFrame with a new 'scaled_time' column.
    """
    df = df.copy()
    scaler = StandardScaler()
    df["scaled_time"] = scaler.fit_transform(df[["Time"]])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline: scale Amount/Time, drop originals.

    Args:
        df: Raw DataFrame with 'Amount', 'Time', and 'Class' columns.

    Returns:
        Cleaned DataFrame ready for train/test split.
    """
    df = scale_amount_column(df)
    df = scale_time_column(df)
    df = df.drop(columns=["Amount", "Time"])
    return df


def split_features_and_target(
    df: pd.DataFrame, target_col: str = "Class", test_size: float = 0.2, random_state: int = 42
):
    """Split a DataFrame into stratified train/test feature and target sets.

    Stratification ensures the fraud ratio stays consistent between
    train and test sets, which matters heavily for imbalanced data.

    Args:
        df: Preprocessed DataFrame including the target column.
        target_col: Name of the target column.
        test_size: Proportion of data to reserve for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )