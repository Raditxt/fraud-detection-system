"""Data loading utilities for the fraud detection pipeline."""

from pathlib import Path
import pandas as pd

# Path relative to project root, so this works regardless of where the script is run from
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "creditcard.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw credit card transaction dataset.

    Args:
        path: Path to the CSV file. Defaults to data/raw/creditcard.csv.

    Returns:
        DataFrame containing raw transaction records.

    Raises:
        FileNotFoundError: If the dataset does not exist at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Make sure creditcard.csv is placed in data/raw/."
        )
    return pd.read_csv(path)


def get_class_distribution(df: pd.DataFrame) -> pd.Series:
    """Return the proportion of fraud (1) vs. non-fraud (0) transactions.

    Args:
        df: DataFrame with a 'Class' column (0 = normal, 1 = fraud).

    Returns:
        Series with normalized class counts.
    """
    return df["Class"].value_counts(normalize=True)


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
    print("\nClass distribution:")
    print(get_class_distribution(df))