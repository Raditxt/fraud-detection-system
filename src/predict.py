"""Prediction utilities for classifying new transactions as fraud or normal."""

from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_trained_model(filename: str = "random_forest.pkl"):
    """Load a trained model from disk.

    Args:
        filename: Model filename in the models/ directory.
            Defaults to the Random Forest model (best precision).

    Returns:
        The deserialized model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run 'python -m src.train' first."
        )
    return joblib.load(path)


def predict_transaction(model, transaction: pd.DataFrame) -> dict:
    """Predict whether a single transaction is fraudulent.

    Args:
        model: Trained classifier with predict/predict_proba methods.
        transaction: DataFrame with a single row of preprocessed features
            (same columns used during training, i.e. V1-V28, scaled_amount,
            scaled_time).

    Returns:
        Dictionary with the predicted label and fraud probability.
    """
    prediction = model.predict(transaction)[0]
    probability = model.predict_proba(transaction)[0][1]

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    model = load_trained_model()

    # Example: reuse one row from the saved test set for a quick sanity check
    X_test, y_test = joblib.load(MODELS_DIR / "test_set.pkl")
    sample = X_test.iloc[[0]]
    actual_label = y_test.iloc[0]

    result = predict_transaction(model, sample)
    print(f"Prediction: {result}")
    print(f"Actual label: {'Fraud' if actual_label == 1 else 'Normal'}")