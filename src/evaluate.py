"""Model evaluation utilities for fraud detection."""

from pathlib import Path

import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model(filename: str):
    """Load a saved model or object from the models/ directory.

    Args:
        filename: Filename of the saved artifact, e.g. 'random_forest.pkl'.

    Returns:
        The deserialized object.
    """
    return joblib.load(MODELS_DIR / filename)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Print and return evaluation metrics for a trained model.

    Uses precision, recall, F1, and ROC-AUC rather than accuracy,
    since accuracy is misleading on this heavily imbalanced dataset
    (99.8% of transactions are non-fraud).

    Args:
        model: Trained scikit-learn model.
        X_test: Test features.
        y_test: Test labels.
        model_name: Label used in printed output.

    Returns:
        Dictionary with key metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'=' * 50}")
    print(f"Evaluation: {model_name}")
    print(f"{'=' * 50}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negative: {cm[0][0]:>6} | False Positive: {cm[0][1]:>6}")
    print(f"  False Negative: {cm[1][0]:>5} | True Positive: {cm[1][1]:>6}")

    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC Score: {auc:.4f}")

    return {
        "model_name": model_name,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
    }


def run_evaluation():
    """Load saved models and the test set, then evaluate both models."""
    X_test, y_test = load_model("test_set.pkl")

    log_reg = load_model("logistic_regression.pkl")
    evaluate_model(log_reg, X_test, y_test, model_name="Logistic Regression")

    rf = load_model("random_forest.pkl")
    evaluate_model(rf, X_test, y_test, model_name="Random Forest")


if __name__ == "__main__":
    run_evaluation()