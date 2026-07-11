"""Risk scoring utilities: convert model output from binary class to
continuous fraud probability (risk score).

This is the foundation for cost-sensitive decision-making. A binary
prediction (0/1) throws away information — two transactions both
classified as "fraud" could have very different confidence levels
(0.51 vs 0.99). Keeping the raw probability lets downstream logic
(cost framework, threshold tuning, decision tiers) make better
trade-offs than a fixed 0.5 cutoff ever could.
"""

from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model(filename: str):
    """Load a trained model from the models/ directory.

    Args:
        filename: Model filename, e.g. 'random_forest.pkl'.

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


def score_transactions(model, X: pd.DataFrame) -> pd.Series:
    """Score transactions with a continuous fraud probability (risk score).

    Args:
        model: Trained classifier with a predict_proba method.
        X: Feature DataFrame (same columns used during training).

    Returns:
        Series of fraud probabilities in [0, 1], indexed like X.
    """
    probabilities = model.predict_proba(X)[:, 1]
    return pd.Series(probabilities, index=X.index, name="risk_score")


def summarize_risk_scores(risk_scores: pd.Series, y_true: pd.Series) -> pd.DataFrame:
    """Summarize risk score distribution, split by actual class.

    Useful sanity check: fraud transactions should generally cluster
    at higher risk scores than normal transactions. If they don't,
    the model isn't separating the classes well and threshold tuning
    later won't help much.

    Args:
        risk_scores: Output of score_transactions.
        y_true: Ground truth labels (0 = normal, 1 = fraud), same index as risk_scores.

    Returns:
        DataFrame with descriptive statistics of risk scores per class.
    """
    df = pd.DataFrame({"risk_score": risk_scores, "actual_class": y_true})
    return df.groupby("actual_class")["risk_score"].describe()


if __name__ == "__main__":
    X_test, y_test = load_model("test_set.pkl")
    model = load_model("random_forest.pkl")

    risk_scores = score_transactions(model, X_test)

    print("Risk score distribution by actual class:")
    print(summarize_risk_scores(risk_scores, y_test))

    print("\nSample of highest-risk transactions:")
    top_risk = risk_scores.sort_values(ascending=False).head(10)
    for idx, score in top_risk.items():
        actual = "Fraud" if y_test.loc[idx] == 1 else "Normal"
        print(f"  Index {idx}: risk_score={score:.4f} | actual={actual}")