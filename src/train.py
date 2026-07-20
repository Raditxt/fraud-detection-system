"""Model training pipeline for fraud detection."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier  # tambahan

from src.data_loader import load_raw_data
from src.preprocessing import prepare_features, split_features_and_target

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train_logistic_regression(X_train, y_train, random_state: int = 42) -> LogisticRegression:
    """Train a Logistic Regression baseline with class balancing.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        Trained LogisticRegression model.
    """
    model = LogisticRegression(
        class_weight="balanced",  # compensates for the 0.17% fraud ratio
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, random_state: int = 42) -> RandomForestClassifier:
    """Train a Random Forest classifier with class balancing.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        Trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state: int = 42) -> XGBClassifier:
    """Train an XGBoost classifier with class balancing.

    Uses scale_pos_weight instead of class_weight (XGBoost's API differs
    from scikit-learn's), computed as the ratio of negative to positive
    class counts to achieve equivalent balancing.

    Args:
        X_train: Training features.
        y_train: Training labels.
        random_state: Seed for reproducibility.

    Returns:
        Trained XGBClassifier model.
    """
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count

    model = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filename: str) -> Path:
    """Save a trained model to the models/ directory.

    Args:
        model: Trained scikit-learn model.
        filename: Output filename, e.g. 'random_forest.pkl'.

    Returns:
        Path where the model was saved.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    return path


def run_training_pipeline():
    """Load data, preprocess, train both models, and save them to disk."""
    print("Loading and preprocessing data...")
    df = load_raw_data()
    df = prepare_features(df)
    X_train, X_test, y_train, y_test = split_features_and_target(df)

    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    print("\nTraining Logistic Regression...")
    log_reg = train_logistic_regression(X_train, y_train)
    save_model(log_reg, "logistic_regression.pkl")
    print("Saved to models/logistic_regression.pkl")

    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train, y_train)
    save_model(rf, "random_forest.pkl")
    print("Saved to models/random_forest.pkl")

    print("\nTraining XGBoost...")
    xgb = train_xgboost(X_train, y_train)
    save_model(xgb, "xgboost.pkl")
    print("Saved to models/xgboost.pkl")

    # Save the test set too, so evaluate.py can reuse the exact same split
    save_model((X_test, y_test), "test_set.pkl")
    print("\nTraining complete. Test set saved for evaluation.")


if __name__ == "__main__":
    run_training_pipeline()