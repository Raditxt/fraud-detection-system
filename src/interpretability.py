"""Model interpretability using SHAP (SHapley Additive exPlanations).

Answers the question a risk analyst or auditor will always ask: "why was
this specific transaction flagged?" Global feature importance (e.g. from
Random Forest's built-in .feature_importances_) only shows which features
matter on average across the whole dataset — it can't explain a single
prediction. SHAP values decompose each individual prediction into the
contribution of each feature, which is what's actually needed for case-level
review and regulatory explainability.

Uses TreeExplainer, which is exact (not approximate) for tree-based models
like XGBoost, and fast because it exploits tree structure directly.
"""

from pathlib import Path

import joblib
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def compute_shap_values(model, X: pd.DataFrame):
    """Compute SHAP values for a tree-based model.

    Args:
        model: Trained tree-based classifier (e.g. XGBoost, Random Forest).
        X: Feature DataFrame to explain.

    Returns:
        shap.Explanation object containing per-transaction, per-feature
        contributions to the fraud prediction.
    """
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def get_global_feature_importance(shap_values, X: pd.DataFrame) -> pd.Series:
    """Rank features by mean absolute SHAP value (global importance).

    Args:
        shap_values: Output of compute_shap_values.
        X: The same feature DataFrame passed to compute_shap_values.

    Returns:
        Series of features sorted by importance, descending.
    """
    importance = pd.Series(
        abs(shap_values.values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    return importance


def explain_single_transaction(shap_values, X: pd.DataFrame, row_position: int) -> pd.DataFrame:
    """Break down why a single transaction received its fraud risk score.

    Args:
        shap_values: Output of compute_shap_values.
        X: The same feature DataFrame passed to compute_shap_values.
        row_position: Integer position (0-indexed) of the transaction to explain.

    Returns:
        DataFrame of feature values and their SHAP contribution, sorted by
        absolute impact, descending.
    """
    contributions = pd.DataFrame({
        "feature": X.columns,
        "value": X.iloc[row_position].values,
        "shap_contribution": shap_values.values[row_position],
    })
    contributions["abs_contribution"] = contributions["shap_contribution"].abs()
    return contributions.sort_values("abs_contribution", ascending=False).drop(
        columns="abs_contribution"
    )


if __name__ == "__main__":
    import joblib as jl

    from src.risk_scoring import load_model, score_transactions

    X_test, y_test = jl.load(MODELS_DIR / "test_set.pkl")
    model = load_model("xgboost.pkl")

    # Use a sample for speed — SHAP on the full 56,962-row test set is slow.
    # A random sample is still representative for global importance ranking.
    sample_X = X_test.sample(n=2000, random_state=42)

    print("Computing SHAP values (this may take a minute)...")
    shap_values = compute_shap_values(model, sample_X)

    print("\nGlobal feature importance (top 10):")
    importance = get_global_feature_importance(shap_values, sample_X)
    print(importance.head(10))

    # Explain the single highest-risk fraud transaction in the sample
    risk_scores = score_transactions(model, sample_X)
    highest_risk_position = risk_scores.values.argmax()
    actual_label = y_test.loc[sample_X.index[highest_risk_position]]

    print(f"\nExplaining highest-risk transaction in sample (actual label: "
          f"{'Fraud' if actual_label == 1 else 'Normal'}):")
    explanation = explain_single_transaction(shap_values, sample_X, highest_risk_position)
    print(explanation.head(10).to_string(index=False))