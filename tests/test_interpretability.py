"""Unit tests for SHAP-based interpretability utilities."""

import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.interpretability import (
    compute_shap_values,
    explain_single_transaction,
    get_global_feature_importance,
)


@pytest.fixture
def dummy_model_and_data():
    """A minimal trained XGBoost model for testing, independent of the real dataset."""
    X = pd.DataFrame({
        "feature_a": [0.1, 5.0, 0.2, 4.8, 0.15, 4.9, 0.05, 5.1],
        "feature_b": [0.0, 3.0, -0.1, 2.9, 0.05, 3.1, -0.05, 2.95],
    })
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    model = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")
    model.fit(X, y)
    return model, X


def test_compute_shap_values_matches_feature_count(dummy_model_and_data):
    model, X = dummy_model_and_data
    shap_values = compute_shap_values(model, X)

    assert shap_values.values.shape == X.shape


def test_get_global_feature_importance_returns_all_features(dummy_model_and_data):
    model, X = dummy_model_and_data
    shap_values = compute_shap_values(model, X)
    importance = get_global_feature_importance(shap_values, X)

    assert set(importance.index) == set(X.columns)
    assert (importance >= 0).all()  # mean absolute values are non-negative


def test_explain_single_transaction_returns_all_features(dummy_model_and_data):
    model, X = dummy_model_and_data
    shap_values = compute_shap_values(model, X)
    explanation = explain_single_transaction(shap_values, X, row_position=0)

    assert len(explanation) == X.shape[1]
    assert set(explanation.columns) == {"feature", "value", "shap_contribution"}


def test_explain_single_transaction_sorted_by_absolute_impact(dummy_model_and_data):
    model, X = dummy_model_and_data
    shap_values = compute_shap_values(model, X)
    explanation = explain_single_transaction(shap_values, X, row_position=0)

    abs_contributions = explanation["shap_contribution"].abs().values
    assert list(abs_contributions) == sorted(abs_contributions, reverse=True)