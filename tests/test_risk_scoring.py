"""Unit tests for risk scoring utilities."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.risk_scoring import score_transactions, summarize_risk_scores


@pytest.fixture
def dummy_model_and_data():
    """A minimal trained model + test set for testing, independent of the real dataset."""
    X_train = pd.DataFrame({
        "feature_1": [0.1, 5.0, 0.2, 4.8, 0.15, 4.9],
        "feature_2": [0.0, 3.0, -0.1, 2.9, 0.05, 3.1],
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1])
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_train, y_train


def test_score_transactions_returns_valid_probability_range(dummy_model_and_data):
    model, X, _ = dummy_model_and_data
    scores = score_transactions(model, X)

    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()


def test_score_transactions_preserves_index(dummy_model_and_data):
    model, X, _ = dummy_model_and_data
    scores = score_transactions(model, X)

    assert list(scores.index) == list(X.index)


def test_score_transactions_higher_for_positive_class(dummy_model_and_data):
    model, X, y = dummy_model_and_data
    scores = score_transactions(model, X)

    # Sanity check: on this clearly-separable dummy data, the model
    # should assign higher average risk to the fraud class.
    avg_fraud_score = scores[y == 1].mean()
    avg_normal_score = scores[y == 0].mean()
    assert avg_fraud_score > avg_normal_score


def test_summarize_risk_scores_returns_stats_per_class(dummy_model_and_data):
    model, X, y = dummy_model_and_data
    scores = score_transactions(model, X)
    summary = summarize_risk_scores(scores, y)

    assert set(summary.index) == {0, 1}
    assert "mean" in summary.columns