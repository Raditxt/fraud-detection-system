"""Unit tests for prediction utilities."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.predict import predict_transaction


@pytest.fixture
def dummy_model():
    """A minimal trained model for testing, independent of the real dataset."""
    X_train = pd.DataFrame({
        "feature_1": [0.1, 5.0, 0.2, 4.8],
        "feature_2": [0.0, 3.0, -0.1, 2.9],
    })
    y_train = [0, 1, 0, 1]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def test_predict_transaction_returns_expected_keys(dummy_model):
    sample = pd.DataFrame({"feature_1": [4.9], "feature_2": [3.1]})
    result = predict_transaction(dummy_model, sample)

    assert "is_fraud" in result
    assert "fraud_probability" in result


def test_predict_transaction_probability_is_valid_range(dummy_model):
    sample = pd.DataFrame({"feature_1": [0.15], "feature_2": [0.05]})
    result = predict_transaction(dummy_model, sample)

    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_predict_transaction_is_fraud_is_boolean(dummy_model):
    sample = pd.DataFrame({"feature_1": [4.9], "feature_2": [3.1]})
    result = predict_transaction(dummy_model, sample)

    assert isinstance(result["is_fraud"], bool)