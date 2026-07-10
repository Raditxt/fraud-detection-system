"""Unit tests for preprocessing utilities."""

import pandas as pd
import pytest

from src.preprocessing import (
    scale_amount_column,
    scale_time_column,
    prepare_features,
    split_features_and_target,
)


def test_scale_amount_column_returns_same_row_count():
    df = pd.DataFrame({"Amount": [10.0, 200.0, 50.5]})
    result = scale_amount_column(df)
    assert len(result) == len(df)


def test_scale_amount_column_produces_zero_mean():
    df = pd.DataFrame({"Amount": [10.0, 200.0, 50.5, 300.0]})
    result = scale_amount_column(df)
    assert result["scaled_amount"].mean() == pytest.approx(0, abs=1e-6)


def test_prepare_features_drops_original_columns():
    df = pd.DataFrame({
        "Amount": [10.0, 20.0],
        "Time": [0, 1000],
        "V1": [0.1, 0.2],
        "Class": [0, 1],
    })
    result = prepare_features(df)
    assert "Amount" not in result.columns
    assert "Time" not in result.columns
    assert "scaled_amount" in result.columns
    assert "scaled_time" in result.columns


def test_split_features_and_target_preserves_class_ratio():
    df = pd.DataFrame({
        "V1": range(100),
        "scaled_amount": range(100),
        "Class": [0] * 90 + [1] * 10,  # 10% fraud
    })
    X_train, X_test, y_train, y_test = split_features_and_target(df, test_size=0.2)

    assert len(X_train) == 80
    assert len(X_test) == 20
    # Stratified split should keep the fraud ratio close to 10% in both sets
    assert y_train.mean() == pytest.approx(0.1, abs=0.02)
    assert y_test.mean() == pytest.approx(0.1, abs=0.02)