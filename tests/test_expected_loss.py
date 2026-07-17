"""Unit tests for expected loss calculation."""

import pandas as pd
import pytest

from src.cost_analysis import CostMatrix
from src.expected_loss import compute_expected_loss, compute_loss_curve


@pytest.fixture
def simple_cost_matrix():
    return CostMatrix(cost_fp=10.0, avg_fn_cost=100.0)


def test_compute_expected_loss_perfect_prediction_has_zero_loss(simple_cost_matrix):
    y_true = pd.Series([0, 0, 1, 1])
    risk_scores = pd.Series([0.1, 0.2, 0.9, 0.95])

    result = compute_expected_loss(y_true, risk_scores, threshold=0.5, cost_matrix=simple_cost_matrix)

    assert result.false_negatives == 0
    assert result.false_positives == 0
    assert result.expected_loss == 0.0


def test_compute_expected_loss_counts_false_negative_correctly(simple_cost_matrix):
    y_true = pd.Series([1])
    risk_scores = pd.Series([0.3])  # below threshold -> classified as normal -> FN

    result = compute_expected_loss(y_true, risk_scores, threshold=0.5, cost_matrix=simple_cost_matrix)

    assert result.false_negatives == 1
    assert result.expected_loss == pytest.approx(100.0)


def test_compute_expected_loss_counts_false_positive_correctly(simple_cost_matrix):
    y_true = pd.Series([0])
    risk_scores = pd.Series([0.8])  # above threshold -> classified as fraud -> FP

    result = compute_expected_loss(y_true, risk_scores, threshold=0.5, cost_matrix=simple_cost_matrix)

    assert result.false_positives == 1
    assert result.expected_loss == pytest.approx(10.0)


def test_compute_loss_curve_returns_one_row_per_threshold(simple_cost_matrix):
    y_true = pd.Series([0, 1, 0, 1])
    risk_scores = pd.Series([0.2, 0.8, 0.4, 0.6])
    thresholds = [0.3, 0.5, 0.7]

    curve = compute_loss_curve(y_true, risk_scores, simple_cost_matrix, thresholds=thresholds)

    assert len(curve) == 3
    assert list(curve["threshold"]) == thresholds


def test_compute_loss_curve_has_expected_columns(simple_cost_matrix):
    y_true = pd.Series([0, 1])
    risk_scores = pd.Series([0.2, 0.8])

    curve = compute_loss_curve(y_true, risk_scores, simple_cost_matrix, thresholds=[0.5])

    expected_cols = {"threshold", "false_negatives", "false_positives", "expected_loss"}
    assert expected_cols.issubset(set(curve.columns))