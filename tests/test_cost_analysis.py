"""Unit tests for the cost framework."""

import pandas as pd
import pytest

from src.cost_analysis import CostMatrix, build_cost_matrix


def test_build_cost_matrix_computes_average_fraud_amount():
    y_test = pd.Series([0, 1, 0, 1, 1], index=[0, 1, 2, 3, 4])
    amounts = pd.Series([10.0, 100.0, 20.0, 200.0, 300.0], index=[0, 1, 2, 3, 4])

    cost_matrix = build_cost_matrix(y_test, amounts=amounts)

    # Average of fraud-only amounts: (100 + 200 + 300) / 3 = 200
    assert cost_matrix.avg_fn_cost == pytest.approx(200.0)


def test_build_cost_matrix_uses_default_cost_fp():
    y_test = pd.Series([1, 0], index=[0, 1])
    amounts = pd.Series([50.0, 5.0], index=[0, 1])

    cost_matrix = build_cost_matrix(y_test, amounts=amounts)

    assert cost_matrix.cost_fp == 25.0


def test_build_cost_matrix_respects_custom_cost_fp():
    y_test = pd.Series([1], index=[0])
    amounts = pd.Series([75.0], index=[0])

    cost_matrix = build_cost_matrix(y_test, amounts=amounts, cost_fp=50.0)

    assert cost_matrix.cost_fp == 50.0


def test_cost_matrix_summary_includes_ratio():
    cost_matrix = CostMatrix(cost_fp=25.0, avg_fn_cost=250.0)

    summary = cost_matrix.summary()

    assert "10.0:1" in summary