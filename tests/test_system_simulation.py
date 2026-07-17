"""Unit tests for end-to-end system cost simulation."""

import pandas as pd
import pytest

from src.system_simulation import compute_do_nothing_baseline, compute_tier_system_cost


def test_compute_tier_system_cost_sums_missed_fraud_correctly():
    y_true = pd.Series([1, 0, 1], index=[0, 1, 2])
    tiers = pd.Series(["approve", "approve", "block"], index=[0, 1, 2])
    amounts = pd.Series([200.0, 10.0, 500.0], index=[0, 1, 2])

    result = compute_tier_system_cost(y_true, tiers, amounts, cost_review=25.0)

    # Only index 0 is fraud AND approved -> missed fraud loss = 200
    assert result.fn_loss == pytest.approx(200.0)


def test_compute_tier_system_cost_sums_review_cost_correctly():
    y_true = pd.Series([0, 0, 1], index=[0, 1, 2])
    tiers = pd.Series(["review", "review", "approve"], index=[0, 1, 2])
    amounts = pd.Series([50.0, 75.0, 30.0], index=[0, 1, 2])

    result = compute_tier_system_cost(y_true, tiers, amounts, cost_review=25.0)

    # 2 transactions reviewed, flat $25 each, regardless of amount
    assert result.review_cost == pytest.approx(50.0)


def test_compute_tier_system_cost_sums_false_block_loss_correctly():
    y_true = pd.Series([0, 1, 0], index=[0, 1, 2])
    tiers = pd.Series(["block", "block", "approve"], index=[0, 1, 2])
    amounts = pd.Series([40.0, 300.0, 10.0], index=[0, 1, 2])

    result = compute_tier_system_cost(y_true, tiers, amounts, cost_review=25.0)

    # Only index 0 is normal AND blocked -> false-block loss = 40
    assert result.false_block_loss == pytest.approx(40.0)


def test_compute_do_nothing_baseline_sums_all_fraud_amounts():
    y_true = pd.Series([1, 0, 1], index=[0, 1, 2])
    amounts = pd.Series([100.0, 20.0, 250.0], index=[0, 1, 2])

    result = compute_do_nothing_baseline(y_true, amounts)

    assert result.fn_loss == pytest.approx(350.0)
    assert result.total_cost == pytest.approx(350.0)