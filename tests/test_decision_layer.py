"""Unit tests for the three-tier decision layer."""

import pandas as pd
import pytest

from src.decision_layer import (
    assign_risk_tier,
    find_block_threshold,
    summarize_decision_tiers,
)


def test_assign_risk_tier_labels_correctly():
    risk_scores = pd.Series([0.05, 0.30, 0.90])
    tiers = assign_risk_tier(risk_scores, review_threshold=0.2, block_threshold=0.8)

    assert list(tiers) == ["approve", "review", "block"]


def test_assign_risk_tier_boundary_is_inclusive():
    risk_scores = pd.Series([0.20, 0.80])
    tiers = assign_risk_tier(risk_scores, review_threshold=0.2, block_threshold=0.8)

    assert list(tiers) == ["review", "block"]


def test_assign_risk_tier_rejects_invalid_thresholds():
    risk_scores = pd.Series([0.5])
    with pytest.raises(ValueError):
        assign_risk_tier(risk_scores, review_threshold=0.8, block_threshold=0.2)


def test_find_block_threshold_finds_high_precision_point():
    # Perfectly separable: fraud always scores >= 0.9, normal always <= 0.5
    y_true = pd.Series([0, 0, 0, 1, 1])
    risk_scores = pd.Series([0.1, 0.2, 0.3, 0.9, 0.95])

    threshold = find_block_threshold(y_true, risk_scores, min_precision=0.95)

    assert threshold <= 0.90


def test_summarize_decision_tiers_counts_are_correct():
    y_true = pd.Series([0, 1, 0, 1], index=[0, 1, 2, 3])
    tiers = pd.Series(["approve", "review", "approve", "block"], index=[0, 1, 2, 3])

    summaries = summarize_decision_tiers(y_true, tiers)
    summary_by_tier = {s.tier: s for s in summaries}

    assert summary_by_tier["approve"].count == 2
    assert summary_by_tier["approve"].actual_fraud_count == 0
    assert summary_by_tier["block"].actual_fraud_count == 1