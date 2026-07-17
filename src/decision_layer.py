"""Three-tier decision layer: approve / review / block.

A single binary threshold treats all "risky" transactions identically. In
practice, institutions use tiered decisioning: very low risk is approved
instantly, very high risk is blocked instantly (no human needed), and the
ambiguous middle band goes to manual review. This mirrors industry practice
where mid-size institutions target a 30-50% false positive rate on reviewed
alerts, rather than reviewing everything (Flagright, 2024, citing LexisNexis
Risk Solutions).
https://www.flagright.com/post/understanding-false-positives-in-transaction-monitoring
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score


@dataclass
class DecisionTierResult:
    """Summary statistics for one tier of the decision system.

    Attributes:
        tier: Name of the tier ('approve', 'review', or 'block').
        count: Number of transactions falling in this tier.
        actual_fraud_count: How many of those transactions are truly fraud.
        actual_normal_count: How many are truly normal.
    """
    tier: str
    count: int
    actual_fraud_count: int
    actual_normal_count: int

    def fraud_rate(self) -> float:
        return self.actual_fraud_count / self.count if self.count else 0.0


def find_block_threshold(
    y_true: pd.Series, risk_scores: pd.Series, min_precision: float = 0.95
) -> float:
    """Find the lowest threshold at which precision meets a target level.

    This identifies the point above which the model is confident enough
    to block transactions automatically without human review.

    Args:
        y_true: Ground truth labels.
        risk_scores: Predicted fraud probabilities.
        min_precision: Minimum acceptable precision for auto-blocking.

    Returns:
        The lowest threshold satisfying the precision target. Falls back
        to 0.99 if no threshold achieves the target precision.
    """
    for threshold in np.arange(0.50, 1.00, 0.01):
        y_pred = (risk_scores >= threshold).astype(int)
        if y_pred.sum() == 0:
            continue
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision >= min_precision:
            return round(threshold, 2)
    return 0.99


def assign_risk_tier(
    risk_scores: pd.Series, review_threshold: float, block_threshold: float
) -> pd.Series:
    """Classify each transaction into approve / review / block.

    Args:
        risk_scores: Predicted fraud probabilities.
        review_threshold: Scores at or above this enter manual review.
        block_threshold: Scores at or above this are auto-blocked.

    Returns:
        Series of tier labels, same index as risk_scores.

    Raises:
        ValueError: If block_threshold is not greater than review_threshold.
    """
    if block_threshold <= review_threshold:
        raise ValueError("block_threshold must be greater than review_threshold")

    conditions = [
        risk_scores >= block_threshold,
        risk_scores >= review_threshold,
    ]
    choices = ["block", "review"]
    tiers = np.select(conditions, choices, default="approve")
    return pd.Series(tiers, index=risk_scores.index, name="tier")


def summarize_decision_tiers(y_true: pd.Series, tiers: pd.Series) -> list[DecisionTierResult]:
    """Summarize how each tier performs against ground truth.

    Args:
        y_true: Ground truth labels.
        tiers: Output of assign_risk_tier.

    Returns:
        List of DecisionTierResult, one per tier present in the data.
    """
    results = []
    for tier_name in ["approve", "review", "block"]:
        mask = tiers == tier_name
        if mask.sum() == 0:
            continue
        tier_labels = y_true[mask]
        results.append(
            DecisionTierResult(
                tier=tier_name,
                count=int(mask.sum()),
                actual_fraud_count=int((tier_labels == 1).sum()),
                actual_normal_count=int((tier_labels == 0).sum()),
            )
        )
    return results


if __name__ == "__main__":
    import joblib

    from src.cost_analysis import build_cost_matrix
    from src.expected_loss import compute_loss_curve
    from src.risk_scoring import MODELS_DIR, load_model, score_transactions

    X_test, y_test = joblib.load(MODELS_DIR / "test_set.pkl")
    model = load_model("random_forest.pkl")
    risk_scores = score_transactions(model, X_test)
    cost_matrix = build_cost_matrix(y_test)

    # Reuse Step 3's cost-optimal threshold as the review cutoff
    loss_curve = compute_loss_curve(y_test, risk_scores, cost_matrix)
    review_threshold = loss_curve.loc[loss_curve["expected_loss"].idxmin(), "threshold"]

    block_threshold = find_block_threshold(y_test, risk_scores, min_precision=0.95)

    print(f"Review threshold (cost-optimal, from Step 3): {review_threshold:.2f}")
    print(f"Block threshold (>=95% precision): {block_threshold:.2f}")
    print()

    tiers = assign_risk_tier(risk_scores, review_threshold, block_threshold)
    summaries = summarize_decision_tiers(y_test, tiers)

    for s in summaries:
        print(
            f"{s.tier.upper():8s} | count={s.count:>6} | "
            f"actual_fraud={s.actual_fraud_count:>3} | "
            f"fraud_rate={s.fraud_rate():.4%}"
        )

    total_reviewed = sum(s.count for s in summaries if s.tier == "review")
    total_transactions = len(y_test)
    print(
        f"\n{total_reviewed:,} of {total_transactions:,} transactions "
        f"({total_reviewed / total_transactions:.2%}) require manual review."
    )