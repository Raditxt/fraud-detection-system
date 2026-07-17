"""End-to-end simulation of the 3-tier decision system: quantifies total
dollar cost and compares it against a do-nothing baseline.

False-block cost (legitimate transactions auto-blocked) uses the
transaction's own dollar amount as a conservative, directly measurable
cost floor — the immediate lost sale. Published research indicates the
true cost is substantially higher once customer lifetime value is
factored in: merchants report losing $30-75 for every $1 of fraud
prevented, due to customers who don't return after a false decline
(Aite Group, 2019, cited by INETCO; Corgi Labs, 2026, citing Merchant
Risk Council 2024). This module deliberately uses the conservative
floor rather than these multipliers, since they were measured in an
e-commerce checkout context that may not transfer directly to this
dataset.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class SystemCostResult:
    """Dollar cost breakdown for one fraud-decision scenario.

    Attributes:
        label: Name of this scenario.
        fn_loss: Total dollar loss from fraud that was not caught.
        review_cost: Total cost of manual review.
        false_block_loss: Total dollar loss from legitimate transactions
            auto-blocked (lost sales).
    """
    label: str
    fn_loss: float
    review_cost: float
    false_block_loss: float

    @property
    def total_cost(self) -> float:
        return self.fn_loss + self.review_cost + self.false_block_loss

    def summary(self) -> str:
        return (
            f"{self.label}:\n"
            f"  Missed fraud loss:   ${self.fn_loss:>12,.2f}\n"
            f"  Manual review cost:  ${self.review_cost:>12,.2f}\n"
            f"  False-block loss:    ${self.false_block_loss:>12,.2f}\n"
            f"  TOTAL:               ${self.total_cost:>12,.2f}"
        )


def compute_tier_system_cost(
    y_true: pd.Series,
    tiers: pd.Series,
    amounts: pd.Series,
    cost_review: float = 25.0,
    label: str = "3-Tier System",
) -> SystemCostResult:
    """Compute total dollar cost of the 3-tier decision system.

    Args:
        y_true: Ground truth labels, indexed like tiers and amounts.
        tiers: Output of assign_risk_tier (approve/review/block).
        amounts: Original dollar transaction amounts, same index.
        cost_review: Flat cost per manually reviewed transaction.
        label: Scenario name used in summary().

    Returns:
        SystemCostResult with the three cost components broken out.
    """
    approve_mask = tiers == "approve"
    review_mask = tiers == "review"
    block_mask = tiers == "block"

    fn_mask = approve_mask & (y_true == 1)  # fraud that slipped through approval
    fn_loss = amounts[fn_mask].sum()

    review_cost = review_mask.sum() * cost_review  # flat fee, regardless of amount

    false_block_mask = block_mask & (y_true == 0)  # legit transactions wrongly blocked
    false_block_loss = amounts[false_block_mask].sum()

    return SystemCostResult(
        label=label, fn_loss=fn_loss, review_cost=review_cost, false_block_loss=false_block_loss
    )


def compute_do_nothing_baseline(y_true: pd.Series, amounts: pd.Series) -> SystemCostResult:
    """Cost if no fraud detection existed: every fraud transaction succeeds.

    Args:
        y_true: Ground truth labels.
        amounts: Original dollar amounts.

    Returns:
        SystemCostResult representing total fraud loss with zero intervention.
    """
    fraud_loss = amounts[y_true == 1].sum()
    return SystemCostResult(
        label="Do-Nothing Baseline", fn_loss=fraud_loss, review_cost=0.0, false_block_loss=0.0
    )


if __name__ == "__main__":
    import joblib

    from src.cost_analysis import build_cost_matrix, get_fraud_amounts_for_index
    from src.decision_layer import assign_risk_tier, find_block_threshold
    from src.expected_loss import compute_loss_curve
    from src.risk_scoring import MODELS_DIR, load_model, score_transactions

    X_test, y_test = joblib.load(MODELS_DIR / "test_set.pkl")
    model = load_model("random_forest.pkl")
    risk_scores = score_transactions(model, X_test)
    cost_matrix = build_cost_matrix(y_test)
    amounts = get_fraud_amounts_for_index(y_test.index)  # returns ALL amounts for this index

    loss_curve = compute_loss_curve(y_test, risk_scores, cost_matrix)
    review_threshold = loss_curve.loc[loss_curve["expected_loss"].idxmin(), "threshold"]
    block_threshold = find_block_threshold(y_test, risk_scores, min_precision=0.95)

    tiers = assign_risk_tier(risk_scores, review_threshold, block_threshold)

    baseline = compute_do_nothing_baseline(y_test, amounts)
    tier_system = compute_tier_system_cost(y_test, tiers, amounts)

    print(baseline.summary())
    print()
    print(tier_system.summary())

    savings = baseline.total_cost - tier_system.total_cost
    savings_pct = (savings / baseline.total_cost) * 100
    print(f"\nNet impact vs. do-nothing baseline: ${savings:,.2f} saved ({savings_pct:.1f}%)")