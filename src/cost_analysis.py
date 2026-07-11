"""Cost framework for cost-sensitive fraud detection.

Converts the problem from an accuracy-based decision (predict class, optimize F1)
into an economic decision (predict risk, minimize expected financial loss).

Cost values are grounded in published benchmarks, not arbitrary numbers:

- False Negative (missed fraud) cost: approximated as the average dollar amount
  of fraud transactions in the dataset itself, following the approach in
  Chen et al. (2026), "A Regulatory Governance Framework for AI-Driven Financial
  Fraud Detection in U.S. Banking" (arXiv:2605.04076), which computes net savings
  using the empirical mean transaction amount rather than an assumed flat cost.
- False Positive (false alarm) cost: $25 per manual review, a conservative
  industry benchmark for alert investigation cost at mid-size financial
  institutions (FluxForce, 2024, citing LexisNexis Risk Solutions' 2024 survey
  of 1,000+ compliance decision-makers).
  https://www.fluxforce.ai/statistics/false-positive-rates-transaction-monitoring
"""

from dataclasses import dataclass

import pandas as pd

from src.data_loader import load_raw_data

DEFAULT_COST_FP = 25.0  # USD per false-alarm manual review (FluxForce/LexisNexis 2024)


@dataclass
class CostMatrix:
    """Represents the asymmetric cost of classification errors.

    Attributes:
        cost_fp: Flat cost of investigating a false alarm.
        avg_fn_cost: Average cost of a missed fraud transaction (typically the
            mean dollar amount of fraud transactions in the dataset).
    """
    cost_fp: float
    avg_fn_cost: float

    def summary(self) -> str:
        ratio = self.avg_fn_cost / self.cost_fp if self.cost_fp else float("inf")
        return (
            f"Cost matrix: False Negative \u2248 ${self.avg_fn_cost:,.2f} (avg missed fraud amount) | "
            f"False Positive = ${self.cost_fp:,.2f} (manual review cost) | "
            f"Cost ratio (FN:FP) \u2248 {ratio:.1f}:1"
        )


def get_fraud_amounts_for_index(index: pd.Index) -> pd.Series:
    """Recover original (unscaled) transaction amounts for given row indices.

    Preprocessing replaces 'Amount' with 'scaled_amount' for modeling, but the
    cost framework needs real dollar values. This re-loads the raw dataset and
    looks up original amounts by index.

    Args:
        index: Row indices to look up (e.g. X_test.index).

    Returns:
        Series of original dollar amounts, aligned to the given index.
    """
    raw_df = load_raw_data()
    return raw_df.loc[index, "Amount"]


def build_cost_matrix(
    y_test: pd.Series, amounts: pd.Series | None = None, cost_fp: float = DEFAULT_COST_FP
) -> CostMatrix:
    """Build a cost matrix using the average dollar amount of fraud in the test set.

    Args:
        y_test: Ground truth labels (0 = normal, 1 = fraud).
        amounts: Original transaction amounts aligned to y_test's index. If None,
            reloads them from the raw dataset.
        cost_fp: Flat cost per false positive. Defaults to $25 (see module docstring).

    Returns:
        A CostMatrix with avg_fn_cost derived empirically from the data.
    """
    if amounts is None:
        amounts = get_fraud_amounts_for_index(y_test.index)

    fraud_amounts = amounts[y_test == 1]
    avg_fn_cost = fraud_amounts.mean()
    return CostMatrix(cost_fp=cost_fp, avg_fn_cost=avg_fn_cost)


if __name__ == "__main__":
    import joblib
    from src.risk_scoring import MODELS_DIR

    X_test, y_test = joblib.load(MODELS_DIR / "test_set.pkl")
    cost_matrix = build_cost_matrix(y_test)
    print(cost_matrix.summary())