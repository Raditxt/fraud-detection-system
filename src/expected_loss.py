"""Expected loss calculation: the core cost-sensitive metric.

Standard metrics like F1 treat all errors as equally important. Expected loss
instead answers the actual business question: "how much money does this
decision threshold cost us?" This directly follows the net-savings framing in
Chen et al. (2026, arXiv:2605.04076), adapted here as a loss to minimize
rather than savings to maximize.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.cost_analysis import CostMatrix


@dataclass
class LossResult:
    """Result of an expected loss calculation at a specific threshold.

    Attributes:
        threshold: The probability cutoff used to classify as fraud.
        false_negatives: Count of missed fraud cases.
        false_positives: Count of false alarms.
        true_positives: Count of correctly caught fraud.
        true_negatives: Count of correctly approved normal transactions.
        expected_loss: Total cost in dollars at this threshold.
    """
    threshold: float
    false_negatives: int
    false_positives: int
    true_positives: int
    true_negatives: int
    expected_loss: float

    def summary(self) -> str:
        return (
            f"Threshold={self.threshold:.2f} | "
            f"FN={self.false_negatives} FP={self.false_positives} | "
            f"Expected Loss=${self.expected_loss:,.2f}"
        )


def compute_expected_loss(
    y_true: pd.Series, risk_scores: pd.Series, threshold: float, cost_matrix: CostMatrix
) -> LossResult:
    """Compute expected loss at a single decision threshold.

    Args:
        y_true: Ground truth labels (0 = normal, 1 = fraud).
        risk_scores: Predicted fraud probabilities, same index as y_true.
        threshold: Probability cutoff; scores >= threshold are classified as fraud.
        cost_matrix: CostMatrix with cost_fp and avg_fn_cost.

    Returns:
        LossResult with the confusion matrix breakdown and total expected loss.
    """
    y_pred = (risk_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    expected_loss = (fn * cost_matrix.avg_fn_cost) + (fp * cost_matrix.cost_fp)

    return LossResult(
        threshold=threshold,
        false_negatives=int(fn),
        false_positives=int(fp),
        true_positives=int(tp),
        true_negatives=int(tn),
        expected_loss=expected_loss,
    )


def compute_loss_curve(
    y_true: pd.Series,
    risk_scores: pd.Series,
    cost_matrix: CostMatrix,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute expected loss across a range of thresholds.

    Args:
        y_true: Ground truth labels.
        risk_scores: Predicted fraud probabilities.
        cost_matrix: CostMatrix with cost_fp and avg_fn_cost.
        thresholds: Array of thresholds to evaluate. Defaults to 0.01 to 0.99
            in steps of 0.01.

    Returns:
        DataFrame with one row per threshold, including expected_loss.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    results = [
        compute_expected_loss(y_true, risk_scores, t, cost_matrix) for t in thresholds
    ]
    return pd.DataFrame([vars(r) for r in results])


if __name__ == "__main__":
    import joblib

    from src.cost_analysis import build_cost_matrix
    from src.risk_scoring import MODELS_DIR, load_model, score_transactions

    X_test, y_test = joblib.load(MODELS_DIR / "test_set.pkl")
    model = load_model("random_forest.pkl")
    risk_scores = score_transactions(model, X_test)
    cost_matrix = build_cost_matrix(y_test)

    print(cost_matrix.summary())
    print()

    # Baseline: default 0.5 threshold
    baseline = compute_expected_loss(y_test, risk_scores, 0.5, cost_matrix)
    print("Baseline (threshold = 0.5):")
    print(f"  {baseline.summary()}")

    # Full curve
    loss_curve = compute_loss_curve(y_test, risk_scores, cost_matrix)
    best_row = loss_curve.loc[loss_curve["expected_loss"].idxmin()]

    print(f"\nBest threshold found: {best_row['threshold']:.2f}")
    print(
        f"  FN={int(best_row['false_negatives'])} FP={int(best_row['false_positives'])} | "
        f"Expected Loss=${best_row['expected_loss']:,.2f}"
    )

    savings = baseline.expected_loss - best_row["expected_loss"]
    savings_pct = (savings / baseline.expected_loss) * 100
    print(f"\nSavings vs. baseline: ${savings:,.2f} ({savings_pct:.1f}%)")