"""Robustness stress-testing: how does the system behave when data conditions
shift away from the static test set's assumptions?

Unlike the cost framework (grounded in published industry benchmarks), the
specific stress scenarios here are exploratory checks, not benchmarked
against external research — no quantitative industry standard for "realistic"
fraud-ratio drift or feature noise was found during this project's research
phase. These tests answer a narrower question: does the system degrade
gracefully or catastrophically under distribution shift?
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


@dataclass
class StressTestResult:
    """Result of one stress test scenario.

    Attributes:
        scenario: Description of the perturbation applied.
        precision: Precision on the perturbed data.
        recall: Recall on the perturbed data.
        n_fraud: Number of fraud cases in the perturbed sample.
    """
    scenario: str
    precision: float
    recall: float
    n_fraud: int

    def summary(self) -> str:
        return (
            f"{self.scenario}: precision={self.precision:.3f} | "
            f"recall={self.recall:.3f} | n_fraud={self.n_fraud}"
        )


def resample_fraud_ratio(
    X: pd.DataFrame, y: pd.Series, target_fraud_ratio: float, random_state: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Resample the dataset to simulate a different fraud prevalence.

    Downsamples the majority (normal) class to hit a target fraud ratio,
    simulating scenarios like a fraud spike (higher ratio) or an
    unusually clean period (lower ratio).

    Args:
        X: Feature DataFrame.
        y: Labels (0 = normal, 1 = fraud).
        target_fraud_ratio: Desired proportion of fraud in the resampled set,
            e.g. 0.01 for 1%.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_resampled, y_resampled).
    """
    fraud_idx = y[y == 1].index
    normal_idx = y[y == 0].index

    n_fraud = len(fraud_idx)
    # Solve for how many normal transactions give the target ratio:
    # n_fraud / (n_fraud + n_normal) = target_fraud_ratio
    n_normal_target = int(n_fraud * (1 - target_fraud_ratio) / target_fraud_ratio)
    n_normal_target = min(n_normal_target, len(normal_idx))

    rng = np.random.RandomState(random_state)
    sampled_normal_idx = rng.choice(normal_idx, size=n_normal_target, replace=False)

    keep_idx = list(fraud_idx) + list(sampled_normal_idx)
    return X.loc[keep_idx], y.loc[keep_idx]


def inject_feature_noise(
    X: pd.DataFrame, noise_std_fraction: float = 0.1, random_state: int = 42
) -> pd.DataFrame:
    """Add Gaussian noise to features, scaled to each feature's own std dev.

    Simulates sensor/measurement noise or minor upstream data quality
    issues. Noise is scaled per-feature so that features with different
    natural scales (e.g. V1-V28 vs. scaled_amount) are perturbed
    proportionally rather than uniformly.

    Args:
        X: Feature DataFrame.
        noise_std_fraction: Noise standard deviation as a fraction of each
            feature's own standard deviation. 0.1 means noise with std
            equal to 10% of the feature's natural variability.
        random_state: Seed for reproducibility.

    Returns:
        A new DataFrame with noise added; original X is not modified.
    """
    rng = np.random.RandomState(random_state)
    X_noisy = X.copy()
    for col in X.columns:
        noise = rng.normal(loc=0, scale=X[col].std() * noise_std_fraction, size=len(X))
        X_noisy[col] = X[col] + noise
    return X_noisy


def evaluate_at_threshold(
    model, X: pd.DataFrame, y: pd.Series, threshold: float, scenario_label: str
) -> StressTestResult:
    """Evaluate a model's precision/recall on given data at a fixed threshold.

    Args:
        model: Trained classifier with predict_proba.
        X: Feature DataFrame.
        y: Ground truth labels.
        threshold: Fixed probability cutoff to apply (kept constant across
            scenarios, since the point is to test whether the SAME decision
            rule still performs well under shifted conditions).
        scenario_label: Description for this test, used in the result.

    Returns:
        StressTestResult with precision, recall, and fraud count.
    """
    risk_scores = model.predict_proba(X)[:, 1]
    y_pred = (risk_scores >= threshold).astype(int)

    return StressTestResult(
        scenario=scenario_label,
        precision=precision_score(y, y_pred, zero_division=0),
        recall=recall_score(y, y_pred, zero_division=0),
        n_fraud=int((y == 1).sum()),
    )


if __name__ == "__main__":
    import joblib

    from src.risk_scoring import MODELS_DIR, load_model

    X_test, y_test = joblib.load(MODELS_DIR / "test_set.pkl")
    model = load_model("xgboost.pkl")
    fixed_threshold = 0.21  # cost-optimal threshold found in Step 3, kept constant here

    print(f"Testing at fixed threshold={fixed_threshold} (unchanged from Step 3)\n")

    # Baseline: original test set, unperturbed
    baseline = evaluate_at_threshold(model, X_test, y_test, fixed_threshold, "Baseline (original test set)")
    print(baseline.summary())

    # Scenario 1: fraud ratio shifts
    for ratio in [0.01, 0.05, 0.30]:
        X_shift, y_shift = resample_fraud_ratio(X_test, y_test, target_fraud_ratio=ratio)
        result = evaluate_at_threshold(
            model, X_shift, y_shift, fixed_threshold, f"Fraud ratio = {ratio:.0%}"
        )
        print(result.summary())

    print()

    # Scenario 2: feature noise injection
    for noise_level in [0.05, 0.10, 0.25]:
        X_noisy = inject_feature_noise(X_test, noise_std_fraction=noise_level)
        result = evaluate_at_threshold(
            model, X_noisy, y_test, fixed_threshold, f"Feature noise = {noise_level:.0%} of std"
        )
        print(result.summary())