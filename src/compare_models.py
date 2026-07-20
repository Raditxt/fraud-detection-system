"""Compare models on total system cost, not just F1/precision/recall.

This directly tests the claim that a "better" model (higher F1/AUC) actually
produces a better business outcome (lower total dollar cost) once threshold
optimization and the 3-tier decision layer are applied to each model
individually. A model can look better on paper and still lose on cost, or
vice versa — this script checks empirically rather than assuming.
"""

import joblib

from src.cost_analysis import build_cost_matrix, get_fraud_amounts_for_index
from src.risk_scoring import MODELS_DIR, load_model
from src.system_simulation import compute_do_nothing_baseline, simulate_model_cost

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}


def run_comparison():
    """Run the full cost-sensitive pipeline for each trained model and compare."""
    X_test, y_test = joblib.load(MODELS_DIR / "test_set.pkl")
    amounts = get_fraud_amounts_for_index(y_test.index)
    cost_matrix = build_cost_matrix(y_test)

    baseline = compute_do_nothing_baseline(y_test, amounts)
    print(baseline.summary())
    print()

    results = []
    for label, filename in MODEL_FILES.items():
        model = load_model(filename)
        result = simulate_model_cost(model, X_test, y_test, amounts, cost_matrix, label=label)
        results.append(result)
        print(result.summary())
        savings_pct = (baseline.total_cost - result.total_cost) / baseline.total_cost * 100
        print(f"  -> {savings_pct:.1f}% cost reduction vs. do-nothing baseline\n")

    best = min(results, key=lambda r: r.total_cost)
    print(f"Best model by total system cost: {best.label} (${best.total_cost:,.2f})")


if __name__ == "__main__":
    run_comparison()