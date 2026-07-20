"""Unit tests for model comparison logic."""

import pandas as pd
import pytest

from src.system_simulation import compute_do_nothing_baseline


def test_baseline_used_for_comparison_is_consistent():
    """Sanity check that the same baseline logic used elsewhere applies here too."""
    y_true = pd.Series([1, 0, 1], index=[0, 1, 2])
    amounts = pd.Series([100.0, 20.0, 250.0], index=[0, 1, 2])

    result = compute_do_nothing_baseline(y_true, amounts)

    assert result.total_cost == pytest.approx(350.0)