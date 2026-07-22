"""Unit tests for stress-testing utilities."""

import pandas as pd
import pytest

from src.stress_testing import inject_feature_noise, resample_fraud_ratio


def test_resample_fraud_ratio_hits_target_approximately():
    y = pd.Series([1] * 10 + [0] * 990, index=range(1000))
    X = pd.DataFrame({"f1": range(1000)}, index=range(1000))

    X_resampled, y_resampled = resample_fraud_ratio(X, y, target_fraud_ratio=0.10)

    actual_ratio = (y_resampled == 1).sum() / len(y_resampled)
    assert actual_ratio == pytest.approx(0.10, abs=0.01)


def test_resample_fraud_ratio_keeps_all_fraud_cases():
    y = pd.Series([1] * 5 + [0] * 100, index=range(105))
    X = pd.DataFrame({"f1": range(105)}, index=range(105))

    _, y_resampled = resample_fraud_ratio(X, y, target_fraud_ratio=0.20)

    assert (y_resampled == 1).sum() == 5


def test_inject_feature_noise_preserves_shape():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    X_noisy = inject_feature_noise(X, noise_std_fraction=0.1)

    assert X_noisy.shape == X.shape


def test_inject_feature_noise_does_not_modify_original():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    X_original = X.copy()
    inject_feature_noise(X, noise_std_fraction=0.1)

    pd.testing.assert_frame_equal(X, X_original)


def test_inject_feature_noise_actually_changes_values():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    X_noisy = inject_feature_noise(X, noise_std_fraction=0.5, random_state=1)

    assert not X_noisy["a"].equals(X["a"])