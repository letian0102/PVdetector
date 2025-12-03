import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peak_valley.batch import _effective_bandwidth
from peak_valley.kde_detector import _mostly_small_discrete


def _expected_bandwidth(x, bw):
    kde = gaussian_kde(x, bw_method=bw)
    if _mostly_small_discrete(x):
        kde.set_bandwidth(kde.factor * 4.0)
    return math.sqrt(float(kde.covariance.squeeze()))


def test_effective_bandwidth_scott_matches_gaussian_kde():
    x = np.linspace(-2, 2, 25)
    expected = _expected_bandwidth(x, "scott")
    actual = _effective_bandwidth(x, "scott")
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)


def test_effective_bandwidth_silverman_matches_gaussian_kde():
    x = np.array([0.5, 1.2, 1.5, 2.2, 2.8, 3.1])
    expected = _expected_bandwidth(x, "silverman")
    actual = _effective_bandwidth(x, "silverman")
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)


def test_effective_bandwidth_scalar_matches_gaussian_kde():
    x = np.array([1.0, 1.4, 1.6, 2.3, 2.9])
    expected = _expected_bandwidth(x, 0.5)
    actual = _effective_bandwidth(x, 0.5)
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0.0)
