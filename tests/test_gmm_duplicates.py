import warnings

import numpy as np

from peak_valley.gmm_fallback import gmm_component_means
from peak_valley.gpt_adapter import _gmm_statistics


def test_gmm_component_means_handles_duplicate_points():
    data = np.concatenate([np.zeros(40), np.ones(40)])
    with warnings.catch_warnings(record=True) as captured:
        means = gmm_component_means(data, 3)
    assert captured == []
    assert np.allclose(means, [0.0, 1.0])


def test_gmm_statistics_skips_unavailable_components():
    data = np.concatenate([np.zeros(30), np.ones(30)])
    with warnings.catch_warnings(record=True) as captured:
        stats = _gmm_statistics(data, max_components=3)
    assert captured == []
    assert stats["bic"].get("k3") is None
    assert np.allclose(stats["means_k2"], [0.0, 1.0])
