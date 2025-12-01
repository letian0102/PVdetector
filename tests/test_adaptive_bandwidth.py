import numpy as np

from peak_valley.kde_detector import kde_peaks_valleys, select_adaptive_bandwidth


def test_select_adaptive_bandwidth_returns_positive_factor():
    rng = np.random.default_rng(42)
    data = np.concatenate(
        [rng.normal(-2.0, 0.25, size=600), rng.normal(2.0, 0.25, size=600)]
    )

    factor = select_adaptive_bandwidth(data, expected_peaks=2)

    assert np.isfinite(factor) and factor > 0

    peaks, _, _, _ = kde_peaks_valleys(data, None, 0.05, factor, grid_size=2_000)
    assert len(peaks) >= 2


def test_adaptive_string_path_is_supported():
    rng = np.random.default_rng(0)
    data = np.concatenate(
        [rng.normal(-1.5, 0.3, size=400), rng.normal(1.5, 0.3, size=400)]
    )

    peaks, _, _, _ = kde_peaks_valleys(data, None, 0.05, "adaptive", grid_size=1_500)

    assert len(peaks) >= 2
