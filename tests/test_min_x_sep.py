import numpy as np

from peak_valley.kde_detector import kde_peaks_valleys


def test_min_x_sep_smaller_than_grid_spacing():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 100)
    # Use a tiny ``min_x_sep`` to ensure the computed distance would
    # previously be zero and raise a ValueError in ``find_peaks``.
    peaks, valleys, xs, ys = kde_peaks_valleys(
        data, n_peaks=None, min_x_sep=1e-9, grid_size=50
    )
    # The call should succeed and return lists for peaks/valleys.
    assert isinstance(peaks, list)
    assert isinstance(valleys, list)


def test_close_peaks_are_removed():
    rng = np.random.default_rng(1)
    data = np.concatenate([
        rng.normal(-0.2, 0.03, 400),
        rng.normal(0.2, 0.03, 400),
        rng.normal(3.0, 0.1, 400),
    ])

    peaks, _, _, _ = kde_peaks_valleys(
        data,
        n_peaks=None,
        min_x_sep=0.6,
        prominence=0.01,
        grid_size=4000,
    )

    assert len(peaks) >= 1
    peaks_sorted = np.sort(peaks)
    if peaks_sorted.size > 1:
        diffs = np.diff(peaks_sorted)
        assert np.all(diffs >= 0.6 - 1e-6)


def test_spacing_survives_gmm_fallback():
    rng = np.random.default_rng(7)
    data = np.concatenate([
        rng.normal(-0.1, 0.02, 300),
        rng.normal(0.25, 0.02, 300),
        rng.normal(4.0, 0.15, 500),
    ])

    peaks, _, _, _ = kde_peaks_valleys(
        data,
        n_peaks=3,
        min_x_sep=0.8,
        prominence=0.02,
        grid_size=4000,
    )

    peaks_sorted = np.sort(peaks)
    if peaks_sorted.size > 1:
        diffs = np.diff(peaks_sorted)
        assert np.all(diffs >= 0.8 - 1e-6)
