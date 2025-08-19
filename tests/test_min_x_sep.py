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
