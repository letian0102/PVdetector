import numpy as np

from peak_valley.batch import BatchOptions
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


def test_default_min_separation_detects_close_doublet():
    rng = np.random.default_rng(123)
    data = np.concatenate(
        [
            rng.normal(-0.3, 0.05, 400),
            rng.normal(0.3, 0.05, 400),
        ]
    )

    options = BatchOptions()
    peaks, valleys, *_ = kde_peaks_valleys(
        data,
        grid_size=2_000,
        bw=0.05,
        prominence=0.01,
        min_x_sep=options.min_separation,
    )

    assert options.min_separation == 0.5
    assert len(peaks) == 2
    assert valleys
    assert peaks[0] < valleys[0] < peaks[1]
    assert peaks[1] - peaks[0] < 1.0
