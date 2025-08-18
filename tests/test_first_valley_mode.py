import numpy as np

from peak_valley.kde_detector import kde_peaks_valleys


def test_first_valley_slope_mode():
    rng = np.random.default_rng(0)
    data = np.concatenate([
        rng.normal(-1, 0.1, 200),
        rng.normal(1, 0.1, 200),
    ])
    peaks, valleys, *_ = kde_peaks_valleys(
        data, n_peaks=2, grid_size=2000, first_valley="slope"
    )
    assert len(valleys) == 1
    assert peaks[0] < valleys[0] < peaks[1]


def test_first_valley_drop_mode_single_peak():
    rng = np.random.default_rng(1)
    data = rng.normal(0, 0.2, 300)
    peaks, valleys, *_ = kde_peaks_valleys(
        data, n_peaks=1, grid_size=2000, drop_frac=0.2, first_valley="drop"
    )
    assert len(peaks) == 1
    assert len(valleys) == 1
    assert valleys[0] > peaks[0]
