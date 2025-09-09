import numpy as np

from peak_valley.kde_detector import kde_peaks_valleys


def test_valleys_between_peaks():
    rng = np.random.default_rng(42)
    data = np.concatenate([
        rng.normal(-2, 0.1, 200),
        rng.normal(0, 0.1, 200),
        rng.normal(2, 0.1, 200),
    ])
    peaks, valleys, *_ = kde_peaks_valleys(data, n_peaks=3, grid_size=2000)
    assert len(peaks) == 3
    assert len(valleys) == 2
    assert peaks[0] < valleys[0] < peaks[1]
    assert peaks[1] < valleys[1] < peaks[2]
