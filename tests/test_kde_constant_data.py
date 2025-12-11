import numpy as np

from peak_valley.kde_detector import kde_peaks_valleys


def test_kde_handles_constant_data_without_hanging():
    data = np.ones(512)

    peaks, valleys, xs, ys, bandwidth = kde_peaks_valleys(data, grid_size=1024)

    assert len(peaks) == 1
    assert bandwidth is not None
    assert np.isfinite(ys).all()
    assert xs.size == ys.size
