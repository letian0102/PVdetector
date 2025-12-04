import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.kde_detector import kde_peaks_valleys


def test_kde_grid_clamped_for_nonnegative_data():
    rng = np.random.default_rng(123)
    data = rng.uniform(0.05, 2.0, 500)

    *_ , xs, _ys, _bw = kde_peaks_valleys(data, grid_size=256)

    assert xs.min() >= 0.0


def test_kde_grid_allows_negative_data():
    rng = np.random.default_rng(456)
    data = rng.normal(-0.2, 0.05, 300)

    *_ , xs, _ys, _bw = kde_peaks_valleys(data, grid_size=256)

    assert xs.min() < -0.1
