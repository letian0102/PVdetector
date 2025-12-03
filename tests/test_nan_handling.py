import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peak_valley.kde_detector import kde_peaks_valleys


def test_kde_peaks_valleys_ignores_nonfinite_values():
    data = np.array([1.0, 2.0, np.nan, np.inf, 3.0, -np.inf])
    peaks, valleys, xs, ys, bw = kde_peaks_valleys(data)

    assert np.all(np.isfinite(xs))
    assert np.all(np.isfinite(ys))
    assert all(np.isfinite(p) for p in peaks)
    assert all(np.isfinite(v) for v in valleys)
    assert np.isfinite(bw)


def test_kde_peaks_valleys_all_nonfinite_returns_empty():
    data = np.array([np.nan, np.inf, -np.inf])
    peaks, valleys, xs, ys, bw = kde_peaks_valleys(data)

    assert peaks == []
    assert valleys == []
    assert xs.size == 0
    assert ys.size == 0
    assert bw is None

