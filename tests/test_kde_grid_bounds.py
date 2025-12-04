from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.kde_detector import kde_peaks_valleys


def test_kde_grid_clamps_negative_tail_for_non_negative_inputs():
    rng = np.random.default_rng(123)
    data = rng.lognormal(mean=0.5, sigma=0.3, size=2000)
    peaks, valleys, xs, ys, h = kde_peaks_valleys(data)

    assert xs.size > 0
    assert xs.min() >= 0.0
    assert ys.size == xs.size
    assert h is not None
    assert peaks or valleys  # ensure KDE produced structure


def test_kde_grid_allows_negative_when_present():
    rng = np.random.default_rng(123)
    data = np.concatenate(
        [rng.normal(loc=-1.0, scale=0.2, size=500), rng.normal(loc=2.0, scale=0.5, size=500)]
    )
    _, _, xs, _, _ = kde_peaks_valleys(data)

    assert xs.min() < 0.0
