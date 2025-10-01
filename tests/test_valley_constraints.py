import numpy as np

from peak_valley.kde_detector import (
    kde_peaks_valleys,
    _enforce_min_valley_gap,
)


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


def test_enforce_min_valley_gap_between_peaks():
    xs = np.linspace(-5.0, 5.0, 501)
    left_idx = 150
    right_idx = 350
    valley_x = xs[left_idx] + 1e-6

    adjusted = _enforce_min_valley_gap(xs, left_idx, valley_x, right_idx)
    min_gap = 0.08 * (xs[right_idx] - xs[left_idx])

    assert adjusted - xs[left_idx] >= min_gap - 1e-9
    assert xs[right_idx] - adjusted >= min_gap - 1e-9


def test_enforce_min_valley_gap_single_peak_tail():
    xs = np.linspace(0.0, 10.0, 401)
    left_idx = 100
    valley_x = xs[left_idx] + 1e-6

    adjusted = _enforce_min_valley_gap(xs, left_idx, valley_x, None)
    min_gap = 0.08 * (xs[-1] - xs[left_idx])

    assert adjusted - xs[left_idx] >= min_gap - 1e-9
