import numpy as np

from peak_valley.kde_detector import kde_peaks_valleys, _first_valley_slope


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


def test_slope_valley_not_past_minimum():
    xs = np.arange(7, dtype=float)
    ys = np.array([5, 4, 3, 0, 0.1, 4, 5], dtype=float)
    valley = _first_valley_slope(xs, ys, 0, 6)
    assert valley == xs[3]


def test_slope_valley_before_deeper_minimum():
    xs = np.arange(10, dtype=float)
    ys = np.array([5, 4, 3, 2, 2.2, 5, 0, 5, 6, 7], dtype=float)
    valley = _first_valley_slope(xs, ys, 0)
    assert valley == xs[3]


def test_first_valley_respects_minimum_separation():
    xs = np.linspace(0, 2, 201)
    ys = np.exp(-((xs - 1.0) / 0.08) ** 2) + 0.8 * np.exp(-((xs - 1.65) / 0.08) ** 2)
    p_left = int(np.argmax(ys[:120]))
    p_right = 120 + int(np.argmax(ys[120:]))

    valley = _first_valley_slope(xs, ys, p_left, p_right, min_sep=0.3)
    assert valley - xs[p_left] >= 0.3
