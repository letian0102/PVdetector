import numpy as np

from peak_valley.batch import BatchOptions
from peak_valley.kde_detector import (
    _resolve_peak_valley_conflicts,
    kde_peaks_valleys,
)


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


def test_requested_peaks_respect_min_separation():
    rng = np.random.default_rng(7)
    # Two peaks only 0.25 apart – below the default 0.5 separation.
    data = np.concatenate(
        [
            rng.normal(0.0, 0.04, 500),
            rng.normal(0.25, 0.04, 400),
        ]
    )

    peaks, *_ = kde_peaks_valleys(
        data,
        n_peaks=3,
        grid_size=4_000,
        bw=0.05,
        prominence=0.01,
        min_x_sep=0.5,
    )

    # Only one well-separated peak should remain despite the higher request.
    assert len(peaks) == 1


def test_valleys_clear_gap_to_nearest_peak():
    rng = np.random.default_rng(21)
    data = rng.normal(0.0, 0.08, 800)

    peaks, valleys, *_ = kde_peaks_valleys(
        data,
        n_peaks=2,
        grid_size=4_000,
        bw=0.05,
        prominence=0.005,
        min_x_sep=1.0,
    )

    assert len(peaks) == 1
    assert all(abs(v - peaks[0]) >= 0.5 - 1e-9 for v in valleys)


def test_valley_too_close_to_second_peak_drops_peak():
    xs = np.linspace(0.0, 4.0, 41)
    ys = np.array([
        0.8112544 , 0.36456734, 0.39457126, 0.7342476 , 1.36737863, 1.09449631,
        0.60329188, 0.94262184, 0.71894057, 0.22669886, 1.16242119, 1.08821173,
        1.47913783, 0.86650736, 0.1225441 , 0.79608678, 0.48721816, 0.97500012,
        0.62042914, 1.00478722, 0.36746621, 0.79489107, 0.48044594, 0.2072658 ,
        0.58103257, 0.53125122, 0.08905604, 1.5941569 , 1.09547204, 0.3625025 ,
        0.4439914 , 0.36044017, 0.58353413, 1.43852265, 2.11880303, 1.34204445,
        0.91980726, 1.12112268, 1.15088303, 0.38477403, 0.15842291,
    ])

    peaks_idx = [10, 30]
    updated_idx, valleys = _resolve_peak_valley_conflicts(
        xs,
        ys,
        peaks_idx,
        min_x_sep=1.0,
        drop_frac=0.1,
        first_valley="slope",
    )

    assert updated_idx == [10]
    assert valleys
    assert len(valleys) == 1
    first_peak_x = xs[updated_idx[0]]
    assert abs(valleys[0] - first_peak_x) >= 0.5


def test_valley_recomputed_when_close_to_first_peak():
    xs = np.linspace(0.0, 4.0, 41)
    ys = np.ones_like(xs) * 0.9
    ys[10] = 3.2
    ys[30] = 2.6
    ys[11] = 0.05  # valley hugging the first peak

    peaks_idx = [10, 30]
    updated_idx, valleys = _resolve_peak_valley_conflicts(
        xs,
        ys,
        peaks_idx,
        min_x_sep=1.0,
        drop_frac=0.1,
        first_valley="drop",
    )

    assert updated_idx == [10, 30]
    assert valleys
    assert len(valleys) == 1
    first_peak_x = xs[updated_idx[0]]
    second_peak_x = xs[updated_idx[1]]
    assert abs(valleys[0] - first_peak_x) >= 0.5
    assert abs(second_peak_x - valleys[0]) >= 0.5
