import numpy as np

from peak_valley.kde_detector import (
    _detect_turning_points,
    _prune_shallow_valleys,
    _vote_consensus_peaks,
)


def test_multiscale_consensus_requires_votes():
    xs = np.linspace(-1, 1, 9)
    peak_sets = [
        np.array([2, 6]),
        np.array([3, 6]),
        np.array([6]),
    ]

    consensus = _vote_consensus_peaks(xs, peak_sets, tol=0.3, min_votes=2)

    assert len(consensus) == 2
    assert min(consensus) < 0 < max(consensus)


def test_prune_shallow_valleys_merges_shoulders():
    xs = np.linspace(0, 4, 5)
    ys = np.array([1.0, 0.95, 0.9, 0.95, 1.0])

    peaks_idx = [0, 4]
    merged = _prune_shallow_valleys(
        xs,
        ys,
        peaks_idx,
        min_valley_drop=0.15,
        valley_area_frac=0.01,
        drop_frac=0.1,
    )

    assert merged == [0]


def test_turning_point_detector_flags_concave_plateau():
    xs = np.linspace(0, 4, 41)
    ys = -0.1 * (xs - 2) ** 2 + 1.0
    ys[ys < 0] = 0

    turning = _detect_turning_points(
        xs,
        ys,
        curvature_thresh=0.05,
        min_x_sep=0.5,
    )

    assert turning.size >= 1
    assert 1.5 < xs[turning[0]] < 2.5
