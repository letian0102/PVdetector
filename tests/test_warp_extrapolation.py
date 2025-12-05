import numpy as np

from peak_valley.alignment import build_warp_function


def test_extrapolation_uses_positive_slope_when_interval_flat():
    # Collapse the final interval so the native slope would be zero and verify
    # that the extrapolated tail still grows (avoiding a right-side cutoff).
    warp = build_warp_function([0.0, 1.0], [0.5, 0.5])

    xs = np.array([0.0, 0.5, 1.0, 2.0])
    warped = warp(xs)

    # The last point should move beyond the anchor at x=1.0 despite the flat
    # target interval.
    assert warped[-1] > warped[-2]
