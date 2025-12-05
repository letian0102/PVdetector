import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.alignment import align_distributions


def test_alignment_ignores_outlier_landmarks_when_choosing_target():
    counts = [
        np.array([0.0, 0.5, 1.0]),
        np.array([0.1, 0.6, 1.1]),
        np.array([5.0, 5.5, 6.0]),
    ]
    peaks = [
        [0.0, 1.0, 2.0],
        [0.1, 1.1, 2.1],
        [5.0, 6.0, 7.0],
    ]
    valleys = [
        [0.5, 1.5],
        [0.6, 1.6],
        [5.5, 6.5],
    ]

    warped_counts, warped_landmarks, warp_funs = align_distributions(
        counts,
        peaks,
        valleys,
    )

    # With median-based targets the warp of a well-behaved sample should stay
    # close to its original scale despite the extreme outlier.
    mid_point = 1.0
    expected = 1.1
    np.testing.assert_allclose(warp_funs[0](mid_point), expected)

    # Each sample's landmarks are shifted onto the shared target grid.
    np.testing.assert_allclose(
        warped_landmarks[0],
        np.array([0.1, 0.6, 2.1]),
    )
    np.testing.assert_allclose(
        warped_landmarks[2],
        np.array([0.1, 0.6, 2.1]),
    )

    # Warp functions should return floats even when applied to integer inputs.
    assert all(isinstance(v, float) for v in warp_funs[0]([0, 1]))
