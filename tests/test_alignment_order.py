import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from peak_valley.alignment import align_distributions


def test_align_distributions_preserves_order_when_row_all_nan():
    counts = [
        np.array([0.0, 1.0, 2.0]),
        np.array([-1.0, -0.5, 0.5]),
        np.array([2.0, 3.0, 4.0]),
    ]
    peaks = [
        [0.0, 2.0],
        [],
        [2.5, 3.5],
    ]
    valleys = [
        [1.0],
        [],
        [3.0],
    ]

    landmark_matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [np.nan, np.nan, np.nan],
            [2.5, 3.0, 3.5],
        ]
    )

    warped_counts, warped_landmarks, warp_funs = align_distributions(
        counts,
        peaks,
        valleys,
        landmark_matrix=landmark_matrix,
    )

    assert len(warp_funs) == len(counts)
    np.testing.assert_allclose(warped_counts[1], counts[1])
    np.testing.assert_allclose(warp_funs[1](counts[1]), counts[1])
    np.testing.assert_allclose(warped_landmarks[1], landmark_matrix[1])
