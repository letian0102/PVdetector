import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.alignment import align_distributions


def test_align_distributions_returns_density_warp():
    counts = [np.array([0.1, 0.2, 0.3])]
    peaks = [[0.1, 0.3]]
    valleys = [[0.2]]

    # Force a left shift via target landmarks; the aligner should then shift
    # everything back to keep the non-negative domain intact.
    target_landmark = [-0.3, -0.2, -0.1]

    xs = np.linspace(-0.2, 0.4, 7)
    ys = np.linspace(0.0, 1.0, 7)
    density_grids = [(xs, ys)]

    warped_counts, warped_landmarks, warp_funs, warped_density = align_distributions(
        counts,
        peaks,
        valleys,
        target_landmark=target_landmark,
        density_grids=density_grids,
    )

    expected_counts = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(warped_counts[0], expected_counts)
    np.testing.assert_allclose(warped_landmarks[0], expected_counts)

    xs_warp, ys_warp = warped_density[0]
    expected_xs = warp_funs[0](np.clip(xs, 0.0, None))
    np.testing.assert_allclose(xs_warp, expected_xs)
    np.testing.assert_allclose(ys_warp, ys)


def test_align_distributions_density_shape_mismatch():
    counts = [np.array([0.0, 0.5, 1.0])]
    peaks = [[0.0, 1.0]]
    valleys = [[0.4]]

    with pytest.raises(ValueError):
        align_distributions(
            counts,
            peaks,
            valleys,
            density_grids=[(np.array([0, 1]), np.array([0, 1, 2]))],
        )

