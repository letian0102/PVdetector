import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.alignment import align_distributions


def test_align_distributions_returns_density_warp():
    counts = [
        np.array([0.0, 0.5, 1.0]),
        np.array([-0.1, 0.0, 0.1, 0.2]),
    ]
    peaks = [[0.0, 1.0], [-0.1, 0.2]]
    valleys = [[0.4], [0.05]]

    xs = np.linspace(-0.2, 1.2, 10)
    ys = np.linspace(0.0, 1.0, 10)
    density_grids = [(xs, ys), None]

    warped_counts, warped_landmarks, warp_funs, warped_density = align_distributions(
        counts,
        peaks,
        valleys,
        density_grids=density_grids,
    )

    assert len(warped_density) == len(counts)
    xs_warp, ys_warp = warped_density[0]
    np.testing.assert_allclose(xs_warp, warp_funs[0](xs))
    np.testing.assert_allclose(ys_warp, ys)
    assert warped_density[1] is None


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

