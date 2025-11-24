from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture

__all__ = ["gmm_component_means"]

def gmm_component_means(x: np.ndarray, k: int) -> list[float]:
    """
    Return the k component means of a 1-D GaussianMixture,
    sorted from left to right.
    """
    rng = np.random.default_rng()

    if x.size > 10_000:                       # speed, like KDE
        if x.size > 200_000:
            # ``np.random.choice`` without replacement shuffles the full input.
            # For very large arrays that partial shuffle can dominate runtime for
            # an individual sample, so switch to replacement to keep the draw
            # proportional to the requested sample size.
            x = rng.choice(x, 10_000, replace=True)
        else:
            x = rng.choice(x, 10_000, replace=False)

    x = x.reshape(-1, 1)

    gm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=2,
        max_iter=200,
        random_state=42,
    ).fit(x)
    return sorted(gm.means_.ravel().tolist())