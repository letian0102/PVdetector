from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture

__all__ = ["gmm_component_means"]

def gmm_component_means(x: np.ndarray, k: int) -> list[float]:
    """
    Return the k component means of a 1-D GaussianMixture,
    sorted from left to right.
    """
    if x.size > 10_000:                       # speed, like KDE
        x = np.random.choice(x, 10_000, False).reshape(-1, 1)
    else:
        x = x.reshape(-1, 1)

    gm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=2,
        max_iter=200,
        random_state=42,
    ).fit(x)
    return sorted(gm.means_.ravel().tolist())