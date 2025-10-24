from __future__ import annotations
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

__all__ = ["gmm_component_means"]

def gmm_component_means(x: np.ndarray, k: int) -> list[float]:
    """
    Return the k component means of a 1-D GaussianMixture,
    sorted from left to right.
    """
    finite = np.asarray(x, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return []

    unique_vals = np.unique(finite)
    if unique_vals.size == 0:
        return []

    effective_k = min(k, int(unique_vals.size))
    if effective_k == 0:
        return []

    if finite.size > 10_000:                       # speed, like KDE
        finite = np.random.choice(finite, 10_000, False)

    data = finite.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gm = GaussianMixture(
            n_components=effective_k,
            covariance_type="full",
            n_init=3,
            max_iter=200,
            random_state=42,
            reg_covar=1e-6,
        ).fit(data)

    means = sorted(gm.means_.ravel().tolist())
    return means