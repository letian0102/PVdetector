"""Bandwidth heuristics inspired by the ADTnorm roughness R helpers.

This module ports the R utilities from ``roughness.R`` to Python so they can be
used alongside the existing KDE-based peak/valley detector.  The primary entry
point is :func:`find_bw_for_roughness`, which searches for the smallest
bandwidth that keeps the density curve smooth while avoiding spurious
double-peaks near the first mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class DensityMetric:
    """Summary statistics about a 1-D density profile."""

    zero_idx: np.ndarray
    y_value: np.ndarray
    big_idx: np.ndarray
    roughness: float
    n_x: int


def _find_extrema(
    x: np.ndarray, y: np.ndarray, min_y_frac: float = 0.1, min_y: float | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices of local maxima and minima.

    Parameters mirror the small helper in ``d2find.R``: we look for sign changes
    in the first derivative and keep only maxima that exceed the chosen
    threshold.  ``min_y`` acts as an absolute cutoff; when omitted, a relative
    threshold based on ``min_y_frac`` is used instead.
    """

    dy = np.diff(y)
    sgn = np.sign(dy)
    sgn[sgn == 0] = np.nan

    idx_max = np.nonzero(np.diff(sgn) < 0)[0] + 1
    idx_min = np.nonzero(np.diff(sgn) > 0)[0] + 1

    if min_y is None:
        thr = float(min_y_frac * np.nanmax(y))
    else:
        thr = float(min_y)

    if np.isfinite(thr):
        idx_max = idx_max[y[idx_max] >= thr]

    return idx_max, idx_min


def _compute_density(
    values: np.ndarray, bandwidth: float | str, grid_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate a 1-D Gaussian KDE on an evenly spaced grid."""

    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite values are required for KDE")

    if isinstance(bandwidth, str):
        kde = gaussian_kde(values, bw_method=bandwidth)
    else:
        kde = gaussian_kde(values)
        bw = float(bandwidth)
        sample_std = float(np.std(values, ddof=1))
        if np.isfinite(bw) and bw > 0 and sample_std > 0:
            kde.set_bandwidth(bw_method=bw / sample_std)

    xs = np.linspace(values.min(), values.max(), grid_size)
    ys = kde(xs)
    return xs, ys


def density_metric(
    dens_x: np.ndarray,
    dens_y: np.ndarray,
    tol: float = 0.2,
    dy_threshold: float | None = 1e-5,
) -> DensityMetric:
    """Replicate ``density_metric`` from the R helper.

    The function measures roughness via the second derivative of the KDE.  It
    also reports where the slope changes sign (candidate extrema) and which of
    those extrema clear a user-defined prominence threshold ``tol``.
    """

    x = np.asarray(dens_x, float)
    y = np.asarray(dens_y, float)
    dx = np.diff(x)
    dy = np.diff(y)

    if dy_threshold is not None:
        dy[np.abs(dy) < dy_threshold] = 0.0

    d1 = dy / dx
    sgn = np.sign(d1)
    sgn[sgn == 0] = np.nan

    ext_idx = np.nonzero(np.diff(sgn) != 0)[0] + 1

    big_mask = (
        (np.maximum(y[ext_idx - 1], y[ext_idx + 1]) - y[ext_idx]) > tol
    ) | ((y[ext_idx] - np.minimum(y[ext_idx - 1], y[ext_idx + 1])) > tol)
    big_ext = ext_idx[big_mask]

    x_mid = (x[:-1] + x[1:]) / 2.0
    d2 = np.diff(d1) / np.diff(x_mid)

    x_mid2 = (x_mid[:-1] + x_mid[1:]) / 2.0
    dx2 = np.diff(x_mid2)
    roughness = float(np.sum((d2[1:] ** 2) * dx2)) if d2.size > 1 else 0.0

    return DensityMetric(
        zero_idx=ext_idx,
        y_value=y[ext_idx],
        big_idx=big_ext,
        roughness=roughness,
        n_x=x.size,
    )


def has_small_double_peak(
    dens_x: np.ndarray,
    dens_y: np.ndarray,
    min_y_frac: float = 0.05,
    valley_prom_frac: float = 0.10,
) -> bool:
    """Detect a problematic shallow double-peak near the start of the density."""

    x = np.asarray(dens_x, float)
    y = np.asarray(dens_y, float)
    max_y = float(np.max(y))

    idx_max, idx_min = _find_extrema(x, y, min_y_frac=min_y_frac)
    if idx_max.size < 2:
        return False

    idx_max = idx_max[y[idx_max] >= min_y_frac * max_y]
    if idx_max.size < 2:
        return False

    idx_max = idx_max[np.argsort(x[idx_max])]
    m1, m2 = idx_max[0], idx_max[1]

    between = np.arange(m1, m2 + 1)
    v = between[np.argmin(y[between])]

    valley_depth = min(y[m1], y[m2]) - y[v]
    return bool(valley_depth < valley_prom_frac * max_y)


def find_bw_for_roughness(
    values: Iterable[float],
    target: float = 5.0,
    lower: float = 0.01,
    upper: float = 0.3,
    tol_bw: float = 1e-3,
    max_iter: int = 50,
    min_y_frac_peak: float = 0.05,
    valley_prom_frac: float = 0.1,
    grid_size: int = 512,
) -> float:
    """Binary-search the smallest bandwidth that passes roughness checks."""

    data = np.asarray(list(values), float)
    data = data[np.isfinite(data)]
    if data.size < 2:
        raise ValueError("Bandwidth search requires at least two finite values")

    def good_bw(bw: float) -> bool:
        xs, ys = _compute_density(data, bw, grid_size)
        dm = density_metric(xs, ys)
        if dm.roughness > target:
            return False
        if has_small_double_peak(xs, ys, min_y_frac=min_y_frac_peak, valley_prom_frac=valley_prom_frac):
            return False
        return True

    xs_low, ys_low = _compute_density(data, lower, grid_size)
    r_low = density_metric(xs_low, ys_low)
    if r_low.roughness <= target and not has_small_double_peak(
        xs_low, ys_low, min_y_frac=min_y_frac_peak, valley_prom_frac=valley_prom_frac
    ):
        return float(lower)

    if not good_bw(upper):
        return float(upper)

    iter_count = 0
    low, high = float(lower), float(upper)
    while (high - low) > tol_bw and iter_count < max_iter:
        mid = 0.5 * (low + high)
        if good_bw(mid):
            high = mid
        else:
            low = mid
        iter_count += 1

    return float(high)
