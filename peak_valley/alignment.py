# peak_valley/alignment.py
# --------------------------------------------------------------
"""
Landmark alignment for 1-D ADT distributions.

This module ports the alignment logic from the original ADTnorm R
implementation:

•  build a common evaluation grid spanning the observed counts with a
   small safety margin
•  smooth each sample's density on that grid via a cubic B-spline fit
•  derive monotone warping curves that map the sample-specific landmarks
   to the cohort-wide targets (or user supplied positions)
•  evaluate the forward / inverse maps on that grid and apply them to the
   raw counts as well as the landmark coordinates

When every sample exposes a single landmark the procedure degenerates to
pure translations so the distribution shapes remain untouched while the
landmarks align to the cohort median.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator, make_interp_spline
from scipy.stats import gaussian_kde

__all__ = [
    "fill_landmark_matrix",
    "build_warp_function",
    "align_distributions",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nanmedian(values: np.ndarray, default: float) -> float:
    """Return the nanmedian or a fallback if all entries are NaN."""
    med = np.nanmedian(values)
    if np.isnan(med):
        return default
    return float(med)


def _ensure_strictly_increasing(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Force an array to be strictly increasing by nudging ties forward."""
    out = np.array(arr, dtype=float, copy=True)
    for idx in range(1, out.size):
        if out[idx] <= out[idx - 1]:
            out[idx] = out[idx - 1] + eps
    return out


def _build_common_grid(counts: Sequence[np.ndarray], margin: float = 0.05,
                       n_points: int = 512) -> np.ndarray:
    """Construct a grid that spans all observed counts with a safety margin."""
    cleaned = []
    for arr in counts:
        a = np.asarray(arr, dtype=float)
        if a.size:
            cleaned.append(a[~np.isnan(a)])
    if cleaned:
        flat = np.concatenate(cleaned)
    else:
        flat = np.array([], dtype=float)

    if flat.size == 0:
        return np.linspace(-1.0, 1.0, n_points)

    lo = float(np.min(flat))
    hi = float(np.max(flat))
    if lo == hi:
        span = max(abs(lo), 1.0)
        lo -= margin * span
        hi += margin * span
    else:
        pad = margin * (hi - lo)
        lo -= pad
        hi += pad
    return np.linspace(lo, hi, n_points)


def _prepare_alignment_grid(
    counts: Sequence[np.ndarray],
    landmark_matrix: np.ndarray,
    targets: np.ndarray,
    extend: float = 0.15,
    n_points: int = 1001,
) -> Tuple[np.ndarray, float, float]:
    """Replicate the range expansion used by ADTnorm's ``peak_alignment``."""

    cleaned: List[np.ndarray] = []
    for arr in counts:
        a = np.asarray(arr, dtype=float)
        if a.size:
            cleaned.append(a[~np.isnan(a)])

    if cleaned:
        flat = np.concatenate(cleaned)
        data_min = float(np.min(flat))
        data_max = float(np.max(flat))
    else:
        data_min, data_max = -1.0, 1.0

    span = data_max - data_min
    if span == 0:
        span = max(abs(data_min), 1.0)

    pad = extend * span

    finite_landmarks = landmark_matrix[np.isfinite(landmark_matrix)]
    finite_targets = targets[np.isfinite(targets)]

    min_candidates: List[float] = [data_min]
    max_candidates: List[float] = [data_max]

    if finite_landmarks.size:
        min_candidates.append(float(np.min(finite_landmarks)))
        max_candidates.append(float(np.max(finite_landmarks)))

    if finite_targets.size:
        min_candidates.append(float(np.min(finite_targets)))
        max_candidates.append(float(np.max(finite_targets)))

    lo = min(min_candidates) - pad
    hi = max(max_candidates) + pad

    grid = np.linspace(lo, hi, n_points)
    return grid, data_min, data_max


def _smooth_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate a smoothed density estimate on *grid* using B-splines."""
    finite = values[~np.isnan(values)]
    if finite.size < 2:
        return np.zeros_like(grid)

    try:
        kde = gaussian_kde(finite)
        dens = kde(grid)
    except (np.linalg.LinAlgError, ValueError):
        mean = float(np.mean(finite))
        sd = float(np.std(finite))
        if not np.isfinite(sd) or sd == 0:
            sd = max(abs(mean), 1.0)
        z = (grid - mean) / sd
        dens = np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))

    spline = make_interp_spline(grid, dens, k=3)
    smoothed = spline(grid)
    smoothed[smoothed < 0] = 0.0
    return smoothed


def _build_monotone_warp(src: np.ndarray, dst: np.ndarray,
                         grid: np.ndarray) -> "WarpMap":
    """Create a monotone warp that matches *src* landmarks to *dst*."""
    order = np.argsort(src)
    src_sorted = np.asarray(src, dtype=float)[order]
    dst_sorted = np.asarray(dst, dtype=float)[order]

    src_unique, idx = np.unique(src_sorted, return_index=True)
    dst_unique = dst_sorted[idx]

    src_ext = np.concatenate(([grid[0]], _ensure_strictly_increasing(src_unique),
                              [grid[-1]]))
    dst_ext = np.concatenate(([grid[0]], _ensure_strictly_increasing(dst_unique),
                              [grid[-1]]))
    src_ext = _ensure_strictly_increasing(src_ext)
    dst_ext = _ensure_strictly_increasing(dst_ext)

    warp_vals = np.interp(grid, src_ext, dst_ext)
    warp_vals = _ensure_strictly_increasing(warp_vals)

    forward = PchipInterpolator(grid, warp_vals, extrapolate=True)
    inverse = PchipInterpolator(warp_vals, grid, extrapolate=True)
    return WarpMap(forward=forward, inverse=inverse, grid=grid)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def fill_landmark_matrix(
    peaks: List[Sequence[float]],
    valleys: List[Sequence[float]],
    align_type: str = "negPeak_valley_posPeak",
    midpoint_type: str = "valley",
    neg_thr: float | None = None,
    return_observed: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Python port of the ADTnorm `landmark_fill_na()` routine.

    When ``return_observed`` is True the function returns a tuple containing
    the filled landmark matrix and a boolean mask flagging which entries were
    observed in the raw peak/valley detections (``True``) versus imputed
    during the fill-in process (``False``).
    """
    if align_type not in {
        "negPeak",
        "negPeak_valley",
        "negPeak_valley_posPeak",
        "valley",
    }:
        raise ValueError(f"Unknown align_type '{align_type}'")

    n = len(peaks)
    neg = np.full(n, np.nan)
    val = np.full(n, np.nan)
    pos = np.full(n, np.nan)

    neg_obs = np.zeros(n, dtype=bool)
    val_obs = np.zeros(n, dtype=bool)
    pos_obs = np.zeros(n, dtype=bool)

    for i, pk in enumerate(peaks):
        if pk:
            neg[i] = pk[0]
            neg_obs[i] = True
            pos[i] = pk[-1]
            pos_obs[i] = len(pk) > 1

    for i, vl in enumerate(valleys):
        if vl:
            val[i] = vl[0]
            val_obs[i] = True

    neg_thr = neg_thr if neg_thr is not None else np.arcsinh(10 / 5 + 1)

    # valley-only mode
    if align_type == "valley":
        out = val[:, None]
        nan_val = np.isnan(out[:, 0])
        out[nan_val, 0] = neg_thr
        if return_observed:
            obs = val_obs[:, None]
            return out, obs
        return out

    has_val_orig = np.isfinite(val).any()
    has_pos_orig = np.isfinite(pos).any()

    # If the cohort truly has a single landmark fall back to that mode.
    if not has_val_orig and not has_pos_orig:
        out = neg[:, None]
        nan_neg = np.isnan(out[:, 0])
        default = neg_thr
        if np.any(~nan_neg):
            default = _nanmedian(out[~nan_neg, 0], neg_thr)
        out[nan_neg, 0] = default
        if return_observed:
            obs = neg_obs[:, None]
            return out, obs
        return out

    # Build the full landmark table and then apply the R-style imputations.
    out = np.vstack([neg, val, pos]).T
    obs = np.vstack([neg_obs, val_obs, pos_obs]).T

    # valley NAs → negative threshold
    nan_val = np.isnan(out[:, 1])
    out[nan_val, 1] = neg_thr

    # negative peak NAs → half of the (filled) valley
    nan_neg = np.isnan(out[:, 0])
    out[nan_neg, 0] = out[nan_neg, 1] / 2.0

    # positive peak NAs → valley + cohort median offset
    nan_pos = np.isnan(out[:, 2])
    diff = out[:, 2] - out[:, 1]
    diff_med = _nanmedian(diff, 0.0)
    out[nan_pos, 2] = out[nan_pos, 1] + diff_med

    # honour the requested alignment regime
    if align_type == "negPeak":
        if return_observed:
            return out[:, :1], obs[:, :1]
        return out[:, :1]
    if align_type == "negPeak_valley" or (has_val_orig and not has_pos_orig):
        if return_observed:
            return out[:, :2], obs[:, :2]
        return out[:, :2]
    if return_observed:
        return out, obs
    return out


@dataclass
class WarpMap:
    """Container for the forward/inverse warp and auxiliary metadata."""

    forward: Callable[[Iterable[float]], np.ndarray]
    inverse: Callable[[Iterable[float]], np.ndarray]
    grid: Optional[np.ndarray] = None
    density: Optional[np.ndarray] = None

    def __call__(self, x: Iterable[float]) -> np.ndarray:
        return self.forward(np.asarray(x, dtype=float))


def build_warp_function(
    landmarks_src: Sequence[float],
    landmarks_tgt: Sequence[float],
) -> Callable[[Iterable[float]], np.ndarray]:
    """
    Compatibility wrapper that exposes the forward monotone warp used by the
    alignment routine.  The returned callable maps the source landmarks to the
    targets and extrapolates with the boundary slopes, mimicking the behaviour
    of `fda::landmarkreg` from the original R code.
    """
    ls = np.asarray(landmarks_src, dtype=float)
    lt = np.asarray(landmarks_tgt, dtype=float)
    if ls.size == 0 or np.any(np.isnan(ls)) or np.any(np.isnan(lt)):
        raise ValueError("Landmarks must be defined & numeric")

    if ls.size == 1:
        delta = float(lt[0] - ls[0])

        def forward(x, d=delta):
            return np.asarray(x, dtype=float) + d

        return forward

    grid = _build_common_grid([ls, lt], margin=0.0, n_points=max(ls.size + 2, 8))
    warp = _build_monotone_warp(ls, lt, grid)
    return warp.forward


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def align_distributions(
    counts: List[np.ndarray],
    peaks: List[Sequence[float]],
    valleys: List[Sequence[float]],
    align_type: str = "negPeak_valley_posPeak",
    landmark_matrix: Optional[np.ndarray] = None,
    target_landmark: Optional[Sequence[float]] = None,
) -> Tuple[List[np.ndarray], np.ndarray, List[WarpMap]]:
    """Align raw count vectors by warping their landmarks to a common target."""
    if landmark_matrix is None:
        lm, observed_mask = fill_landmark_matrix(
            peaks,
            valleys,
            align_type=align_type,
            return_observed=True,
        )
    else:
        lm = np.asarray(landmark_matrix, dtype=float)
        if lm.shape[0] != len(counts):
            raise ValueError("landmark_matrix rows ≠ #samples")
        observed_mask = np.isfinite(lm)
        if np.any(~observed_mask):
            _, nan_cols = np.where(~observed_mask)
            col_median = np.nanmedian(lm, axis=0)
            col_median = np.where(np.isnan(col_median), 0.0, col_median)
            lm[~observed_mask] = col_median[nan_cols]

    if target_landmark is None:
        tgt = np.empty(lm.shape[1], dtype=float)
        for j in range(lm.shape[1]):
            obs_col = observed_mask[:, j]
            if np.any(obs_col):
                tgt[j] = float(np.mean(lm[obs_col, j]))
            else:
                tgt[j] = float(np.mean(lm[:, j]))
    else:
        tgt = np.asarray(target_landmark, dtype=float)
        if tgt.size != lm.shape[1]:
            raise ValueError("target_landmark length ≠ #landmarks")

    tgt = np.atleast_1d(np.asarray(tgt, dtype=float))

    grid, lower_bound, upper_bound = _prepare_alignment_grid(counts, lm, tgt)
    densities = [_smooth_density(np.asarray(c, dtype=float), grid) for c in counts]

    if lm.shape[1] >= 2:
        valley = lm[:, 1]
        too_high = valley > upper_bound
        if np.any(too_high):
            lm[too_high, 1] = upper_bound

    if lm.shape[1] >= 1:
        neg = lm[:, 0]
        too_low = neg < lower_bound
        if np.any(too_low):
            lm[too_low, 0] = lower_bound

    warped_counts: List[np.ndarray] = []
    warped_landmark = np.empty_like(lm)
    warp_funs: List[WarpMap] = []

    for i, (c, l_src) in enumerate(zip(counts, lm)):
        c_arr = np.asarray(c, dtype=float)
        density = densities[i]
        valid = observed_mask[i]
        if not np.any(valid):
            def identity(x):
                return np.asarray(x, dtype=float)

            warp = WarpMap(
                forward=identity,
                inverse=identity,
                grid=grid,
                density=density,
            )
            warped_counts.append(c_arr.copy())
            warped_landmark[i] = l_src
            warp_funs.append(warp)
            continue

        src = np.asarray(l_src[valid], dtype=float)
        dst = np.asarray(tgt[valid], dtype=float)

        if src.size == 1:
            delta = float(src[0] - dst[0])

            def forward(x, d=delta):
                return np.asarray(x, dtype=float) - d

            def inverse(x, d=delta):
                return np.asarray(x, dtype=float) + d

            warp = WarpMap(forward=forward, inverse=inverse, grid=grid, density=density)
        else:
            warp = _build_monotone_warp(src, dst, grid)
            warp.density = density

        new_c = warp.forward(c_arr)
        missing = np.isnan(new_c)
        if np.any(missing):
            new_c[missing] = c_arr[missing]
        warped_counts.append(new_c)

        warped_landmark[i] = warp.forward(l_src)
        warp_funs.append(warp)

    return warped_counts, warped_landmark, warp_funs
