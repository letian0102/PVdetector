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
    src_sorted = np.sort(src)
    dst_sorted = dst[np.argsort(src)]

    src_unique, idx = np.unique(src_sorted, return_index=True)
    dst_unique = dst_sorted[idx]

    src_ext = np.concatenate(([grid[0]], _ensure_strictly_increasing(src_unique),
                              [grid[-1]]))
    dst_ext = np.concatenate(([grid[0]], _ensure_strictly_increasing(dst_unique),
                              [grid[-1]]))

    forward = PchipInterpolator(src_ext, dst_ext, extrapolate=True)
    inverse = PchipInterpolator(dst_ext, src_ext, extrapolate=True)
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
) -> np.ndarray:
    """Python port of the ADTnorm `landmark_fill_na()` routine."""
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

    for i, pk in enumerate(peaks):
        if pk:
            neg[i] = pk[0]
            pos[i] = pk[-1]

    for i, vl in enumerate(valleys):
        if vl:
            val[i] = vl[0]

    neg_thr = neg_thr if neg_thr is not None else np.arcsinh(10 / 5 + 1)

    # valley-only mode
    if align_type == "valley":
        out = val[:, None]
        nan_val = np.isnan(out[:, 0])
        out[nan_val, 0] = neg_thr
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
        return out

    # Build the full landmark table and then apply the R-style imputations.
    out = np.vstack([neg, val, pos]).T

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
        return out[:, :1]
    if align_type == "negPeak_valley" or (has_val_orig and not has_pos_orig):
        return out[:, :2]
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
        lm = fill_landmark_matrix(peaks, valleys, align_type=align_type)
    else:
        lm = np.asarray(landmark_matrix, dtype=float)
        if lm.shape[0] != len(counts):
            raise ValueError("landmark_matrix rows ≠ #samples")

    nan_mask = np.isnan(lm)
    if np.any(nan_mask):
        col_median = np.nanmedian(lm, axis=0)
        lm[nan_mask] = np.take(col_median, np.where(nan_mask)[1])

    tgt = (
        np.mean(lm, axis=0)
        if target_landmark is None
        else np.asarray(target_landmark, dtype=float)
    )

    grid = _build_common_grid(counts)
    densities = [_smooth_density(np.asarray(c, dtype=float), grid) for c in counts]

    warped_counts: List[np.ndarray] = []
    warped_landmark = np.empty_like(lm)
    warp_funs: List[WarpMap] = []

    for i, (c, l_src) in enumerate(zip(counts, lm)):
        c_arr = np.asarray(c, dtype=float)
        density = densities[i]
        valid = ~np.isnan(l_src)
        if not np.any(valid):
            warped_counts.append(c_arr.copy())
            warped_landmark[i] = l_src
            continue

        src = np.asarray(l_src[valid], dtype=float)
        dst = np.asarray(tgt[valid], dtype=float)

        if src.size == 1:
            delta = float(dst[0] - src[0])

            def forward(x, d=delta):
                return np.asarray(x, dtype=float) + d

            def inverse(x, d=delta):
                return np.asarray(x, dtype=float) - d

            warp = WarpMap(forward=forward, inverse=inverse, grid=grid, density=density)
        else:
            warp = _build_monotone_warp(src, dst, grid)
            warp.density = density

        new_c = warp.forward(c_arr)
        new_c[np.isnan(c_arr)] = np.nan
        warped_counts.append(new_c)

        wl = np.full_like(l_src, np.nan)
        wl[valid] = warp.forward(src)
        warped_landmark[i] = wl
        warp_funs.append(warp)

    return warped_counts, warped_landmark, warp_funs
