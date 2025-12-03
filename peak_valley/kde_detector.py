from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import fftconvolve, find_peaks

from .roughness import find_bw_for_roughness

from .consistency import _enforce_valley_rule

__all__ = ["kde_peaks_valleys", "quick_peak_estimate"]


# ----------------------------------------------------------------------
# peak_valley/kde_detector.py
import numpy as np

# peak_valley/kde_detector.py  (only the helper changed)
# ----------------------------------------------------------------------
def _merge_close_peaks(
    xs: np.ndarray,
    ys: np.ndarray,
    p_idx: np.ndarray,
    min_x_sep: float = 0.4,
    min_valley_drop: float = 0.15,
) -> np.ndarray:
    """
    Collapse peaks that violate the minimum separation rule.

    Whenever two peaks lie closer than ``min_x_sep`` we keep only the taller
    one, regardless of how deep the valley between them is.  This matches the
    user expectation that "too-close" peaks should always be reduced to their
    single dominant representative.
    """

    _ = min_valley_drop  # kept for API compatibility with previous heuristic

    if p_idx.size < 2:
        return p_idx

    if not np.isfinite(min_x_sep) or min_x_sep <= 0:
        return np.sort(p_idx)

    # Work on a mutable copy so that we can drop indices in-place while
    # scanning through the sorted peak list.
    keep = list(np.sort(p_idx))

    i = 0
    while i < len(keep) - 1:
        left = keep[i]
        right = keep[i + 1]
        if xs[right] - xs[left] >= min_x_sep:
            i += 1
            continue

        if ys[left] >= ys[right]:
            del keep[i + 1]
        else:
            del keep[i]

        if i > 0:
            i -= 1

    return np.asarray(keep, dtype=int)


def _peak_height(xs: np.ndarray, ys: np.ndarray, peak_x: float) -> float:
    """Return KDE height at the closest grid point to ``peak_x``."""

    if xs.size == 0 or ys.size == 0:
        return float("nan")

    idx = int(np.argmin(np.abs(xs - peak_x)))
    idx = max(0, min(idx, ys.size - 1))
    return float(ys[idx])


def _enforce_min_separation(
    peaks_x: list[float],
    xs: np.ndarray,
    ys: np.ndarray,
    min_x_sep: float | None,
) -> list[float]:
    """Drop or swap peaks so every pair obeys ``min_x_sep`` spacing."""

    if (
        min_x_sep is None
        or not np.isfinite(min_x_sep)
        or min_x_sep <= 0
        or len(peaks_x) < 2
    ):
        return sorted(peaks_x)

    keep = sorted(float(px) for px in peaks_x)
    heights = [_peak_height(xs, ys, px) for px in keep]

    i = 0
    while i < len(keep) - 1:
        left_x = keep[i]
        right_x = keep[i + 1]

        if right_x - left_x >= min_x_sep:
            i += 1
            continue

        left_height = heights[i]
        right_height = heights[i + 1]

        if left_height >= right_height:
            del keep[i + 1]
            del heights[i + 1]
        else:
            del keep[i]
            del heights[i]
            if i > 0:
                i -= 1
            continue

        if i > 0:
            i -= 1

    return keep


def _resolve_peak_valley_conflicts(
    xs: np.ndarray,
    ys: np.ndarray,
    peaks_idx: list[int],
    *,
    min_x_sep: float | None,
    drop_frac: float,
    first_valley: str,
) -> tuple[list[int], list[float]]:
    """Ensure valleys clear neighbouring peaks, pruning peaks when needed."""

    peaks_idx = sorted(set(int(i) for i in peaks_idx))
    if not peaks_idx:
        return [], []

    grid_dx = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    if not np.isfinite(grid_dx) or grid_dx <= 0:
        grid_dx = 1.0

    min_gap: float | None
    if min_x_sep is None or not np.isfinite(min_x_sep) or min_x_sep <= 0:
        min_gap = None
    else:
        min_gap = 0.5 * float(min_x_sep)

    def _single_peak_valley(idx: int) -> float:
        if first_valley == "drop":
            val = _first_valley_drop(xs, ys, idx, drop_frac)
        else:
            val = _first_valley_slope(
                xs, ys, idx, None, min_sep=min_x_sep, drop_frac=drop_frac
            )
            if val is None:
                val = _first_valley_drop(xs, ys, idx, drop_frac)

        if val is None:
            # Fallback: step one grid cell to the right.
            if idx + 1 < xs.size:
                val = float(xs[idx + 1])
            else:
                val = float(xs[idx]) + grid_dx

        if min_gap is not None:
            base = float(xs[idx])
            if abs(val - base) < min_gap:
                val = base + min_gap
        return float(val)

    def _pair_valley(left: int, right: int, prefer_first: bool) -> float:
        if prefer_first and first_valley != "drop":
            val = _first_valley_slope(
                xs, ys, left, right, min_sep=min_x_sep, drop_frac=drop_frac
            )
            if val is None:
                val = _valley_between(xs, ys, left, right, drop_frac)
        elif prefer_first and first_valley == "drop":
            val = _valley_between(xs, ys, left, right, drop_frac)
        else:
            val = _valley_between(xs, ys, left, right, drop_frac)
        return float(val)

    def _recompute_valleys(current: list[int]) -> list[float]:
        if len(current) > 1:
            vals: list[float] = []
            left, right = current[0], current[1]
            vals.append(_pair_valley(left, right, True))
            for l_idx, r_idx in zip(current[1:-1], current[2:]):
                vals.append(_pair_valley(l_idx, r_idx, False))
            return vals
        return [_single_peak_valley(current[0])]

    valleys = _recompute_valleys(peaks_idx)

    if min_gap is None:
        return peaks_idx, valleys

    while True:
        remove_idx: int | None = None
        recalc_idx: int | None = None

        for i, vx in enumerate(valleys):
            left_idx = peaks_idx[min(i, len(peaks_idx) - 1)]
            left_x = float(xs[left_idx])

            if abs(vx - left_x) < min_gap:
                recalc_idx = i
                break

            right_idx = peaks_idx[i + 1] if i + 1 < len(peaks_idx) else None
            if right_idx is not None:
                right_x = float(xs[right_idx])
                if abs(right_x - vx) < min_gap:
                    remove_idx = i + 1
                    break

        if remove_idx is not None:
            if remove_idx < len(peaks_idx):
                del peaks_idx[remove_idx]
            if not peaks_idx:
                break
            valleys = _recompute_valleys(peaks_idx)
            continue

        if recalc_idx is not None:
            left_idx = peaks_idx[min(recalc_idx, len(peaks_idx) - 1)]
            right_idx = peaks_idx[recalc_idx + 1] if recalc_idx + 1 < len(peaks_idx) else None

            if right_idx is None:
                valleys[recalc_idx] = _single_peak_valley(left_idx)
            else:
                candidate = _valley_between(xs, ys, left_idx, right_idx, drop_frac)
                left_x = float(xs[left_idx])
                right_x = float(xs[right_idx])

                if (
                    abs(candidate - left_x) < min_gap
                    or abs(right_x - candidate) < min_gap
                ):
                    lo = left_x + min_gap
                    hi = right_x - min_gap
                    if hi <= lo:
                        candidate = (left_x + right_x) / 2.0
                    else:
                        candidate = 0.5 * (lo + hi)

                valleys[recalc_idx] = float(candidate)
            continue

        break

    if not valleys and peaks_idx:
        valleys = _recompute_valleys(peaks_idx)

    return peaks_idx, valleys

def _mostly_small_discrete(x: np.ndarray, threshold: float = 0.9) -> bool:
    """Heuristic to catch almost-discrete samples near zero (0..3).

    Parameters
    ----------
    x : np.ndarray
        Input data (1‑D array).
    threshold : float, optional
        Fraction of points within ``0‥3`` required to trigger.

    Returns
    -------
    bool
        ``True`` if the majority of values are integers 0–3.
    """
    x = np.asarray(x, float)
    if x.size == 0:
        return False

    good = x[np.isfinite(x)]
    if good.size == 0:
        return False

    mask = (good >= 0) & (good <= 3)
    if mask.sum() / good.size < threshold:
        return False

    uniq = np.unique(np.round(good[mask]))
    return uniq.size <= 4


def _prepare_boundary_kde_data(
    x: np.ndarray,
    *,
    boundary_correction: bool,
    boundary_lower: float | None,
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Return data/weights for boundary-aware KDE training.

    Reflection is enabled when ``boundary_correction`` is ``True`` and all
    finite samples lie at or above ``boundary_lower``.  We mirror the data about
    the boundary with equal mass on the original and reflected points to reduce
    edge bias without changing the total probability mass.
    """

    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return finite, None, False

    use_boundary = False
    if boundary_correction and boundary_lower is not None and np.isfinite(boundary_lower):
        boundary_lower = float(boundary_lower)

        # Avoid reflecting when the distribution sits well away from the boundary.
        # Only turn on correction if the lower tail actually hugs the boundary and
        # a meaningful fraction of points live there.
        low_q, high_q = np.percentile(finite, [5, 95])
        span = float(high_q - low_q)
        if not np.isfinite(span) or span <= 0:
            span = float(np.std(finite, ddof=1))
        if not np.isfinite(span) or span <= 0:
            span = float(np.max(finite) - np.min(finite))

        tol = max(1e-6, 0.02 * span)
        lower_tail = float(np.percentile(finite, 2))
        frac_near = float(np.mean(finite <= boundary_lower + tol))

        if (
            lower_tail >= boundary_lower - 1e-9
            and (lower_tail - boundary_lower) <= tol
            and frac_near >= 0.05
        ):
            use_boundary = True

    if not use_boundary:
        return finite, None, False

    mirrored = (2.0 * float(boundary_lower)) - finite
    weight = 0.5 / finite.size
    weights = np.full(finite.size * 2, weight)
    data = np.concatenate([finite, mirrored])
    return data, weights, True


def _stable_sample(values: np.ndarray) -> np.ndarray:
    """Ensure enough spread for KDE when data collapse to a point."""

    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return vals

    if vals.size == 1 or np.allclose(vals, vals[0]):
        base = float(vals[0])
        span = max(1e-6, 0.001 * max(1.0, abs(base)))
        offsets = np.linspace(-0.5, 0.5, num=max(2, vals.size)) * span
        vals = base + offsets

    return vals


def _fft_gaussian_kde(
    x: np.ndarray,
    xs: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Approximate Gaussian KDE via FFT-based convolution.

    Parameters
    ----------
    x : np.ndarray
        1-D sample array (finite values only).
    xs : np.ndarray
        Evaluation grid (uniform spacing assumed).
    bandwidth : float
        Scalar bandwidth of the 1-D Gaussian kernel.

    Returns
    -------
    np.ndarray
        KDE evaluated on ``xs``.
    """

    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return np.zeros_like(xs)

    if xs.size < 2:
        return np.zeros_like(xs)

    grid_dx = float(xs[1] - xs[0])
    if not np.isfinite(grid_dx) or grid_dx <= 0:
        return np.zeros_like(xs)

    # Histogram aligned with the KDE grid (bin centers = xs)
    edges = np.linspace(
        xs[0] - 0.5 * grid_dx,
        xs[-1] + 0.5 * grid_dx,
        xs.size + 1,
    )
    counts, _ = np.histogram(x, bins=edges)

    if not counts.any():
        return np.zeros_like(xs)

    # Truncate kernel at ±4σ (covers >99.99 % of mass)
    max_radius = max(1, int(np.ceil(4.0 * bandwidth / grid_dx)))
    max_radius = min(max_radius, xs.size // 2)
    offsets = np.arange(-max_radius, max_radius + 1)
    kernel = np.exp(-0.5 * ((offsets * grid_dx) / bandwidth) ** 2)
    kernel /= kernel.sum()

    smooth = fftconvolve(counts, kernel, mode="same")
    return smooth / (x.size * grid_dx)


def _evaluate_kde(
    sample: np.ndarray,
    xs: np.ndarray,
    kde: gaussian_kde,
) -> np.ndarray:
    """Evaluate KDE, switching to FFT when the grid is large."""

    n = sample.size
    if n == 0:
        return np.zeros_like(xs)

    # ``gaussian_kde`` evaluation cost grows with ``n × len(xs)``.
    brute_cost = n * xs.size

    # ``kde.factor`` already incorporates any ``set_bandwidth`` calls.
    sample_std = float(np.std(sample, ddof=1)) if n > 1 else 0.0
    bandwidth = kde.factor * sample_std

    if brute_cost <= 5_000_000 or bandwidth <= 0:
        return kde(xs)

    return _fft_gaussian_kde(sample, xs, bandwidth)


def _first_valley_slope(
    xs: np.ndarray,
    ys: np.ndarray,
    p_left: int,
    p_right: int | None = None,
    *,
    min_sep: float | None = None,
    drop_frac: float = 0.10,
) -> float | None:
    """Pick first valley via maximum increase in slope.

    The valley is constrained to lie at or before the local minimum
    between the first peak and the next peak (if given).  If the point of
    maximum slope change occurs after the local minimum, the minimum is
    chosen instead.  A minimum separation from the first peak can be
    enforced via ``min_sep``.

    Parameters
    ----------
    xs, ys : np.ndarray
        Grid and KDE values.
    p_left : int
        Index of the first peak.
    p_right : int | None, optional
        Index of the next peak.  If ``None`` the search extends to the
        end of the grid.
    min_sep : float | None, optional
        Minimum distance allowed between the peak at ``p_left`` and the
        returned valley.  When ``None`` a small fraction of the
        peak-to-peak distance is used.
    drop_frac : float, optional
        Drop-fraction passed to :func:`_valley_between` for fallback
        valley selection.

    Returns
    -------
    float | None
        X‑coordinate of the valley or ``None`` if the slope never
        increases.
    """

    if p_right is None:
        right = len(xs) - 1
    else:
        if p_right - p_left <= 2:
            return None
        right = p_right

    if right <= p_left + 1:
        return None

    span = xs[right] - xs[p_left]
    if not np.isfinite(span) or span <= 0:
        span = float(xs[1] - xs[0]) if xs.size > 1 else 0.0

    if min_sep is not None and min_sep > 0:
        min_sep_abs = float(min_sep)
    else:
        min_sep_abs = 0.05 * span

    if xs.size > 1:
        min_sep_abs = max(min_sep_abs, float(xs[1] - xs[0]))

    start_idx = p_left + 1
    while start_idx < right and xs[start_idx] - xs[p_left] < min_sep_abs:
        start_idx += 1

    d2y = np.diff(ys, n=2)
    candidate_idx: int | None = None
    if start_idx < right:
        seg = d2y[start_idx - 1 : right - 1]
        if seg.size:
            rel = int(np.argmax(seg))
            if seg[rel] > 0:
                candidate_idx = start_idx + rel

    dy = np.diff(ys[p_left + 1 : right + 1])
    turn = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0]
    min_idx = p_left + 2 + int(turn[0]) if turn.size else None

    if candidate_idx is None:
        if min_idx is not None and xs[min_idx] - xs[p_left] >= min_sep_abs:
            candidate_idx = min_idx
        elif p_right is not None:
            return _valley_between(xs, ys, p_left, p_right, drop_frac)
        else:
            seg = ys[start_idx : right + 1]
            if seg.size:
                candidate_idx = start_idx + int(np.argmin(seg))
            else:
                candidate_idx = max(p_left + 1, min(right, start_idx))

    if min_idx is not None and candidate_idx > min_idx:
        candidate_idx = min_idx

    if p_right is not None:
        candidate_idx = max(p_left + 1, min(candidate_idx, right - 1))
    else:
        candidate_idx = max(p_left + 1, min(candidate_idx, right))

    if xs[candidate_idx] - xs[p_left] < min_sep_abs:
        if min_idx is not None and xs[min_idx] - xs[p_left] >= min_sep_abs:
            candidate_idx = min_idx
        elif p_right is not None:
            return _valley_between(xs, ys, p_left, p_right, drop_frac)
        else:
            seg = ys[start_idx : right + 1]
            if seg.size:
                rel = int(np.argmin(seg))
                candidate_idx = start_idx + rel

    return float(xs[candidate_idx])


def _first_valley_drop(xs: np.ndarray,
                       ys: np.ndarray,
                       p_left: int,
                       drop_frac: float) -> float | None:
    """Pick first valley via drop‑fraction rule.

    Searches to the right of ``p_left`` for the first turning point where
    the KDE falls below ``drop_frac`` of the peak height.  If no such point
    exists, the absolute minimum in the tail is returned.
    """
    seg = np.arange(p_left + 1, len(xs))
    if seg.size == 0:
        return None

    y_thr = drop_frac * ys[p_left]
    dy = np.diff(ys[seg])
    turn = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0] + 1
    turn = turn[ys[seg[turn]] <= y_thr]

    if turn.size:
        idx_val = seg[turn[0]]
    else:
        idx_val = seg[np.argmin(ys[seg])]
    return float(xs[idx_val])


def _valley_between(xs: np.ndarray,
                    ys: np.ndarray,
                    p_left: int,
                    p_right: int,
                    drop_frac: float = 0.10) -> float:
    """
    Valley chooser that guarantees **one** valley between every pair of
    peaks and never overlaps peaks.

    Order of preference
    -------------------
    1. *Real dip* — deepest minimum that falls below `drop_frac` of the
       lower-peak height (classic rule).
    2. *Turning-point rule* from previous version (first slope –→+ bend).
    3. *Steep-slope rule* (NEW):
       • if right-hand peak is HIGHER   →  pick the grid point where the
         **up-slope**  (first derivative) is most positive;  
       • if right-hand peak is LOWER    →  pick the grid point where the
         **down-slope** (first derivative) is most negative.
    A small safety shift (`grid_half`) is applied at the end so the
    valley is at least ~½ % of the interval away from either peak.
    """
    seg = np.arange(p_left + 1, p_right)           # interior points
    if seg.size == 0:                              # adjacent peaks
        return float(xs[(p_left + p_right) // 2])

    # ── 1. genuine dip ───────────────────────────────────────────────
    mins, _ = find_peaks(-ys[seg])
    if mins.size:
        idx_dip = seg[mins[np.argmin(ys[seg][mins])]]
        if ys[idx_dip] < drop_frac * min(ys[p_left], ys[p_right]):
            valley_idx = idx_dip
        else:
            valley_idx = None
    else:
        valley_idx = None

    # ── 2. previous turning-point rule ───────────────────────────────
    if valley_idx is None:
        valley_tp = _turning_valley(xs, ys, p_left, p_right, drop_frac)
        valley_idx = int(np.searchsorted(xs, valley_tp))

    # ── 3. NEW steep-slope fallback  (only if still bad) ─────────────
    bad = (valley_idx <= p_left) or (valley_idx >= p_right)
    if bad:
        dy = np.diff(ys)                                    # first-derivative
        # ----- ignore a small slice close to each peak -----------------
        edge_frac  = 0.05                                   # 5 % of interval
        skip       = max(1, int(edge_frac * (p_right - p_left)))
        cand_left  = p_left + 1 + skip                      # first usable grid
        cand_right = p_right - skip                         # last  usable grid
        if cand_left >= cand_right:                         # interval too thin
            cand_left, cand_right = p_left + 1, p_right - 1

        dy_seg = dy[cand_left : cand_right]                 # safe interior

        if ys[p_right] > ys[p_left]:                        # right peak higher
            rel = np.argmax(dy_seg)                         # steepest ↑ slope
        else:                                               # right peak lower
            rel = np.argmin(dy_seg)                         # steepest ↓ slope
        valley_idx = cand_left + rel

        # ── 4. safety-shift so valley ≠ peak ──────────────────────────
    min_sep   = 0.05 * (xs[p_right] - xs[p_left])      # 10 % of peak gap
    if (xs[valley_idx] - xs[p_left]  < min_sep or
        xs[p_right]  - xs[valley_idx] < min_sep):

        valley_idx = (p_left + p_right) // 2           # exact mid-point

    return float(xs[valley_idx])



# ----------------------------------------------------------------------
def _normalise_bandwidth(bw: str | float | Callable | None) -> str | float | Callable:
    """Coerce an arbitrary bandwidth specification to something scipy accepts."""

    if bw is None:
        return "scott"
    if callable(bw):
        return bw
    if isinstance(bw, (int, float)):
        return float(bw)
    if isinstance(bw, str):
        label = bw.strip()
        if not label:
            return "scott"
        lowered = label.lower()
        if lowered in {"scott", "silverman", "roughness"}:
            return lowered
        try:
            return float(label)
        except ValueError:
            warnings.warn(
                f"Invalid bandwidth '{bw}' provided; defaulting to 'scott'",
                RuntimeWarning,
                stacklevel=2,
            )
            return "scott"
    warnings.warn(
        "Unrecognised bandwidth type; defaulting to 'scott'",
        RuntimeWarning,
        stacklevel=2,
    )
    return "scott"


def kde_peaks_valleys(
    data           : np.ndarray,
    n_peaks        : int  | None = None,
    prominence     : float        = 0.05,
    bw             : str  | float = "scott",
    min_width      : int  | None  = None,
    grid_size      : int          = 20_000,
    drop_frac      : float        = 0.10,
    min_x_sep      : float        = 1.0,   # absolute, same units as `data`
    min_valley_drop: float        = 0.15,
    curvature_thresh: float | None = None,
    turning_peak   : bool = False,
    first_valley   : str = "slope",
    boundary_correction: bool = True,
    boundary_lower: float | None = 0.0,
) -> tuple[list[float], list[float], np.ndarray, np.ndarray, float | None]:
    x = np.asarray(data, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [], [], np.array([]), np.array([]), None

    # ---------- KDE grid ----------
    if x.size > 10_000:
        x = np.random.choice(x, 10_000, replace=False)

    kde_data, kde_weights, used_boundary = _prepare_boundary_kde_data(
        x,
        boundary_correction=boundary_correction,
        boundary_lower=boundary_lower,
    )
    bw_use = _normalise_bandwidth(bw)
    if bw_use == "roughness":
        try:
            bw_use = _normalise_bandwidth(
                find_bw_for_roughness(x, boundary_lower=boundary_lower)
            )
        except Exception as exc:
            warnings.warn(
                "Roughness bandwidth search failed; defaulting to 'scott'"
                f" ({exc})",
                RuntimeWarning,
                stacklevel=2,
            )
            bw_use = "scott"

    kde_data = _stable_sample(kde_data)
    try:
        kde = gaussian_kde(kde_data, bw_method=bw_use, weights=kde_weights)
    except ValueError:
        warnings.warn(
            "Invalid KDE bandwidth provided; falling back to 'scott'",
            RuntimeWarning,
            stacklevel=2,
        )
        bw_use = "scott"
        kde = gaussian_kde(kde_data, bw_method=bw_use, weights=kde_weights)
    if _mostly_small_discrete(x):
        kde.set_bandwidth(kde.factor * 4.0)
    sample_for_bandwidth = kde_data if kde_data.size else x
    h = kde.factor * sample_for_bandwidth.std(ddof=1)
    if not np.isfinite(h) or h <= 0:
        span = float(np.max(x) - np.min(x)) if x.size else 1.0
        h = max(1e-3, 0.1 * span)

    # When boundary correction is active we still extend the grid slightly below
    # the boundary so the reflected mass is visible in ridge plots (otherwise the
    # left half of the first peak can look "cut off" at zero).  The lower bound
    # stays anchored to the data range when reflection is disabled.
    if used_boundary and boundary_lower is not None:
        candidate_min = float(np.min(x) - h)
        grid_min = min(candidate_min, float(boundary_lower) - 0.1 * abs(h))
    else:
        grid_min = float(np.min(x) - h)
    grid_max = float(np.max(x) + h)
    if grid_max <= grid_min:
        grid_max = grid_min + max(h, 1e-3)

    xs = np.linspace(
        grid_min,
        grid_max,
        min(grid_size, max(4000, 4 * x.size)),
    )
    ys = _evaluate_kde(kde_data, xs, kde)

    # ---------- primary peaks ----------

    grid_dx  = xs[1] - xs[0]
    distance = None                       # (NEW)  hard x-spacing
    if min_x_sep is not None:
        # ``scipy.signal.find_peaks`` requires ``distance`` >= 1.  For very
        # small ``min_x_sep`` relative to the grid spacing the integer
        # division above could yield ``0`` which would raise a ``ValueError``.
        # Clamp the value so that ``find_peaks`` always receives a valid
        # minimum distance.
        distance = max(1, int(math.ceil(min_x_sep / grid_dx)))
    kw = {"prominence": prominence}
    if min_width:
        kw["width"] = min_width
    locs, _ = find_peaks(ys, **kw, distance=distance)

    if turning_peak and curvature_thresh and curvature_thresh > 0:
        dy  = np.gradient(ys, xs)                 # 1st derivative
        d2y = np.gradient(dy,  xs)                # 2nd derivative
        slope_tol = 0.05 * np.max(np.abs(dy))     # “almost flat”

        turning = np.where(
            (np.abs(dy) < slope_tol) &           # near-flat slope
            (d2y < -curvature_thresh)            # concave-down
        )[0]

        # keep only the strongest point inside every min_x_sep window
        if turning.size:
            strongest = [turning[0]]
            for idx in turning[1:]:
                if xs[idx] - xs[strongest[-1]] >= min_x_sep:
                    strongest.append(idx)
                elif ys[idx] > ys[strongest[-1]]:     # higher shoulder wins
                    strongest[-1] = idx
            locs = np.unique(np.concatenate([locs, strongest]))

    locs = _merge_close_peaks(xs, ys, locs,
            min_x_sep=min_x_sep, min_valley_drop=min_valley_drop)

    if locs.size == 0 and n_peaks is None:
        return [], [], xs, ys, float(h)

    # keep tallest as before
    if n_peaks is None or locs.size >= n_peaks:
        peaks_idx = np.sort(locs) if n_peaks is None else \
                    np.sort(locs[np.argsort(ys[locs])[-n_peaks:]])
    else:
        peaks_idx = np.sort(locs)
    
    peaks_x = _enforce_min_separation(xs[peaks_idx].tolist(), xs, ys, min_x_sep)

    peaks_idx = [int(np.argmin(np.abs(xs - px))) for px in peaks_x]

    # ---------- GMM fallback (unchanged) ----------
    if n_peaks is not None and len(peaks_x) < n_peaks:
        from .gmm_fallback import gmm_component_means
        gmm_means = gmm_component_means(x, n_peaks)

        # merge (uniqueness tolerance = 5 % IQR)
        iqr  = np.subtract(*np.percentile(x, [75, 25]))
        tol  = 0.05 * iqr if iqr else 0.1
        def uniq(v, L): return all(abs(v - w) > tol for w in L)

        for m in gmm_means:
            if uniq(m, peaks_x):
                peaks_x.append(m)

        if len(peaks_x) > n_peaks:
            # retain tallest KDE peaks before enforcing separation
            heights = [(_peak_height(xs, ys, px), px) for px in peaks_x]
            heights.sort(reverse=True)
            peaks_x = [px for _, px in heights[: n_peaks]]

    # Enforce separation a final time after any fallback adjustments.
    peaks_x = _enforce_min_separation(peaks_x, xs, ys, min_x_sep)

    if n_peaks is not None and peaks_x and len(peaks_x) < n_peaks:
        # Fall back to a single dominant peak whenever we cannot honour the
        # requested count under the minimum-separation constraint.
        dominant = max(peaks_x, key=lambda px: _peak_height(xs, ys, px))
        peaks_x = [dominant]

    # ---------- *re-derive* indices for every peak we now have ----------
    peaks_idx = [int(np.argmin(np.abs(xs - px))) for px in peaks_x]

    # ---------- valleys ----------
    valleys_x: list[float] = []
    if len(peaks_idx) > 1:
        p0, p1 = peaks_idx[0], peaks_idx[1]
        if first_valley == "drop":
            val = _valley_between(xs, ys, p0, p1, drop_frac)
        else:
            val = _first_valley_slope(
                xs, ys, p0, p1, min_sep=min_x_sep, drop_frac=drop_frac
            )
            if val is None:
                val = _valley_between(xs, ys, p0, p1, drop_frac)
        valleys_x.append(val)

        # remaining valleys (if any): classic rule
        for left, right in zip(peaks_idx[1:-1], peaks_idx[2:]):
            valleys_x.append(
                _valley_between(xs, ys, left, right, drop_frac)
            )
    elif len(peaks_idx) == 1:
        p0 = peaks_idx[0]
        if first_valley == "drop":
            val = _first_valley_drop(xs, ys, p0, drop_frac)
        else:
            val = _first_valley_slope(
                xs, ys, p0, None, min_sep=min_x_sep, drop_frac=drop_frac
            )
        if val is not None:
            valleys_x.append(val)

    # --- enforce valley/peak relationship ----------------------------------
    peaks_idx, valleys_x = _resolve_peak_valley_conflicts(
        xs,
        ys,
        peaks_idx,
        min_x_sep=min_x_sep,
        drop_frac=drop_frac,
        first_valley=first_valley,
    )
    peaks_x = [float(xs[idx]) for idx in peaks_idx]
    peaks_x = np.round(peaks_x, 10).tolist()
    valleys_x = _enforce_valley_rule(peaks_x, valleys_x)
    if not valleys_x and peaks_x:
        # Guarantee at least one valley for single-peak outputs.
        idx0 = int(np.argmin(np.abs(xs - peaks_x[0])))
        valleys_x = [_resolve_peak_valley_conflicts(
            xs,
            ys,
            [idx0],
            min_x_sep=min_x_sep,
            drop_frac=drop_frac,
            first_valley=first_valley,
        )[1][0]]

    return peaks_x, valleys_x, xs, ys, float(h)

# ----------------------------------------------------------------------
def quick_peak_estimate(
    counts: np.ndarray,
    prominence: float,
    bw: str | float,
    min_width: int | None,
    grid_size: int,
) -> tuple[int, bool]:
    """
    *Unchanged* quick heuristic (dual-prominence trick).
    """
    p1, *_ = kde_peaks_valleys(counts, None, prominence,
                               bw, min_width, grid_size)
    p2, *_ = kde_peaks_valleys(counts, None, prominence / 2,
                               bw, min_width, grid_size)
    return len(p1), (len(p1) == len(p2) and len(p1) > 0)

def _turning_valley(
    xs: np.ndarray,
    ys: np.ndarray,
    p_left: int,
    p_right: int,
    drop_frac: float
) -> float:
    """
    Valley between two peaks = first turning-point where KDE switches
    from falling to rising **and** is already below `drop_frac` of the
    lower of the two peak heights.

    Fallback = absolute minimum on the interval (never fails).
    """
    y_thr  = drop_frac * min(ys[p_left], ys[p_right])
    seg    = np.arange(p_left + 1, p_right)               # open interval
    dy     = np.diff(ys[seg])                             # slope between grid pts

    # indices in `seg` (except last) where slope changes (-) → (+)
    turn   = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0] + 1
    turn   = turn[ys[seg[turn]] <= y_thr]                 # depth rule

    if turn.size:                                         # first real gutter
        idx_val = seg[turn[0]]
    else:                                                 # rare edge-case
        idx_val = seg[np.argmin(ys[seg])]

    return float(xs[idx_val])
