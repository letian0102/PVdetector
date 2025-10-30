from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, fftconvolve, peak_prominences

from .consistency import _enforce_valley_rule

try:  # NumPy ≥ 2.1
    from numpy import get_array_backend as _np_get_array_backend  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    _np_get_array_backend = None


def get_array_backend(*arrays, **kwargs):
    """Return the array backend associated with ``arrays`` (NumPy fallback)."""

    if _np_get_array_backend is None:
        return np

    try:
        return _np_get_array_backend(*arrays, **kwargs)
    except TypeError:
        # ``get_array_backend`` without ``default`` argument (NumPy 2.0) raises
        # ``TypeError`` when no array-like inputs are provided.  Fall back to
        # the standard NumPy module in that case.
        return np

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
    Collapse spurious twin-peaks produced by flat tops / wiggles.

    A pair (i,j) is merged if
      • xs[j] − xs[i]  <  min_x_sep        AND
      • the deepest point between them is higher than
        (1 − min_valley_drop) × min(ys[i], ys[j]).
    """
    if p_idx.size < 2:
        return p_idx

    keep = []
    i = 0
    while i < len(p_idx):
        j = i + 1
        winner = p_idx[i]
        while j < len(p_idx):
            # break if they are already far enough apart
            if xs[p_idx[j]] - xs[winner] >= min_x_sep:
                break

            # valley height between the two candidates
            lo, hi = p_idx[j - 1], p_idx[j]
            valley_h = ys[lo:hi + 1].min()

            # if the dip is shallow → keep the taller one only
            if valley_h > (1 - min_valley_drop) * min(ys[winner], ys[p_idx[j]]):
                if ys[p_idx[j]] > ys[winner]:
                    winner = p_idx[j]
                j += 1             # look at the next neighbour
            else:
                break              # real valley – stop merging chain

        keep.append(winner)
        i = j

    return np.asarray(sorted(keep), dtype=int)


def _collapse_with_bandwidth(
    xs: np.ndarray,
    ys: np.ndarray,
    p_idx: np.ndarray,
    bandwidth: float,
    min_x_sep: float | None,
    min_valley_drop: float,
    data_iqr: float,
) -> np.ndarray:
    """Merge peaks that cannot be sustained under the effective bandwidth."""

    if p_idx.size < 2:
        return p_idx

    if not np.isfinite(bandwidth) or bandwidth <= 0 or xs.size < 2:
        return p_idx

    grid_dx = float(xs[1] - xs[0])
    if not np.isfinite(grid_dx) or grid_dx <= 0:
        return p_idx

    # Require wider separation when smoothing is heavy.
    collapse_sep = 0.75 * bandwidth
    if min_x_sep is not None:
        collapse_sep = max(collapse_sep, float(min_x_sep))
    else:
        collapse_sep = max(collapse_sep, 0.0)

    spread = float(data_iqr)
    if not np.isfinite(spread) or spread <= 0:
        spread = float(np.ptp(xs)) if xs.size else grid_dx
        if not np.isfinite(spread) or spread <= 0:
            spread = grid_dx

    # Stronger smoothing demands a deeper valley to keep multi-peak structure.
    ratio = min(3.0, bandwidth / max(spread, 1e-9))
    drop_required = np.clip(min_valley_drop + 0.2 * ratio, min_valley_drop, 0.8)

    keep: list[int] = [int(p_idx[0])]
    for idx in map(int, p_idx[1:]):
        prev = keep[-1]
        sep = float(xs[idx] - xs[prev])
        if sep <= collapse_sep:
            lo = min(prev, idx)
            hi = max(prev, idx)
            valley = float(np.min(ys[lo : hi + 1]))
            base = float(min(ys[prev], ys[idx]))
            if base <= 0:
                base = float(max(ys[prev], ys[idx], 1e-12))
            if base <= 0 or valley >= base * (1.0 - drop_required):
                if ys[idx] > ys[prev]:
                    keep[-1] = idx
                continue
        keep.append(idx)

    return np.asarray(sorted(set(keep)), dtype=int)


def _filter_prominence(ys: np.ndarray, p_idx: np.ndarray, ratio: float = 0.25) -> np.ndarray:
    """Discard peaks with tiny prominence relative to the strongest mode."""

    if p_idx.size <= 1:
        return p_idx

    prominences, _, _ = peak_prominences(ys, p_idx)
    if prominences.size == 0:
        return p_idx

    best = float(np.max(prominences))
    if not np.isfinite(best) or best <= 0:
        return p_idx

    keep = [int(idx) for idx, prom in zip(p_idx, prominences) if prom >= best * ratio]
    if not keep:
        keep = [int(p_idx[int(np.argmax(prominences))])]

    return np.asarray(sorted(set(keep)), dtype=int)

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
    x: np.ndarray,
    xs: np.ndarray,
    kde: gaussian_kde,
) -> np.ndarray:
    """Evaluate KDE, switching to FFT when the grid is large."""

    n = x.size
    if n == 0:
        return np.zeros_like(xs)

    # ``gaussian_kde`` evaluation cost grows with ``n × len(xs)``.
    brute_cost = n * xs.size

    # ``kde.factor`` already incorporates any ``set_bandwidth`` calls.
    sample_std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    bandwidth = kde.factor * sample_std

    if brute_cost <= 5_000_000 or bandwidth <= 0:
        return kde(xs)

    return _fft_gaussian_kde(x, xs, bandwidth)


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
def kde_peaks_valleys(
    data           : np.ndarray,
    n_peaks        : int  | None = None,
    prominence     : float        = 0.05,
    bw             : str  | float = "scott",
    min_width      : int  | None  = None,
    grid_size      : int          = 20_000,
    drop_frac      : float        = 0.10,
    min_x_sep      : float        = 0.5,   # absolute, same units as `data`
    min_valley_drop: float        = 0.15,
    curvature_thresh: float | None = None,
    turning_peak   : bool = False,
    first_valley   : str = "slope",
):
    x = np.asarray(data, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return [], [], np.array([]), np.array([])

    q25, q75 = np.percentile(x, [25, 75])
    data_iqr = float(q75 - q25)
    if not np.isfinite(data_iqr) or data_iqr <= 0:
        data_iqr = float(np.ptp(x)) if x.size else 0.0

    # ---------- KDE grid ----------
    if x.size > 10_000:
        x = np.random.choice(x, 10_000, replace=False)
    kde = gaussian_kde(x, bw_method=bw)
    if _mostly_small_discrete(x):
        kde.set_bandwidth(kde.factor * 4.0)
    h   = kde.factor * x.std(ddof=1)
    xs  = np.linspace(x.min() - h, x.max() + h,
                      min(grid_size, max(4000, 4 * x.size)))
    ys  = _evaluate_kde(x, xs, kde)
    bandwidth_eff = float(kde.factor * x.std(ddof=1)) if x.size > 1 else 0.0
    if not np.isfinite(bandwidth_eff) or bandwidth_eff <= 0:
        bandwidth_eff = float(abs(xs[1] - xs[0])) if xs.size > 1 else 0.0

    # ---------- primary peaks ----------

    grid_dx  = xs[1] - xs[0]
    distance = None                       # (NEW)  hard x-spacing
    if min_x_sep is not None:
        # ``scipy.signal.find_peaks`` requires ``distance`` >= 1.  For very
        # small ``min_x_sep`` relative to the grid spacing the integer
        # division above could yield ``0`` which would raise a ``ValueError``.
        # Clamp the value so that ``find_peaks`` always receives a valid
        # minimum distance.
        distance = max(1, int(min_x_sep / grid_dx))
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

    locs_initial = np.array(locs, copy=True)
    locs = _merge_close_peaks(
        xs,
        ys,
        locs,
        min_x_sep=min_x_sep,
        min_valley_drop=min_valley_drop,
    )
    if n_peaks is None:
        locs = _collapse_with_bandwidth(
            xs,
            ys,
            locs,
            bandwidth_eff,
            min_x_sep,
            min_valley_drop,
            data_iqr,
        )
        locs = _filter_prominence(ys, locs)
        if locs.size == 0 and locs_initial.size:
            locs = np.asarray([int(locs_initial[np.argmax(ys[locs_initial])])])

    # relax prominence if too few
    if n_peaks is not None and locs.size < n_peaks:
        prom = prominence
        for _ in range(4):
            prom /= 2
            locs, _ = find_peaks(ys, prominence=prom,
                                 **({"width": min_width} if min_width else {}))
            if locs.size >= n_peaks:
                break

    if locs.size == 0 and n_peaks is None:
        return [], [], xs, ys

    # keep tallest as before
    if n_peaks is None or locs.size >= n_peaks:
        peaks_idx = np.sort(locs) if n_peaks is None else \
                    np.sort(locs[np.argsort(ys[locs])[-n_peaks:]])
    else:
        peaks_idx = np.sort(locs)
    
    peaks_x = xs[peaks_idx].tolist()

    if (min_x_sep is not None) and (len(peaks_x) > 1):   # 2 safe now
        peaks_x.sort()
        keep = [peaks_x[0]]
        for px in peaks_x[1:]:
            if px - keep[-1] >= min_x_sep:
                keep.append(px)
        peaks_x = keep

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

        peaks_x = sorted(peaks_x)[:n_peaks]

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
    valleys_x = _enforce_valley_rule(peaks_x, valleys_x)
    return np.round(peaks_x, 10).tolist(), valleys_x, xs, ys

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
