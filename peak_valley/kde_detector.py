from __future__ import annotations
import numpy as np
from scipy.stats  import gaussian_kde
from scipy.signal import find_peaks

__all__ = ["kde_peaks_valleys", "quick_peak_estimate"]


# ----------------------------------------------------------------------
# peak_valley/kde_detector.py
import numpy as np

# peak_valley/kde_detector.py  (only the helper changed)
# ----------------------------------------------------------------------
def _merge_close_peaks(xs: np.ndarray,
                       ys: np.ndarray,
                       p_idx: np.ndarray,
                       min_x_sep: float = 0.4,
                       min_valley_drop: float = 0.15) -> np.ndarray:
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


def _first_valley_slope(xs: np.ndarray,
                        ys: np.ndarray,
                        p_left: int,
                        p_right: int | None = None) -> float | None:
    """Pick first valley via maximum increase in slope.

    Parameters
    ----------
    xs, ys : np.ndarray
        Grid and KDE values.
    p_left : int
        Index of the first peak.
    p_right : int | None, optional
        Index of the next peak.  If ``None`` the search extends to the
        end of the grid.

    Returns
    -------
    float | None
        X‑coordinate of the valley or ``None`` if the slope never
        increases.
    """
    d2y = np.diff(ys, n=2)
    if p_right is None:
        seg = d2y[p_left:]
        base = p_left
    else:
        if p_right - p_left <= 2:
            return None
        seg = d2y[p_left : p_right - 1]
        base = p_left
    if seg.size == 0:
        return None
    rel = np.argmax(seg)
    if seg[rel] <= 0:
        return None
    idx = base + 1 + rel
    return float(xs[idx])


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
    min_x_sep      : float        = 1.0,   # absolute, same units as `data`
    min_valley_drop: float        = 0.15,
    curvature_thresh: float | None = None,
    turning_peak   : bool = False,
    first_valley   : str = "slope",
):
    x = np.asarray(data, float)
    if x.size == 0:
        return [], [], np.array([]), np.array([])

    # ---------- KDE grid ----------
    if x.size > 10_000:
        x = np.random.choice(x, 10_000, replace=False)
    kde = gaussian_kde(x, bw_method=bw)
    if _mostly_small_discrete(x):
        kde.set_bandwidth(kde.factor * 4.0)
    h   = kde.factor * x.std(ddof=1)
    xs  = np.linspace(x.min() - h, x.max() + h,
                      min(grid_size, max(4000, 4 * x.size)))
    ys  = kde(xs)

    # ---------- primary peaks ----------

    grid_dx  = xs[1] - xs[0]
    distance = None                       # (NEW)  hard x-spacing
    if min_x_sep is not None:
        distance = int(min_x_sep / grid_dx)
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

    if (min_x_sep is not None) and (len(peaks_x) > 1):   # 2️⃣ safe now
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
            val = _first_valley_slope(xs, ys, p0, p1)
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
            val = _first_valley_slope(xs, ys, p0)
        if val is not None:
            valleys_x.append(val)

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
