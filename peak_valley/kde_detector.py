from __future__ import annotations
import numpy as np
from scipy.stats  import gaussian_kde
from scipy.signal import find_peaks

__all__ = ["kde_peaks_valleys", "quick_peak_estimate"]


# ----------------------------------------------------------------------
# peak_valley/kde_detector.py
import numpy as np
from scipy.signal import argrelextrema

# peak_valley/kde_detector.py  (only the helper changed)
# ----------------------------------------------------------------------
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
    data: np.ndarray,
    n_peaks: int | None = None,
    prominence: float   = 0.05,
    bw: str | float     = "scott",
    min_width: int | None = None,
    grid_size: int      = 20_000,
    drop_frac: float    = 0.10,
):
    x = np.asarray(data, float)
    if x.size == 0:
        return [], [], np.array([]), np.array([])

    # ---------- KDE grid ----------
    if x.size > 10_000:
        x = np.random.choice(x, 10_000, replace=False)
    kde = gaussian_kde(x, bw_method=bw)
    h   = kde.factor * x.std(ddof=1)
    xs  = np.linspace(x.min() - h, x.max() + h,
                      min(grid_size, max(4000, 4 * x.size)))
    ys  = kde(xs)

    # ---------- primary peaks ----------
    kw = {"prominence": prominence}
    if min_width:
        kw["width"] = min_width
    locs, _ = find_peaks(ys, **kw)

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

    # ---------- valleys: ALWAYS between every consecutive pair ----------
    valleys_x: list[float] = []
    if len(peaks_idx) > 1:
        for left, right in zip(peaks_idx[:-1], peaks_idx[1:]):
            valleys_x.append(
                _valley_between(xs, ys, left, right, drop_frac)
            )

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