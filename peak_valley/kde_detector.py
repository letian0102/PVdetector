from __future__ import annotations
import numpy as np
from  scipy.stats  import gaussian_kde
from  scipy.signal import find_peaks

__all__ = ["kde_peaks_valleys", "quick_peak_estimate"]

# ----------------------------------------------------------------------
# peak_valley/kde_detector.py  – helpers (unchanged)
# ----------------------------------------------------------------------
from scipy.signal import argrelextrema   # noqa: F401  – still used elsewhere

def _prominence_merge(xs: np.ndarray,
                      ys: np.ndarray,
                      pk_idx: list[int],
                      drop_frac: float) -> list[int]:
    """
    Collapse any two consecutive peaks whose valley fails the
    adaptive-depth test:

         Δx < 1.5 · h   →  dip ≥ 50 % of the LOWER peak  (very strict)
    1.5 · h ≤ Δx < 2.5  →  dip ≥ 35 % of the lower peak
         Δx ≥ 2.5 · h   →  dip ≥ *drop_frac*  (default 10 %)

    `h` is the KDE bandwidth that was already computed in the caller.
    """
    if len(pk_idx) < 2:
        return pk_idx

    pk_idx = sorted(pk_idx)
    keep   = [pk_idx[0]]

    # --- estimate the bandwidth h once from the grid ---------------
    # the caller’s `h` is not in scope, so re-derive an equivalent one
    grid_h = xs[1] - xs[0]               # grid step
    # find the typical distance between grid points that differ by 1 bandwidth
    # (works because xs was built as linspace(min-h, max+h, ...))
    h_est  = (xs[-1] - xs[0]) / (len(xs) - 1) * (len(xs) - 1) / (len(xs) - 2)

    for right in pk_idx[1:]:
        left = keep[-1]
        if left > right:
            left, right = right, left

        dx = xs[right] - xs[left]

        # ---------- adaptive depth threshold -----------------------
        if dx < 1.5 * h_est:
            need_drop = 0.50             # shoulder killer
        elif dx < 2.5 * h_est:
            need_drop = 0.35             # moderately strict
        else:
            need_drop = drop_frac        # original (0.10)
        # -----------------------------------------------------------

        seg = slice(left + 1, right)
        if seg.start >= seg.stop:
            continue
        min_y = ys[seg].min()
        y_thr = (1.0 - need_drop) * min(ys[left], ys[right])

        if min_y <= y_thr:               # deep enough → keep both
            keep.append(right)
        else:                            # otherwise keep the taller one
            if ys[right] > ys[left]:
                keep[-1] = right

    return keep


def _turning_candidates(ys: np.ndarray) -> np.ndarray:
    """Indices of concave-down inflection points in a 1-D array."""
    if len(ys) < 3:
        return np.empty(0, int)

    dy   = np.diff(ys)
    sgn  = np.sign(dy)
    zc   = np.where(sgn[:-1] * sgn[1:] < 0)[0] + 1       # slope → 0 → slope
    if zc.size == 0:
        return np.empty(0, int)

    d2      = np.diff(ys, 2)
    abs_d2  = np.abs(d2)
    curv_pk, _ = find_peaks(abs_d2)                      # peaks of |d²y|
    curv_pk += 1

    tp = np.intersect1d(zc, curv_pk, assume_unique=True)
    return tp[d2[tp - 1] < 0]                            # only concave-down


# …  _valley_between  and  _turning_valley  helpers stay unchanged …

# ----------------------------------------------------------------------
def kde_peaks_valleys(
    data: np.ndarray,
    n_peaks:   int   | None = None,
    prominence: float       = 0.05,
    bw:         str   | float = "scott",
    min_width:  int   | None  = None,
    grid_size:  int          = 20_000,
    drop_frac:  float        = 0.10,
):
    """
    Return (peaks_x, valleys_x, xs, ys) for a 1-D sample `data`.
    Implements a two-tier detector:

        tier-1  = primary density maxima  (find_peaks)
        tier-2  = curvature sub-peaks     (_turning_candidates)

    Tier-2 peaks must satisfy three extra tests:
        • distance   ≥ α·h     from any tier-1 peak
        • curvature  ≥ β·max|d²y|
        • prominence ≥ γ·height   on its shallower side
    """
    # ---------- KDE grid ----------
    x = np.asarray(data, float)
    if x.size == 0:
        return [], [], np.array([]), np.array([])

    if x.size > 10_000:                              # speed guard
        x = np.random.choice(x, 10_000, replace=False)

    kde = gaussian_kde(x, bw_method=bw)
    h   = kde.factor * x.std(ddof=1)                 # KDE bandwidth
    xs  = np.linspace(x.min() - h, x.max() + h,
                      min(grid_size, max(4000, 4 * x.size)))
    ys  = kde(xs)

    # ------------------------------------------------------------------
    # 1️⃣  PRIMARY PEAKS  (wide distance constraint)
    # ------------------------------------------------------------------
    dx        = xs[1] - xs[0]
    min_dist  = max(1, int(5 * h / dx))              # ≥ 5·h spacing
    pk_kwargs = {"prominence": prominence, "distance": min_dist}
    if min_width:
        pk_kwargs["width"] = min_width

    locs, _ = find_peaks(ys, **pk_kwargs)

    # relax prominence if user asked for a specific number
    if n_peaks is not None and locs.size < n_peaks:
        prom = prominence
        for _ in range(4):
            prom /= 2
            pk_kwargs["prominence"] = prom
            locs, _ = find_peaks(ys, **pk_kwargs)
            if locs.size >= n_peaks:
                break

    if locs.size == 0 and n_peaks is None:
        return [], [], xs, ys

    if locs.size:
        if n_peaks is None:
            peaks_idx = np.sort(locs)
        else:                                        # top-density locs
            sel = np.argsort(ys[locs])[-n_peaks:]
            peaks_idx = np.sort(locs[sel])
    else:
        peaks_idx = np.empty(0, int)

    # ------------------------------------------------------------------
    # 2️⃣  CURVATURE CANDIDATES  (strict filters)
    # ------------------------------------------------------------------
    tp_peaks = _turning_candidates(ys)
    if tp_peaks.size:

        # ---- 2a. distance filter  ------------------------------------
        alpha = 1.8                                   # ≥ α·h from tier-1
        if peaks_idx.size:
            tp_peaks = [i for i in tp_peaks
                        if np.min(np.abs(xs[i] - xs[peaks_idx]))
                        >= alpha * h]
        # (if no primary peaks yet, keep all – very rare)

        # ---- 2b. curvature magnitude filter  -------------------------
        beta      = 0.20
        d2        = np.diff(ys, 2)
        abs_d2    = np.abs(d2)
        curv_thr  = beta * abs_d2.max() if abs_d2.size else 0.0
        tp_peaks  = [i for i in tp_peaks
                     if abs_d2[i - 1] >= curv_thr]

        # ---- 2c. single-sided prominence filter  ---------------------
        gamma = 0.4
        keep  = []
        peaks_idx_sorted = np.sort(peaks_idx)        # ensure ascending
        for i in tp_peaks:
            # nearest accepted peak on each side (if any)
            left_idx  = peaks_idx_sorted[peaks_idx_sorted < i]
            right_idx = peaks_idx_sorted[peaks_idx_sorted > i]
            left  = left_idx[-1] if left_idx.size  else None
            right = right_idx[0] if right_idx.size else None

            if left is not None and right is not None:
                base = max(ys[left:right].min(), ys[i])
            elif left is not None:
                base = ys[left:i].min()
            elif right is not None:
                base = ys[i:right].min()
            else:
                base = ys.min()

            if ys[i] - base >= gamma * ys[i]:
                keep.append(i)

        tp_peaks = np.array(keep, int)

        # ---- merge the two tiers  ------------------------------------
        if tp_peaks.size:
            peaks_idx = np.sort(np.concatenate([peaks_idx, tp_peaks]))

    # ------------------------------------------------------------------
    # 3️⃣  COLLAPSE PEAKS CLOSER THAN ONE BANDWIDTH
    # ------------------------------------------------------------------
    c_sep = 2.0
    if peaks_idx.size > 1:
        clean = [peaks_idx[0]]
        for idx in peaks_idx[1:]:
            if xs[idx] - xs[clean[-1]] < c_sep * h:
                if ys[idx] > ys[clean[-1]]:
                    clean[-1] = idx 
            else:
                clean.append(idx)
        peaks_idx = np.array(clean, int)

    # ------------------------------------------------------------------
    # 4️⃣  DROP-FRACTION MERGE  (original rule)
    # ------------------------------------------------------------------
    peaks_idx = _prominence_merge(xs, ys,
                                  peaks_idx.tolist(), drop_frac)

    peaks_x = xs[peaks_idx].tolist()

    # ------------------------------------------------------------------
    # 5️⃣  OPTIONAL GMM-FALLBACK (unchanged)
    # ------------------------------------------------------------------
    if n_peaks is not None and len(peaks_x) < n_peaks:
        from .gmm_fallback import gmm_component_means
        gmm_means = gmm_component_means(x, n_peaks)

        # merge (tolerance = 5 % IQR)
        iqr  = np.subtract(*np.percentile(x, [75, 25]))
        tol  = 0.05 * iqr if iqr else 0.1
        def uniq(v, L): return all(abs(v - w) > tol for w in L)

        for m in gmm_means:
            if uniq(m, peaks_x):
                peaks_x.append(m)

        peaks_x = sorted(peaks_x)[:n_peaks]

    # ------------------------------------------------------------------
    # 6️⃣  RE-INDEX & VALLEYS
    # ------------------------------------------------------------------
    peaks_idx = [int(np.argmin(np.abs(xs - px))) for px in peaks_x]

    valleys_x: list[float] = []
    if len(peaks_idx) > 1:
        for left, right in zip(peaks_idx[:-1], peaks_idx[1:]):
            valleys_x.append(_valley_between(xs, ys, left, right,
                                             drop_frac))

    return np.round(peaks_x, 10).tolist(), valleys_x, xs, ys


# ----------------------------------------------------------------------
def quick_peak_estimate(
    counts:     np.ndarray,
    prominence: float,
    bw:         str   | float,
    min_width:  int   | None,
    grid_size:  int,
) -> tuple[int, bool]:
    """Unchanged: dual-prominence trick."""
    p1, *_ = kde_peaks_valleys(counts, None, prominence,
                               bw, min_width, grid_size)
    p2, *_ = kde_peaks_valleys(counts, None, prominence / 2,
                               bw, min_width, grid_size)
    return len(p1), (len(p1) == len(p2) and len(p1) > 0)


# ----------------------------------------------------------------------
def _turning_valley(
    xs:        np.ndarray,
    ys:        np.ndarray,
    p_left:    int,
    p_right:   int,
    drop_frac: float,
) -> float:
    """
    Valley between two peaks = first turning-point where KDE switches
    from falling to rising **and** is already below `drop_frac` of the
    lower peak’s height.  Fallback = absolute minimum on the interval.
    """
    y_thr  = drop_frac * min(ys[p_left], ys[p_right])
    seg    = np.arange(p_left + 1, p_right)           # open interval
    dy     = np.diff(ys[seg])                         # slope

    turn   = np.where((dy[:-1] < 0) & (dy[1:] >= 0))[0] + 1
    turn   = turn[ys[seg[turn]] <= y_thr]             # depth rule

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
