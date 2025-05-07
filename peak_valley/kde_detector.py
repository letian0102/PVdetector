from __future__ import annotations
import numpy as np
from scipy.stats  import gaussian_kde
from scipy.signal import find_peaks

__all__ = ["kde_peaks_valleys", "quick_peak_estimate"]


# ----------------------------------------------------------------------
def _first_drop_valley(xs: np.ndarray,
                       ys: np.ndarray,
                       p_left: int,
                       p_right: int,
                       drop_frac: float = 0.10) -> float:
    """
    Return x-coord of the *first* minimum between two peaks once the KDE
    has fallen below (drop_frac × height_of_left_peak).

    Parameters
    ----------
    xs, ys     : full KDE grid (1-D)
    p_left     : index of the left peak in xs / ys
    p_right    : index of the right peak in xs / ys
    drop_frac  : e.g. 0.10  →   10 % of left-peak height
    """
    y_peak = ys[p_left]
    seg    = np.arange(p_left + 1, p_right)

    # 1) find first point where KDE < drop threshold
    under  = seg[ys[seg] < drop_frac * y_peak]
    if under.size == 0:                         # never dropped far enough
        idx_val = seg[np.argmin(ys[seg])]
        return float(xs[idx_val])

    start  = under[0]

    # 2) from that point on, march forward while still decreasing
    cur    = start
    while cur + 1 <= seg[-1] and ys[cur + 1] <= ys[cur]:
        cur += 1

    return float(xs[cur])


# ----------------------------------------------------------------------
def kde_peaks_valleys(
    data: np.ndarray,
    n_peaks: int | None = None,
    prominence: float   = 0.05,
    bw: str | float     = "scott",
    min_width: int | None = None,
    grid_size: int      = 20_000,
    drop_frac: float    = 0.10,          # <- NEW: valley threshold (10 %)
):
    """
    Peaks & valleys finder.

    Valleys are chosen as the *first* dip after the left peak where the
    density has dropped below `drop_frac × height_of_left_peak` and is
    no longer decreasing – exactly the visual “gutter” you described.
    """
    x = np.asarray(data, float)
    if x.size == 0:
        return [], [], np.array([]), np.array([])

    # thin / KDE grid – unchanged
    if x.size > 10_000:
        x = np.random.choice(x, 10_000, replace=False)

    kde = gaussian_kde(x, bw_method=bw)
    h   = kde.factor * x.std(ddof=1)
    xs  = np.linspace(x.min() - h, x.max() + h,
                      min(grid_size, max(4000, 4 * x.size)))
    ys  = kde(xs)

    # find peaks (unchanged)
    kw = {"prominence": prominence}
    if min_width:
        kw["width"] = min_width
    locs, _ = find_peaks(ys, **kw)

    # relax prominence if requested number not found
    if n_peaks is not None and locs.size < n_peaks:
        prom = prominence
        for _ in range(4):
            prom /= 2
            locs, _ = find_peaks(
                ys, prominence=prom,
                **({"width": min_width} if min_width else {}))
            if locs.size >= n_peaks:
                break

    if locs.size == 0:
        return [], [], xs, ys

    # keep tallest n_peaks if user limited them
    if n_peaks is None or n_peaks >= locs.size:
        peaks_idx = np.sort(locs)
    else:
        tallest   = np.argsort(ys[locs])[-n_peaks:]
        peaks_idx = np.sort(locs[tallest])

    peaks_x = np.round(xs[peaks_idx], 10).tolist()

    # ----------- valleys using the new rule -----------------------------
    valleys_x: list[float] = []
    if len(peaks_idx) > 1:
        for left, right in zip(peaks_idx[:-1], peaks_idx[1:]):
            valleys_x.append(
                _first_drop_valley(xs, ys, left, right, drop_frac)
            )

    return peaks_x, valleys_x, xs, ys


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
