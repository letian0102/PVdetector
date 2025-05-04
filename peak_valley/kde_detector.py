from __future__ import annotations
import numpy as np
from scipy.stats  import gaussian_kde
from scipy.signal import find_peaks

__all__ = ["kde_peaks_valleys", "quick_peak_estimate"]


# ------------------------------------------------------------------
def kde_peaks_valleys(
    data: np.ndarray,
    n_peaks: int | None = None,
    prominence: float   = 0.05,
    bw: str | float     = "scott",
    min_width: int | None = None,
    grid_size: int      = 20_000,
):
    """
    Core detector: returns (peaks, valleys, xs, ys)
    * identical math to the latest monolithic version.
    """
    x = np.asarray(data, float)
    if x.size == 0:
        return [], [], np.array([]), np.array([])

    if x.size > 10_000:
        x = np.random.choice(x, 10_000, replace=False)

    kde = gaussian_kde(x, bw_method=bw)
    h   = kde.factor * x.std(ddof=1)
    xs  = np.linspace(x.min() - h, x.max() + h,
                      min(grid_size, max(4000, 4 * x.size)))
    ys  = kde(xs)

    kw = {"prominence": prominence}
    if min_width:
        kw["width"] = min_width
    locs, _ = find_peaks(ys, **kw)

    # relax prominence … (unchanged)
    if n_peaks is not None and locs.size < n_peaks:
        prom = prominence
        for _ in range(4):
            prom /= 2
            locs, _ = find_peaks(
                ys, prominence=prom,
                **({"width": min_width} if min_width else {}),
            )
            if locs.size >= n_peaks:
                break

    if locs.size == 0:
        return [], [], xs, ys

    if n_peaks is None or n_peaks >= locs.size:
        peaks_idx = np.sort(locs)
    else:
        tallest   = np.argsort(ys[locs])[-n_peaks:]
        peaks_idx = np.sort(locs[tallest])

    peaks_x = np.round(xs[peaks_idx], 10).tolist()

    valleys_x = []
    if len(peaks_idx) > 1:
        for L, R in zip(peaks_idx[:-1], peaks_idx[1:]):
            seg   = np.arange(L + 1, R)
            mins, _ = find_peaks(-ys[seg])
            idx   = seg[mins[np.argmin(ys[seg][mins])]] if mins.size \
                    else seg[np.argmin(ys[seg])]
            valleys_x.append(np.round(xs[idx], 10))

    return peaks_x, valleys_x, xs, ys


# ------------------------------------------------------------------
def quick_peak_estimate(
    counts: np.ndarray,
    prominence: float,
    bw: str | float,
    min_width: int | None,
    grid_size: int,
) -> tuple[int, bool]:
    """
    Fast, cheap peak counter used before calling GPT.
    Returns (n_peaks, confident?).
    We call the detector twice with 2× diff prominence — if the number
    stays the same we treat it as 'confident'.
    """
    p1, *_ = kde_peaks_valleys(counts, None, prominence,
                               bw, min_width, grid_size)
    p2, *_ = kde_peaks_valleys(counts, None, prominence / 2,
                               bw, min_width, grid_size)
    return len(p1), (len(p1) == len(p2) and len(p1) > 0)
