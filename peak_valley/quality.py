# ---------- stain_quality.py ------------------------------------------
import numpy as np
from scipy.stats import gaussian_kde

def _fallback_valley(peaks):
    """
    Very coarse guess for a missing first valley so that a score
    can still be produced.

    * 2 peaks  -> midpoint between the two
    * ≥3 peaks -> midpoint between the first two (matches the R code that
                  uses the *first* valley for k<=2, the *last* otherwise;
                  our scorer only needs the first one)
    """
    if len(peaks) < 2:
        return None
    return 0.5 * (peaks[0] + peaks[1])

def _auc_right(dens_x, dens_y, thresh):
    """area under the density curve **to the right** of `thresh`"""
    idx = np.where(dens_x >= thresh)[0]
    if idx.size < 2:
        return 0.0
    # trapezoidal rule
    return np.trapz(dens_y[idx], dens_x[idx])

def _deep_scaler(x, peaks, valleys, peak_num):
    """depth of the first valley × AUC right of that valley"""
    # KDE on *all* counts (faster than per-sample adapt.)
    kde = gaussian_kde(x, bw_method="scott")
    xs  = np.linspace(x.min(), x.max(), 4000)
    ys  = kde(xs)

    if peak_num > 1 and valleys:
        valley = valleys[0]                 # first valley
        y_val  = ys[np.argmin(np.abs(xs - valley))]
        right_peak = peaks[-1]
        y_pk   = ys[np.argmin(np.abs(xs - right_peak))]
    else:                                   # single-peak staining
        valley, y_val, y_pk = peaks[0], 0.0, 0.0

    auc   = _auc_right(xs, ys, valley)
    if peak_num == 1:
        auc = min(auc, 1 - auc)

    return (1 + y_pk - y_val) * (1 + auc)

def stain_quality(counts, peaks, valleys):
    """
    Counts = 1-D numpy array (already arcsinh-transformed)  
    peaks / valleys = lists of floats

    Returns a single quality score *or* np.nan.
    """
    counts = np.asarray(counts, float)
    if counts.size > 10_000:
        # ``np.random.choice`` without replacement partially shuffles the entire
        # array.  For extremely large inputs this shuffle can dominate runtime
        # for a single sample, so mirror the fast sampling guard used by the
        # KDE/GMM helpers to keep quality scoring responsive in both the CLI and
        # Streamlit flows.
        rng = np.random.default_rng()
        if counts.size > 200_000:
            counts = rng.choice(counts, 10_000, replace=True)
        else:
            counts = rng.choice(counts, 10_000, replace=False)
    peaks  = [p for p in peaks  if np.isfinite(p)]
    valleys= [v for v in valleys if np.isfinite(v)]

    # ⇢ fabricate a valley if the detector didn't return one
    if len(valleys) == 0 and len(peaks) >= 2:
        fv = _fallback_valley(peaks)
        if fv is not None:
            valleys = [fv]

    if counts.size == 0 or len(valleys) == 0:
        return np.nan                         # still impossible → NaN

    k = len(peaks)
    valley0 = valleys[0]

    if k == 1:                               # one-peak formula
        mean_diff = abs(valley0 - peaks[0])
        within_sd = np.sqrt(np.var(counts, ddof=1))
        deep_scal = _deep_scaler(counts, peaks, valleys, 1)
        return (mean_diff * deep_scal) / within_sd

    elif k == 2:                             # two-peak formula
        mean_diff = abs(peaks[-1] - peaks[0])
        left  = counts[counts <  valley0]
        right = counts[counts >= valley0]
        within_var = ((left - peaks[0])**2).sum() + ((right - peaks[-1])**2).sum()
        within_sd  = np.sqrt(within_var / counts.size)
        deep_scal  = _deep_scaler(counts, peaks, valleys, 2)
        return (mean_diff * deep_scal) / within_sd

    else:                                    # ≥3 peaks
        mean_diff = abs(peaks[-1] - peaks[0])
        # ensure we have enough valleys to separate peak segments
        required = len(peaks) - 1
        valleys_sorted = sorted(valleys)
        while len(valleys_sorted) < required:
            idx = len(valleys_sorted)
            next_idx = min(idx, len(peaks) - 2)
            midpoint = 0.5 * (peaks[next_idx] + peaks[next_idx + 1])
            valleys_sorted.append(midpoint)
        valleys_sorted = valleys_sorted[:required]

        # assign every point to the nearest peak segment
        seg_var = 0.0
        borders = [float(counts.min())] + valleys_sorted + [float(counts.max())]
        for i, pk in enumerate(peaks):
            upper = borders[i + 1]
            if i == len(peaks) - 1:
                in_seg = counts >= borders[i]
            else:
                in_seg = (counts >= borders[i]) & (counts < upper)
            if not np.any(in_seg):
                continue
            seg_var += ((counts[in_seg] - pk)**2).sum()
        within_sd = np.sqrt(seg_var / counts.size)
        deep_scal = _deep_scaler(counts, peaks, valleys_sorted, k)
        return (mean_diff * deep_scal) / within_sd
