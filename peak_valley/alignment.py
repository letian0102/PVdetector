# peak_valley/alignment.py
# --------------------------------------------------------------
"""
Light-weight landmark alignment for 1-D ADT distributions.

The implementation is an intentionally simple, **piece-wise linear,
monotone warping** that mimics the behaviour of the original R/fda
`landmarkreg()` call but without external heavy dependencies.

•  If three landmarks are available
      (negPeak, valley, posPeak)  → 3-segment warp  
•  If two landmarks are available (negPeak, valley) → 2-segment warp  
•  If only one landmark            (negPeak)        → pure shift

The target landmark set defaults to the **column means** across samples
(just like the R code) but may be overridden.
"""
from __future__ import annotations
import numpy as np
from typing import Sequence, Tuple, List, Optional, Union

DensityGrid = Tuple[Sequence[float], Sequence[float]]


def _identity_warp(x):
    """Return a numpy array copy of *x* (used when no warping is needed)."""
    return np.asarray(x, float)

__all__ = ["fill_landmark_matrix",
           "build_warp_function",
           "align_distributions"]


# ------------------------------------------------------------------
def _median_or_nan(values: list[float]) -> float:
    valid = np.asarray(values, float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def _reference_peak_stats(peaks: List[Sequence[float]]) -> tuple[float, float, float]:
    """Return cohort medians for negative/positive peaks and their gap."""

    negs = [pk[0] for pk in peaks if pk]
    poss = [pk[-1] for pk in peaks if pk and len(pk) > 1]

    neg_med = _median_or_nan(negs)
    pos_med = _median_or_nan(poss)
    gap = pos_med - neg_med if np.isfinite(neg_med) and np.isfinite(pos_med) else float("nan")
    return neg_med, pos_med, gap


def _find_histogram_peaks(bin_centers: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """Lightweight peak finder for 1-D histograms (no SciPy dependency)."""

    if hist.size < 3:
        return np.array([], dtype=int)

    left = hist[:-2]
    mid = hist[1:-1]
    right = hist[2:]
    mask = (mid >= left) & (mid >= right)
    return np.where(mask)[0] + 1


def _infer_negative_peak_from_counts(
    counts: Sequence[float],
    detected_positive: float,
    typical_gap: float,
) -> tuple[Optional[float], Optional[float]]:
    """Infer a missing negative peak and optional valley using histogram heuristics."""

    data = np.asarray(counts, float)
    data = data[np.isfinite(data)]
    if data.size < 10:
        return None, None

    left_data = data[data < detected_positive]
    if left_data.size < 5:
        return None, None

    hist, edges = np.histogram(left_data, bins=128)
    centers = 0.5 * (edges[:-1] + edges[1:])

    candidates = _find_histogram_peaks(centers, hist)
    candidates = candidates[centers[candidates] < detected_positive]

    neg_peak: Optional[float] = None
    if candidates.size:
        tallest = candidates[int(np.argmax(hist[candidates]))]
        neg_peak = float(centers[tallest])
    elif np.isfinite(typical_gap) and typical_gap > 0:
        neg_peak = float(detected_positive - typical_gap)

    valley: Optional[float] = None
    if neg_peak is not None:
        between = (centers > neg_peak) & (centers < detected_positive)
        if np.any(between):
            seg_centers = centers[between]
            seg_hist = hist[between]
            valley = float(seg_centers[int(np.argmin(seg_hist))])

    return neg_peak, valley


def _augment_single_peak_samples(
    peaks: List[Sequence[float]],
    valleys: List[Sequence[float]],
    counts: List[np.ndarray],
) -> tuple[List[Sequence[float]], List[Sequence[float]]]:
    """
    Detect single-peak samples whose lone peak matches the cohort's positive
    peak and try to recover the missing negative peak from the raw counts.
    """

    peaks_adj = [list(pk) if pk else [] for pk in peaks]
    valleys_adj = [list(vl) if vl else [] for vl in valleys]

    neg_med, pos_med, gap_med = _reference_peak_stats(peaks_adj)
    if not (np.isfinite(neg_med) and np.isfinite(pos_med)):
        return peaks_adj, valleys_adj

    margin = 0.1 * gap_med if np.isfinite(gap_med) and gap_med > 0 else 0.1

    for i, pk in enumerate(peaks_adj):
        if len(pk) != 1:
            continue

        current = float(pk[0])
        if not np.isfinite(current):
            continue

        dist_neg = abs(current - neg_med)
        dist_pos = abs(current - pos_med)
        if dist_pos >= dist_neg - margin:
            continue

        neg_guess, valley_guess = _infer_negative_peak_from_counts(
            counts[i], current, gap_med
        )

        if neg_guess is None or neg_guess >= current:
            continue

        pk.insert(0, neg_guess)
        peaks_adj[i] = pk

        if valley_guess is not None:
            valleys_adj[i] = [valley_guess]

    return peaks_adj, valleys_adj


# ------------------------------------------------------------------
def fill_landmark_matrix(
    peaks   : List[Sequence[float]],
    valleys : List[Sequence[float]],
    align_type     : str = "negPeak_valley_posPeak",
    midpoint_type  : str = "valley",
    neg_thr        : float | None = None,
) -> np.ndarray:
    """
    Re-implement (in Python) the `landmark_fill_na()` logic from R.
    Only the modes that matter for our UI are ported.
    """
    if align_type not in {
        "negPeak", "negPeak_valley", "negPeak_valley_posPeak", "valley"
    }:
        raise ValueError(f"Unknown align_type '{align_type}'")

    # ----- convert ragged → dense ------------------------------------------------
    n = len(peaks)
    pk_lengths = [len(pk) for pk in peaks]
    vl_lengths = [len(vl) for vl in valleys]

    max_pk = max(pk_lengths, default=0)
    max_vl = max(vl_lengths, default=0)

    has_val_orig = max_vl > 0
    has_pos_orig = any(length > 1 for length in pk_lengths)

    pk_mat = np.full((n, max_pk if max_pk > 0 else 1), np.nan)
    vl_mat = np.full((n, max_vl if max_vl > 0 else 1), np.nan)

    for i, pk in enumerate(peaks):
        if pk:
            pk_mat[i, : len(pk)] = pk
    for i, vl in enumerate(valleys):
        if vl:
            vl_mat[i, : len(vl)] = vl

    # Negative‑peak  = first peak in the list
    # Valley         = first valley in the list
    # Positive‑peak  = **last** peak in the list (≠ third column!)

    neg = pk_mat[:, 0] if max_pk > 0 else np.full(n, np.nan)
    val = vl_mat[:, 0] if max_vl > 0 else np.full(n, np.nan)

    # the “last” peak has to be extracted explicitly or it disappears
    pos = np.full(n, np.nan)
    for i, pk in enumerate(peaks):
        if pk and len(pk) > 1:           # needs a distinct positive peak
            pos[i] = pk[-1]              # last element, true positive peak
    neg_thr = neg_thr if neg_thr is not None else np.arcsinh(10 / 5 + 1)

    if align_type == "valley":
        out = val[:, None]
        out[np.isnan(out[:, 0]), 0] = neg_thr
        return out

    # ---- only ONE peak detected ----------------------------------------
    # If the caller asked for the 3-landmark mode we still have to deliver
    # three columns, therefore we **impute** the missing positive peak by
    #   pos = valley + median(pos-valley)   (R logic)
    if max_pk <= 1:
        out = np.vstack([neg, val]).T          # (N, 2)  so far

        # fill NAs (neg & val) exactly as in the R code ------------------
        nan_neg = np.isnan(out[:, 0])
        if np.any(~nan_neg):
            out[nan_neg, 0] = np.nanmedian(out[~nan_neg, 0])

        nan_val = np.isnan(out[:, 1])
        alt = np.nanmedian(out[~nan_val, 1]) if np.any(~nan_val) else np.nan
        fill_val = np.where(
            neg_thr > out[nan_val, 0], neg_thr, alt
        )
        if np.isnan(alt):
            fill_val[:] = neg_thr
        out[nan_val, 1] = fill_val

        # --- return early for 1- or 2-anchor regimes --------------------
        if align_type == "negPeak":
            return out[:, :1]
        if align_type == "negPeak_valley" or (has_val_orig and not has_pos_orig):
            return out

        # ---- need a surrogate positive peak ----------------------------
        diff_candidates = pos - val
        valid_diff = ~np.isnan(diff_candidates)
        if np.any(valid_diff):
            diff_med = np.nanmedian(diff_candidates[valid_diff])
        else:
            diff_med = np.nanmedian(out[:, 1])
        if np.isnan(diff_med) or diff_med <= 0:
            diff_med = max(neg_thr, 1.0)

        pos_imputed = np.where(
            np.isnan(pos),
            out[:, 1] + diff_med,
            pos,
        )
        out = np.column_stack([out, pos_imputed])
        return out

    # ── have ≥2 peaks ──────────────────────────────────────────────────────
    if midpoint_type == "midpoint" and max_pk > 2 and max_vl > 0:
        use_val = np.nanmean(vl_mat, axis=1)
    else:
        use_val = val

    out = np.vstack([neg, use_val, pos]).T

    # valley NAs
    nan_val = np.isnan(out[:, 1])
    out[nan_val, 1] = neg_thr

    # neg-peak NAs
    nan_neg = np.isnan(out[:, 0])
    out[nan_neg, 0] = out[nan_neg, 1] / 2.0

    # pos-peak NAs
    nan_pos = np.isnan(out[:, 2])
    if np.any(~nan_pos):
        diff_med = np.nanmedian(out[~nan_pos, 2] - out[~nan_pos, 1])
    else:
        diff_med = np.nanmedian(out[:, 1])
    if np.isnan(diff_med) or diff_med <= 0:
        diff_med = max(neg_thr, 1.0)
    out[nan_pos, 2] = out[nan_pos, 1] + diff_med

    if align_type == "negPeak":
        return out[:, :1]
    if align_type == "negPeak_valley":
        return out[:, :2]
    return out


# ------------------------------------------------------------------  
def build_warp_function(
    landmarks_src : Sequence[float],
    landmarks_tgt : Sequence[float],
):
    """
    Return a **monotone piece-wise linear** function f such that
        f(landmarks_src[i]) == landmarks_tgt[i]     for all i
    and the slopes of the segments outside the landmark range equal the
    first / last internal slope, respectively.
    """
    ls = np.asarray(landmarks_src, float)
    lt = np.asarray(landmarks_tgt, float)
    if ls.size == 0 or np.any(np.isnan(ls)) or np.any(np.isnan(lt)):
        raise ValueError("Landmarks must be defined & numeric")

    # sort by source x
    order = np.argsort(ls)
    ls, lt = ls[order], lt[order]
    lt = np.maximum.accumulate(lt)

    # —— single-landmark → constant shift ————————————————
    if ls.size == 1:
        delta = float(lt[0] - ls[0])    
        def f(x):
            return np.asarray(x, float) + delta
    
        return f
    
    # slopes for each interval  (≥ 2 landmarks from here on)
    slopes = np.diff(lt) / np.diff(ls)

    def f(x):
        x = np.asarray(x, float)
        y = np.empty_like(x)

        # three regions: left of ls[0], internal, right of ls[-1]
        left_mask  = x <= ls[0]
        right_mask = x >= ls[-1]
        mid_mask   = ~(left_mask | right_mask)

        # ---- left - constant slope == slopes[0] --------------------------
        y[left_mask] = lt[0] + slopes[0] * (x[left_mask] - ls[0])

        # ---- right -------------------------------------------------------
        y[right_mask] = lt[-1] + slopes[-1] * (x[right_mask] - ls[-1])

        # ---- interior: vectorised piece-wise linear ---------------------
        if mid_mask.any():
            idx = np.searchsorted(ls, x[mid_mask], side="right") - 1
            d   = x[mid_mask] - ls[idx]
            y[mid_mask] = lt[idx] + slopes[idx] * d

        return y

    return f


# ------------------------------------------------------------------  
AlignmentReturn = Tuple[List[np.ndarray], np.ndarray, List]
AlignmentReturnWithDensity = Tuple[
    List[np.ndarray],
    np.ndarray,
    List,
    List[Optional[Tuple[np.ndarray, np.ndarray]]],
]


def align_distributions(
    counts         : List[np.ndarray],
    peaks          : List[Sequence[float]],
    valleys        : List[Sequence[float]],
    align_type     : str = "negPeak_valley_posPeak",
    landmark_matrix: Optional[np.ndarray] = None,
    target_landmark: Optional[Sequence[float]] = None,
    density_grids  : Optional[Sequence[Optional[DensityGrid]]] = None,
) -> Union[AlignmentReturn, AlignmentReturnWithDensity]:
    """
    • *counts* – list of raw (arcsinh-transformed) count vectors
    • returns (normalised_counts, warped_landmarks_matrix)
    • optionally returns aligned density grids when *density_grids* provided
    """
    peaks_use, valleys_use = _augment_single_peak_samples(peaks, valleys, counts)

    # ---------- fill / merge landmark matrix -----------------------------
    # 1. build / accept the landmark matrix  -----------------------------
    if landmark_matrix is None:
        lm = fill_landmark_matrix(
            peaks_use, valleys_use, align_type=align_type
        )
    else:
        lm = np.asarray(landmark_matrix, float)
        if lm.shape[0] != len(counts):
            raise ValueError("landmark_matrix rows ≠ #samples")

    n_samples = lm.shape[0]

    if density_grids is None:
        density_list = None
    else:
        if len(density_grids) != n_samples:
            raise ValueError("density_grids length does not match #samples")
        density_list: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        for grid in density_grids:
            if grid is None:
                density_list.append(None)
                continue
            xs, ys = grid
            xs_arr = np.asarray(xs, float)
            ys_arr = np.asarray(ys, float)
            if xs_arr.shape != ys_arr.shape:
                raise ValueError("density grid x/y arrays must share shape")
            density_list.append((xs_arr, ys_arr))

    # ---------- fill any remaining NA  (R’s rules: cohort median) --------
    nan_by_col = np.isnan(lm)
    if nan_by_col.any():
        col_median = np.nanmedian(lm, axis=0)
        lm[nan_by_col] = np.take(col_median, np.where(nan_by_col)[1])

    # ---------- choose the common target positions -----------------------
    tgt_raw = (np.nanmean(lm, axis=0)
               if target_landmark is None else np.asarray(target_landmark, float))
    tgt = np.maximum.accumulate(np.asarray(tgt_raw, float))
    if tgt.shape[0] != lm.shape[1]:
        raise ValueError(
            "target_landmark length does not match landmark columns"
        )

    # ---------- build & apply warps --------------------------------------
    warped_counts   : List[np.ndarray] = []
    warped_landmark = np.empty_like(lm)
    base_warp_funs  : List[callable]   = []      # <<< keep 1-to-1 length
    warped_density: Optional[List[Optional[Tuple[np.ndarray, np.ndarray]]]]
    if density_grids is None:
        warped_density = None
    else:
        warped_density = [None] * n_samples

    lower_bound: float | None = None
    all_mins: List[float] = []
    for c in counts:
        finite_c = np.asarray(c, float)[np.isfinite(c)]
        if finite_c.size:
            all_mins.append(float(finite_c.min()))
    if all_mins and min(all_mins) >= 0.0:
        lower_bound = 0.0

    global_min = np.inf if lower_bound is not None else None

    for i in range(n_samples):
        c = np.asarray(counts[i], float)
        l_src = lm[i]
        valid = ~np.isnan(l_src)

        if valid.any():
            base_f = build_warp_function(l_src[valid], tgt[valid])
            new_c = base_f(c)
            new_c[np.isnan(c)] = np.nan

            warped_counts.append(new_c)
            wl = np.full_like(l_src, np.nan)
            wl[valid] = base_f(l_src[valid])
        else:
            warped_counts.append(c.copy())
            wl = l_src.copy()
            base_f = _identity_warp

        warped_landmark[i] = wl
        base_warp_funs.append(base_f)

        if warped_density is not None:
            dens = density_list[i]
            if dens is None:
                warped_density[i] = None
            else:
                xs, ys = dens

                # Constrain the density grid to the observed range before warping.
                xs_clipped = np.asarray(xs, float)
                if lower_bound is not None:
                    xs_clipped = np.clip(xs_clipped, lower_bound, None)

                warped_xs = base_f(xs_clipped)

                warped_density[i] = (warped_xs, ys.copy())

        if lower_bound is not None:
            candidates = [warped_counts[-1], wl]
            if warped_density is not None and warped_density[i] is not None:
                candidates.append(warped_density[i][0])
            for arr in candidates:
                finite_vals = np.asarray(arr, float)[np.isfinite(arr)]
                if finite_vals.size:
                    min_val = float(finite_vals.min())
                    global_min = min(global_min, min_val)

    shift = 0.0
    if lower_bound is not None and global_min is not None and global_min != np.inf:
        shift = max(0.0, lower_bound - global_min)

    warp_funs: List[callable] = []
    if shift:
        for i, base_f in enumerate(base_warp_funs):
            warped_counts[i] = warped_counts[i] + shift
            warped_landmark[i] = warped_landmark[i] + shift
            if warped_density is not None and warped_density[i] is not None:
                xs_w, ys_w = warped_density[i]
                warped_density[i] = (xs_w + shift, ys_w)
            warp_funs.append(lambda x, f=base_f, s=shift: np.asarray(f(x), float) + s)
    else:
        warp_funs = base_warp_funs

    if warped_density is None:
        return warped_counts, warped_landmark, warp_funs

    return warped_counts, warped_landmark, warp_funs, warped_density