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
from typing import Sequence, Tuple, List, Optional, Iterable

__all__ = ["fill_landmark_matrix",
           "build_warp_function",
           "align_distributions"]


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
        if pk:                           # non‑empty list
            pos[i] = pk[-1]              # last element, no matter how many
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
        pos_fill = out[:, 1] + diff_med
        out      = np.column_stack([out, pos_fill])
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
def align_distributions(
    counts         : List[np.ndarray],
    peaks          : List[Sequence[float]],
    valleys        : List[Sequence[float]],
    align_type     : str = "negPeak_valley_posPeak",
    landmark_matrix: Optional[np.ndarray] = None,
    target_landmark: Optional[Sequence[float]] = None,
) -> Tuple[List[np.ndarray], np.ndarray, List]:
    """
    • *counts* – list of raw (arcsinh-transformed) count vectors  
    • returns (normalised_counts, warped_landmarks_matrix)
    """
    # ---------- fill / merge landmark matrix -----------------------------
    # 1. build / accept the landmark matrix  -----------------------------
    if landmark_matrix is None:
        lm = fill_landmark_matrix(
            peaks, valleys, align_type=align_type
        )
    else:
        lm = np.asarray(landmark_matrix, float)
        if lm.shape[0] != len(counts):
            raise ValueError("landmark_matrix rows ≠ #samples")

    # ---------- fill any remaining NA  (R’s rules: cohort median) --------
    nan_by_col = np.isnan(lm)
    if nan_by_col.any():
        col_median = np.nanmedian(lm, axis=0)
        lm[nan_by_col] = np.take(col_median, np.where(nan_by_col)[1])

    # ---------- choose the common target positions -----------------------
    tgt = (np.nanmean(lm, axis=0)
           if target_landmark is None else np.asarray(target_landmark, float))
    if tgt.shape[0] != lm.shape[1]:
        raise ValueError(
            "target_landmark length does not match landmark columns"
        )

    # ---------- build & apply warps --------------------------------------
    warped_counts   : List[np.ndarray] = []
    warped_landmark = np.empty_like(lm)
    warp_funs       : List[callable]   = []      # <<< keep 1-to-1 length
    for i, (c, l_src) in enumerate(zip(counts, lm)):
        valid = ~np.isnan(l_src)
        if not valid.any():                 # nothing to align…
            warped_counts.append(c.copy())
            warped_landmark[i] = l_src

            def _identity(x):
                return np.asarray(x, float)

            warp_funs.append(_identity)
            continue

        f = build_warp_function(l_src[valid], tgt[valid])

        new_c = f(c)
        # keep NaNs intact (e.g. missing cells in whole-dataset mode)
        new_c[np.isnan(c)] = np.nan

        warped_counts.append(new_c)
        wl = np.full_like(l_src, np.nan)
        wl[valid] = f(l_src[valid])
        warped_landmark[i] = wl
        warp_funs.append(f)             # <<< inside the loop!

    return warped_counts, warped_landmark, warp_funs