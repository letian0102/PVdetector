from __future__ import annotations
import numpy as np
from typing import Sequence
from .kde_detector import kde_peaks_valleys
from .signature     import shape_signature          # you already ship this

__all__ = ["auto_bandwidth"]

# (re-use the runtime cache already defined in gpt_adapter)
try:
    from .gpt_adapter import _cache                 # (tag, sig) ➟ value
except ImportError:
    _cache = {}                                     # paranoia fallback


def _first_plateau(values: Sequence[int], width: int = 3) -> int | None:
    """
    Return the index where the first run of identical values of length ≥ `width`
    starts, or None if no such run.
    """
    run_len = 1
    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            run_len += 1
            if run_len >= width:
                return i - width + 1
        else:
            run_len = 1
    return None


def auto_bandwidth(
    counts_full: np.ndarray,
    prominence: float   = 0.05,
    min_width:  int | None = None,
    grid_size:  int      = 20_000,
    plateau:    int      = 3,
    fallback:   float    = 0.8,      # if all else fails
) -> float:
    """
    Pick a KDE bandwidth *scale factor* purely from the sample.

    • Returns a float in [0.3 … 1.0] (1.0 = Scott).
    • Uses memo-cache keyed by the distribution signature so repeated
      calls are O(1).
    """
    sig = shape_signature(counts_full)
    key = ("bw_auto", sig)
    if key in _cache:
        return _cache[key]

    scales = np.linspace(0.3, 1.0, 9)              # 0.3, 0.4, …, 1.0
    n_peaks = []
    for s in scales:
        peaks, *_ = kde_peaks_valleys(
            counts_full,
            n_peaks=None,
            prominence=prominence,
            bw=s,
            min_width=min_width,
            grid_size=grid_size,
        )
        n_peaks.append(len(peaks))

    # 1️⃣ plateau rule
    idx = _first_plateau(n_peaks, width=plateau)
    if idx is not None:
        val = float(scales[idx])

    # 2️⃣ elbow fallback
    else:
        diffs = np.abs(np.diff(n_peaks))
        val   = float(scales[np.argmin(diffs) + 1]) if diffs.size else fallback

    val = float(np.clip(val, 0.3, 1.0))
    _cache[key] = val
    return val
