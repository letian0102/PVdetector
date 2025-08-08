from __future__ import annotations
import numpy as np
from typing import Dict, Sequence

__all__ = ["enforce_marker_consistency"]


def _local_extreme(xs: np.ndarray, ys: np.ndarray, center: float,
                   window: float, find_max: bool) -> float:
    """Return local maximum or minimum near *center* within *window*.

    If the window does not overlap ``xs`` at all the original ``center``
    value is returned unchanged.
    """
    mask = (xs >= center - window) & (xs <= center + window)
    if not mask.any():
        return center
    seg_x = xs[mask]
    seg_y = ys[mask]
    idx = np.argmax(seg_y) if find_max else np.argmin(seg_y)
    return float(seg_x[idx])


def enforce_marker_consistency(results: Dict[str, Dict[str, Sequence[float]]],
                               tol: float = 0.5,
                               window: float | None = None) -> None:
    """Detect and correct peak/valley outliers across samples of a marker.

    Parameters
    ----------
    results : dict
        Mapping of sample name to the detector output.  Each value must
        contain ``peaks``, ``valleys``, ``xs`` and ``ys`` entries.
    tol : float, optional
        Deviation from the consensus beyond which a landmark is treated
        as an outlier.  Expressed in the same units as ``peaks``.
    window : float | None, optional
        Search window around the consensus position when fixing outliers.
        Defaults to ``tol`` if ``None``.

    Notes
    -----
    The function updates ``results`` *in place*.
    """
    if len(results) < 2:
        return

    pk_lists = [info["peaks"] for info in results.values() if info.get("peaks")]
    vl_lists = [info["valleys"] for info in results.values() if info.get("valleys")]
    if not pk_lists:
        return

    n_peaks = min(len(p) for p in pk_lists)
    n_valleys = min(len(v) for v in vl_lists) if vl_lists else 0
    if n_peaks == 0:
        return

    pk_cons = [float(np.median([p[i] for p in pk_lists if len(p) > i]))
               for i in range(n_peaks)]
    vl_cons = [float(np.median([v[i] for v in vl_lists if len(v) > i]))
               for i in range(n_valleys)]

    win = tol if window is None else window

    for info in results.values():
        xs = np.asarray(info.get("xs", []), float)
        ys = np.asarray(info.get("ys", []), float)
        pk = list(info.get("peaks", []))
        vl = list(info.get("valleys", []))

        for i, exp in enumerate(pk_cons):
            if i < len(pk) and abs(pk[i] - exp) > tol:
                pk[i] = _local_extreme(xs, ys, exp, win, True)

        for i, exp in enumerate(vl_cons):
            if i < len(vl) and abs(vl[i] - exp) > tol:
                vl[i] = _local_extreme(xs, ys, exp, win, False)

        info["peaks"], info["valleys"] = pk, vl
