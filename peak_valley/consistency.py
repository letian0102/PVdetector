from __future__ import annotations
import numpy as np
from typing import Dict, Sequence

__all__ = ["enforce_marker_consistency"]


def _enforce_valley_rule(peaks: Sequence[float],
                         valleys: Sequence[float]) -> list[float]:
    """Keep at most one valley after the first peak and one between each pair."""

    peaks = sorted(peaks)
    valleys = sorted(valleys)
    if not peaks:
        return []
    if len(peaks) == 1:
        for v in valleys:
            if v > peaks[0]:
                return [v]
        return []

    kept: list[float] = []
    j = 0
    for left, right in zip(peaks[:-1], peaks[1:]):
        while j < len(valleys) and valleys[j] <= left:
            j += 1
        if j < len(valleys) and left < valleys[j] < right:
            kept.append(valleys[j])
            j += 1
    return kept


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
    """Detect and correct peak/valley outliers within each marker group.

    Samples may contain data for multiple protein markers.  Each entry in
    ``results`` may therefore include a ``"marker"`` field identifying the
    protein marker for that sample.  Landmarks are aligned only among samples
    sharing the same marker value.  Samples lacking a ``"marker"`` entry are
    grouped together and treated as a single marker.  Outliers are snapped to
    local extremes near the consensus position and missing peaks/valleys are
    added in the same way.  After landmarks beyond the first are corrected,
    the first peak and valley are adjusted using the final consensus so that
    every sample in a marker group is treated consistently.

    Parameters
    ----------
    results : dict
        Mapping of sample name to the detector output.  Each value must
        contain ``peaks``, ``valleys``, ``xs`` and ``ys`` entries and may
        optionally define ``marker`` with the associated protein marker name.
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

    def _enforce_group(group: Dict[str, Dict[str, Sequence[float]]]) -> None:
        if len(group) < 2:
            return

        pk_lists = []
        vl_lists = []
        for info in group.values():
            pk = list(info.get("peaks", []))
            vl = _enforce_valley_rule(pk, info.get("valleys", []))
            info["peaks"], info["valleys"] = pk, vl
            pk_lists.append(pk)
            vl_lists.append(vl)

        if not pk_lists:
            return

        n_peaks = max(len(p) for p in pk_lists)
        n_valleys = max((len(v) for v in vl_lists), default=0)
        pk_cons = [float(np.median([p[i] for p in pk_lists if len(p) > i]))
                   for i in range(n_peaks)]
        vl_cons = [float(np.median([v[i] for v in vl_lists if len(v) > i]))
                   for i in range(n_valleys)]

        win = tol if window is None else window

        for info in group.values():
            xs = np.asarray(info.get("xs", []), float)
            ys = np.asarray(info.get("ys", []), float)
            pk = list(info.get("peaks", []))
            vl = list(info.get("valleys", []))

            for i, exp in enumerate(pk_cons):
                if i < len(pk):
                    if not np.isfinite(pk[i]) or abs(pk[i] - exp) > tol:
                        pk[i] = _local_extreme(xs, ys, exp, win, True)
                else:
                    pk.append(_local_extreme(xs, ys, exp, win, True))

            for i, exp in enumerate(vl_cons):
                if i < len(vl):
                    if not np.isfinite(vl[i]) or abs(vl[i] - exp) > tol:
                        vl[i] = _local_extreme(xs, ys, exp, win, False)
                else:
                    vl.append(_local_extreme(xs, ys, exp, win, False))

            info["peaks"], info["valleys"] = pk, vl

        # After processing later landmarks, revisit the first peak/valley
        pk_lists = [info["peaks"] for info in group.values()
                    if info.get("peaks") is not None]
        vl_lists = [info["valleys"] for info in group.values()
                    if info.get("valleys") is not None]
        pk_cons0 = float(np.median([p[0] for p in pk_lists if len(p) > 0]))
        vl_cons0 = (float(np.median([v[0] for v in vl_lists if len(v) > 0]))
                    if vl_lists else None)

        for info in group.values():
            xs = np.asarray(info.get("xs", []), float)
            ys = np.asarray(info.get("ys", []), float)
            pk = list(info.get("peaks", []))
            vl = list(info.get("valleys", []))

            if pk and abs(pk[0] - pk_cons0) > tol:
                pk[0] = _local_extreme(xs, ys, pk_cons0, win, True)

            if vl and vl_cons0 is not None and abs(vl[0] - vl_cons0) > tol:
                vl[0] = _local_extreme(xs, ys, vl_cons0, win, False)

            vl = _enforce_valley_rule(pk, vl)
            info["peaks"], info["valleys"] = pk, vl

    if len(results) < 2:
        return

    groups: Dict[str | None, Dict[str, Dict[str, Sequence[float]]]] = {}
    for stem, info in results.items():
        marker = info.get("marker")
        groups.setdefault(marker, {})[stem] = info

    for group in groups.values():
        _enforce_group(group)
