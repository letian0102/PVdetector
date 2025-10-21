"""Helpers for importing CLI summary outputs into the Streamlit app."""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = ["parse_peak_positions", "derive_min_separation"]


def parse_peak_positions(values: Any) -> list[float]:
    """Parse peak positions from CSV summary cells.

    The CLI summary encodes peak locations as a semi-colon separated string.
    This helper accepts strings, iterables, or sequences and returns the
    numeric values that could be parsed successfully.
    """

    if values is None:
        return []

    if isinstance(values, str):
        parts = [part.strip() for part in values.replace(",", ";").split(";")]
    elif isinstance(values, Sequence):  # handles lists/tuples/np arrays
        parts = list(values)
    elif isinstance(values, Iterable):
        parts = list(values)
    elif isinstance(values, numbers.Real) and not isinstance(values, bool):
        value = float(values)
        if math.isfinite(value):
            return [value]
        return []
    else:
        return []

    peaks: list[float] = []
    for part in parts:
        if part in ("", None):
            continue
        try:
            peaks.append(float(part))
        except (TypeError, ValueError):
            continue
    return peaks


def derive_min_separation(peaks: Sequence[float], baseline: float | None = None) -> float | None:
    """Infer a ``min_separation`` override from detected peak locations.

    Parameters
    ----------
    peaks:
        Iterable of peak locations.
    baseline:
        Baseline minimum separation (e.g. current UI default).  When provided
        the derived value is only returned if the spacing between the CLI
        peaks is smaller than the baseline.
    """

    if not peaks:
        return None

    try:
        ordered = sorted(float(p) for p in peaks if p is not None)
    except (TypeError, ValueError):
        return None

    gaps: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        diff = float(right) - float(left)
        if diff > 0:
            gaps.append(diff)

    if not gaps:
        return None

    smallest = min(gaps)
    if baseline is not None and smallest >= baseline:
        return None

    # Leave a tiny margin below the observed spacing so the detector is free
    # to rediscover the original peaks even if they were closer than the new
    # UI default.
    buffer = max(smallest * 0.05, 1e-6)
    adjusted = smallest - buffer
    if adjusted <= 0:
        adjusted = smallest * 0.5
    return adjusted
