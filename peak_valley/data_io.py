from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pandas as pd

__all__ = ["arcsinh_transform", "read_counts"]


# ------------------------------------------------------------------
def arcsinh_transform(
    x: pd.Series | np.ndarray,
    a: float = 1.0,
    b: float = 1 / 5,
    c: float = 0.0,
) -> pd.Series:
    """Same arcsinh transform you used in the notebook."""
    return (1 / b) * np.arcsinh(a * x + c)


# ------------------------------------------------------------------
def read_counts(
    file: io.BytesIO | str | Path,
    header_row: int,
    skip_rows: int,
) -> np.ndarray:
    """
    Read a *_raw_counts.csv* that contains **one numeric column**.
    Handles giant files by streaming.
    """
    # accept path-like too (handy for future CLI use)
    if not hasattr(file, "read"):
        file = open(file, "rb")

    file.seek(0)
    hdr = None if header_row < 0 else header_row
    try:                                             # fast path
        return (
            pd.read_csv(
                file,
                header=hdr,
                skiprows=skip_rows or None,
                usecols=[0],
                dtype="float64",
                engine="c",
                memory_map=True,
            )
            .squeeze("columns")
            .values
        )
    except ValueError:                              # stream when huge
        file.seek(0)
        chunks = pd.read_csv(
            file,
            header=hdr,
            skiprows=skip_rows or None,
            usecols=[0],
            dtype="float64",
            engine="c",
            chunksize=200_000,
        )
        return np.concatenate([c.values.ravel() for c in chunks])
