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
) -> tuple[np.ndarray, dict[str, str | None]]:
    """
    Read a *_raw_counts.csv* that contains one column of counts.

    The function tolerates a free-form text entry (for example the marker name)
    in the first data row and returns it via the accompanying metadata dict.
    """

    def _extract_values(series: pd.Series,
                        protein: str | None) -> tuple[np.ndarray, str | None]:
        """Return numeric values and update ``protein`` when present."""

        raw_values = series.astype("object")
        numeric = pd.to_numeric(raw_values, errors="coerce")

        if protein is None:
            mask = numeric.isna()
            if mask.any():
                first = raw_values[mask].dropna()
                for value in first:
                    text = str(value).strip()
                    if text and text.lower() not in {"nan", "none"}:
                        protein = text
                        break

        valid = numeric.dropna().astype("float64")
        return valid.values, protein

    # accept path-like too (handy for future CLI use)
    if not hasattr(file, "read"):
        file = open(file, "rb")

    file.seek(0)
    hdr = None if header_row < 0 else header_row
    common_kwargs = dict(
        header=hdr,
        skiprows=skip_rows or None,
        usecols=[0],
        engine="c",
    )

    protein_name: str | None = None

    try:  # fast path
        series = pd.read_csv(
            file,
            dtype="object",
            memory_map=True,
            **common_kwargs,
        ).squeeze("columns")
        values, protein_name = _extract_values(series, protein_name)
    except ValueError:  # stream when huge
        file.seek(0)
        chunks = pd.read_csv(
            file,
            dtype="object",
            chunksize=200_000,
            **common_kwargs,
        )
        arrays: list[np.ndarray] = []
        for frame in chunks:
            col = frame.squeeze("columns")
            vals, protein_name = _extract_values(col, protein_name)
            if vals.size:
                arrays.append(vals)
        values = np.concatenate(arrays) if arrays else np.empty(0)

    metadata = {"protein_name": protein_name}
    return values, metadata
