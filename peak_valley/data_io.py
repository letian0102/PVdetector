from __future__ import annotations
import io
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

__all__ = ["arcsinh_transform", "read_counts", "load_combined_csv"]


# ------------------------------------------------------------------
def arcsinh_transform(
    x: pd.Series | np.ndarray,
    a: float = 1.0,
    b: float = 1 / 5,
    c: float = 0.0,
) -> pd.Series:
    """Apply ``asinh(a + b * x) + c`` to ``x``.

    The parameters mirror the R implementation outlined in the user-facing
    documentation: ``a`` is a shift applied inside the ``asinh`` call, ``b``
    scales the input before transformation, and ``c`` shifts the result
    afterward.
    """

    return np.arcsinh(a + b * x) + c


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


# ------------------------------------------------------------------
def load_combined_csv(
    file: io.BytesIO | bytes | str | Path,
    *,
    low_memory: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a DataFrame from ``file``, concatenating CSV parts when needed.

    Parameters
    ----------
    file:
        A path or file-like object pointing to a CSV file. The input may also be
        a ZIP archive containing one or more CSV files that share the same
        column structure. When multiple CSV files are present in the archive,
        they are read in sorted order and concatenated row-wise.
    low_memory:
        Passed to :func:`pandas.read_csv` for every CSV that is parsed.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        The combined dataframe together with a list of source file names used
        to produce it. The list is empty when a plain CSV was provided.

    Raises
    ------
    ValueError
        If the archive contains no CSV files or if the CSV parts disagree on
        their column layout.
    """

    close_after = False

    if isinstance(file, (str, Path)):
        file = open(file, "rb")
        close_after = True

    try:
        data: bytes
        if isinstance(file, bytes):
            data = file
        else:
            data = file.read()
            if hasattr(file, "seek"):
                file.seek(0)

        buffer = io.BytesIO(data)
        buffer.seek(0)

        if zipfile.is_zipfile(buffer):
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as archive:
                members = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".csv")
                    and not name.endswith("/")
                    and "__MACOSX" not in name
                ]
                members.sort()

                frames: list[pd.DataFrame] = []
                for name in members:
                    with archive.open(name) as fh:
                        frames.append(pd.read_csv(fh, low_memory=low_memory))

                if not frames:
                    raise ValueError("The archive does not contain any CSV files.")

                columns = list(frames[0].columns)
                for idx, frame in enumerate(frames[1:], start=1):
                    if list(frame.columns) != columns:
                        raise ValueError(
                            "CSV files in the archive have inconsistent columns "
                            f"(mismatch at part {idx + 1})."
                        )

                combined = pd.concat(frames, ignore_index=True)
                return combined, members

        buffer.seek(0)
        frame = pd.read_csv(buffer, low_memory=low_memory)
        return frame, []
    finally:
        if close_after:
            file.close()
