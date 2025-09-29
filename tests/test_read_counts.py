from __future__ import annotations

import io

import numpy as np

from peak_valley.data_io import read_counts


def _make_csv(contents: str) -> io.BytesIO:
    bio = io.BytesIO(contents.encode("utf-8"))
    bio.seek(0)
    return bio


def test_read_counts_extracts_protein_name_without_header() -> None:
    csv = "\n".join(["CD3", "1", "2", "3"]) + "\n"
    counts, meta = read_counts(_make_csv(csv), header_row=-1, skip_rows=0)

    assert meta["protein_name"] == "CD3"
    np.testing.assert_array_equal(counts, np.array([1.0, 2.0, 3.0]))


def test_read_counts_respects_header_row() -> None:
    csv = "\n".join(["Counts", "CD4", "5", "6"]) + "\n"
    counts, meta = read_counts(_make_csv(csv), header_row=0, skip_rows=0)

    assert meta["protein_name"] == "CD4"
    np.testing.assert_array_equal(counts, np.array([5.0, 6.0]))
