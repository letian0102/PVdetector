from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from peak_valley.data_io import load_combined_csv


def test_load_combined_csv_plain_bytes() -> None:
    payload = b"cell_id,markerA\n1,2.0\n2,3.5\n"
    df, sources = load_combined_csv(io.BytesIO(payload))

    assert list(df.columns) == ["cell_id", "markerA"]
    assert df.shape == (2, 2)
    assert sources == []


def test_load_combined_csv_zip_concatenates_parts() -> None:
    df1 = pd.DataFrame({"cell_id": [1, 2], "markerA": [0.1, 0.2]})
    df2 = pd.DataFrame({"cell_id": [3], "markerA": [0.3]})

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("folder/part1.csv", df1.to_csv(index=False))
        z.writestr("part2.csv", df2.to_csv(index=False))
        z.writestr("__MACOSX/._junk", "")

    buf.seek(0)
    df, sources = load_combined_csv(io.BytesIO(buf.getvalue()))

    assert df.shape == (3, 2)
    assert list(df["cell_id"]) == [1, 2, 3]
    assert sources == ["folder/part1.csv", "part2.csv"]


def test_load_combined_csv_zip_mismatched_columns() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.csv", "cell_id,markerA\n1,2\n")
        z.writestr("b.csv", "cell_id,markerB\n1,5\n")

    buf.seek(0)

    with pytest.raises(ValueError):
        load_combined_csv(io.BytesIO(buf.getvalue()))
