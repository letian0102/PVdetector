import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from peak_valley.batch import (
    BatchOptions,
    collect_counts_files,
    collect_dataset_samples,
    run_batch,
    save_outputs,
)


def _write_counts(path: Path, values: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for val in values:
            fh.write(f"{val}\n")


def test_run_batch_on_counts(tmp_path):
    rng = np.random.default_rng(42)
    low = rng.normal(loc=-2.0, scale=0.2, size=200)
    high = rng.normal(loc=2.0, scale=0.3, size=200)
    values = np.concatenate([low, high])

    counts_path = tmp_path / "SampleA_CD3_raw_counts.csv"
    _write_counts(counts_path, values)

    options = BatchOptions(apply_arcsinh=False)
    samples = collect_counts_files([counts_path], options, header_row=-1, skip_rows=0)
    batch = run_batch(samples, options)

    assert len(batch.samples) == 1
    result = batch.samples[0]
    assert result.peaks, "Detector should find at least one peak"
    assert result.params["n_peaks"] >= 1

    out_dir = tmp_path / "outputs"
    save_outputs(batch, out_dir)

    summary_path = out_dir / "summary.csv"
    assert summary_path.exists()
    summary_df = pd.read_csv(summary_path)
    assert counts_path.stem in summary_df["stem"].tolist()

    plots = list((out_dir / "plots").glob("*.png"))
    assert plots, "Expected per-sample plot export"


def test_combined_zip_has_expected_exports(tmp_path):
    expr = pd.DataFrame(
        {
            "markerA": [
                0.8,
                1.1,
                1.0,
                3.8,
                4.1,
                4.0,
                2.2,
                2.4,
            ],
            "markerB": [
                4.0,
                4.2,
                4.1,
                6.8,
                7.1,
                6.9,
                5.5,
                5.8,
            ],
        }
    )
    meta = pd.DataFrame(
        {
            "sample": ["S1"] * 4 + ["S2"] * 4,
            "batch": ["B1"] * 8,
        }
    )

    expr_path = tmp_path / "expr.csv"
    meta_path = tmp_path / "meta.csv"
    expr.to_csv(expr_path, index=False)
    meta.to_csv(meta_path, index=False)

    options = BatchOptions(apply_arcsinh=False, align=True, max_peaks=2)
    samples, meta_info = collect_dataset_samples(
        expr_path,
        meta_path,
        options,
    )

    batch = run_batch(samples, options)
    out_dir = tmp_path / "outputs"
    save_outputs(batch, out_dir, run_metadata=meta_info)

    zip_path = out_dir / "before_after_alignment.zip"
    assert zip_path.exists(), "Expected combined zip export"

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "before_alignment/cell_metadata_combined.csv" in names
        assert "before_alignment/expression_matrix_combined.csv" in names
        assert "before_alignment/before_alignment_ridge.png" in names
        assert "after_alignment/aligned_ridge.png" in names
        assert "after_alignment/expression_matrix_aligned.csv" in names
        # ensure per-sample CSVs are not exported
        assert all("raw_counts" not in name for name in names)

        meta_bytes = archive.read("before_alignment/cell_metadata_combined.csv")
        meta_df = pd.read_csv(io.BytesIO(meta_bytes))
        assert set(meta_df["sample"]) == {"S1", "S2"}
def test_collect_dataset_samples(tmp_path):
    expr = pd.DataFrame(
        {
            "markerA": [1.0, 2.0, 3.0, 4.0],
            "markerB": [5.0, 6.0, 7.0, 8.0],
        }
    )
    meta = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2", "S2"],
            "batch": ["B1", "B1", "B2", "B2"],
        }
    )

    expr_path = tmp_path / "expr.csv"
    meta_path = tmp_path / "meta.csv"
    expr.to_csv(expr_path, index=False)
    meta.to_csv(meta_path, index=False)

    options = BatchOptions(apply_arcsinh=False)
    samples, meta_info = collect_dataset_samples(
        expr_path,
        meta_path,
        options,
        markers=["markerA"],
        samples_filter=["S1"],
        batches=["B1"],
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.metadata["sample"] == "S1"
    assert sample.metadata["marker"] == "markerA"
    assert sample.metadata["batch"] == "B1"
    assert np.allclose(sample.counts, np.array([1.0, 2.0]))
    assert "expression_sources" in meta_info
