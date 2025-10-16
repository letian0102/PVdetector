import json
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
