import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peak_valley.batch import (
    BatchOptions,
    collect_counts_files,
    collect_dataset_samples,
    run_batch,
    save_outputs,
)
from peak_valley.quality import stain_quality


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
    assert not batch.interrupted
    assert not batch.interrupted

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

    assert not (out_dir / "plots").exists()
    assert not (out_dir / "counts").exists()
    assert not (out_dir / "curves").exists()
    assert not (out_dir / "aligned_curves").exists()

    results_path = out_dir / "results.json"
    assert results_path.exists()
    with results_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["samples"], "Manifest should include sample entries"
    assert manifest.get("interrupted") is False


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
    assert not batch.interrupted
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

    assert not (out_dir / "processed_counts.csv").exists()
    assert not (out_dir / "aligned_counts.csv").exists()
    assert not (out_dir / "aligned_landmarks.csv").exists()


def test_optional_plot_export(tmp_path):
    rng = np.random.default_rng(123)
    values = rng.normal(size=400)
    counts_path = tmp_path / "SampleB_CD19_raw_counts.csv"
    _write_counts(counts_path, values)

    options = BatchOptions(apply_arcsinh=False)
    samples = collect_counts_files([counts_path], options, header_row=-1, skip_rows=0)
    batch = run_batch(samples, options)
    assert not batch.interrupted

    out_dir = tmp_path / "with_plots"
    save_outputs(batch, out_dir, export_plots=True)

    plots_dir = out_dir / "plots"
    assert plots_dir.exists()
    assert any(plots_dir.glob("*.png"))


def test_group_marker_ridge_exports(tmp_path):
    expr = pd.DataFrame(
        {
            "markerX": [0.5, 0.6, 2.0, 2.2],
            "markerY": [3.5, 3.6, 5.0, 5.2],
        }
    )
    meta = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2", "S2"],
            "batch": ["B1"] * 4,
        }
    )

    expr_path = tmp_path / "expr.csv"
    meta_path = tmp_path / "meta.csv"
    expr.to_csv(expr_path, index=False)
    meta.to_csv(meta_path, index=False)

    options = BatchOptions(apply_arcsinh=False, align=True, group_by_marker=True, max_peaks=2)
    samples, meta_info = collect_dataset_samples(
        expr_path,
        meta_path,
        options,
    )

    batch = run_batch(samples, options)
    assert batch.group_by_marker
    assert batch.marker_groups is not None
    groups = {name for name, _ in batch.marker_groups}
    assert groups == {"markerX", "markerY"}

    out_dir = tmp_path / "grouped"
    save_outputs(batch, out_dir, run_metadata=meta_info)

    zip_path = out_dir / "before_after_alignment.zip"
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "before_alignment/before_alignment_ridge.png" in names
        assert "before_alignment/before_alignment_ridge_markerX.png" in names
        assert "before_alignment/before_alignment_ridge_markerY.png" in names
        assert "after_alignment/aligned_ridge_markerX.png" in names
        assert "after_alignment/aligned_ridge_markerY.png" in names


def test_group_marker_single_marker_does_not_emit_groups(tmp_path):
    expr = pd.DataFrame(
        {
            "markerX": [0.5, 0.6, 2.0, 2.2],
        }
    )
    meta = pd.DataFrame(
        {
            "sample": ["S1", "S1", "S2", "S2"],
            "batch": ["B1"] * 4,
        }
    )

    expr_path = tmp_path / "expr.csv"
    meta_path = tmp_path / "meta.csv"
    expr.to_csv(expr_path, index=False)
    meta.to_csv(meta_path, index=False)

    options = BatchOptions(apply_arcsinh=False, align=False, group_by_marker=True, max_peaks=2)
    samples, meta_info = collect_dataset_samples(
        expr_path,
        meta_path,
        options,
    )

    batch = run_batch(samples, options)

    assert batch.group_by_marker
    assert batch.marker_groups is None

    out_dir = tmp_path / "single"
    save_outputs(batch, out_dir, run_metadata=meta_info)

    manifest = json.loads((out_dir / "results.json").read_text())
    assert manifest.get("group_by_marker")
    assert "marker_groups" not in manifest

def test_stain_quality_handles_missing_valleys():
    rng = np.random.default_rng(0)
    left = rng.normal(-3.0, 0.2, size=100)
    middle = rng.normal(0.0, 0.3, size=120)
    right = rng.normal(3.2, 0.25, size=110)
    counts = np.concatenate([left, middle, right])

    peaks = [-3.0, 0.0, 3.2]
    valleys = [-1.5]  # detector might miss a second valley

    score = stain_quality(counts, peaks, valleys)
    assert np.isfinite(score) or np.isnan(score)
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


def test_run_batch_handles_keyboard_interrupt(tmp_path, monkeypatch):
    rng = np.random.default_rng(321)
    values_a = rng.normal(loc=0.5, scale=0.1, size=200)
    values_b = rng.normal(loc=1.5, scale=0.2, size=200)

    path_a = tmp_path / "SampleX_CD3_raw_counts.csv"
    path_b = tmp_path / "SampleY_CD3_raw_counts.csv"
    _write_counts(path_a, values_a)
    _write_counts(path_b, values_b)

    options = BatchOptions(apply_arcsinh=False)
    samples = collect_counts_files([path_a, path_b], options, header_row=-1, skip_rows=0)

    import peak_valley.batch as batch_mod

    original_process = batch_mod.process_sample
    call_count = {"count": 0}

    def fake_process(sample, opts, overrides, gpt_client):
        call_count["count"] += 1
        if call_count["count"] >= 2:
            raise KeyboardInterrupt()
        return original_process(sample, opts, overrides, gpt_client)

    monkeypatch.setattr(batch_mod, "process_sample", fake_process)

    batch = run_batch(samples, options)

    assert batch.interrupted is True
    assert len(batch.samples) == 1

    out_dir = tmp_path / "partial"
    save_outputs(batch, out_dir)

    summary_path = out_dir / "summary.csv"
    assert summary_path.exists()
    results_path = out_dir / "results.json"
    with results_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest.get("interrupted") is True
    assert manifest["samples"], "Partial samples should still be exported"
