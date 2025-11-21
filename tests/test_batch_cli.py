import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peak_valley.batch import (
    BatchOptions,
    BatchResults,
    SampleInput,
    SampleResult,
    collect_counts_files,
    collect_dataset_samples,
    run_batch,
    save_outputs,
)
from peak_valley.alignment import build_warp_function
from peak_valley.quality import stain_quality


def _write_counts(path: Path, values: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for val in values:
            fh.write(f"{val}\n")


def test_collect_counts_files_sanitizes_and_deduplicates(tmp_path):
    path_one = tmp_path / "Sample !.csv"
    path_two = tmp_path / "Sample @.csv"
    _write_counts(path_one, np.array([1.0, 2.0]))
    _write_counts(path_two, np.array([3.0, 4.0]))

    options = BatchOptions(apply_arcsinh=False)
    samples = collect_counts_files(
        [path_one, path_two],
        options,
        header_row=-1,
        skip_rows=0,
    )

    stems = [sample.stem for sample in samples]
    assert stems == ["Sample", "Sample_2"]
    assert samples[0].metadata["source_stem"] == "Sample !"
    assert samples[1].metadata["source_stem"] == "Sample @"


def test_collect_dataset_samples_sanitizes_stems(tmp_path):
    expr = pd.DataFrame(
        {
            "CD3+": [0.2, 0.5, 0.8, 1.0],
            "CD3*": [1.2, 1.5, 1.8, 2.0],
        }
    )
    meta = pd.DataFrame(
        {
            "sample": ["Alpha/Beta"] * 4,
        }
    )

    expr_path = tmp_path / "expr.csv"
    meta_path = tmp_path / "meta.csv"
    expr.to_csv(expr_path, index=False)
    meta.to_csv(meta_path, index=False)

    options = BatchOptions(apply_arcsinh=False)
    samples, _ = collect_dataset_samples(
        expr_path,
        meta_path,
        options,
    )

    stems = [sample.stem for sample in samples]
    assert stems == ["Alpha_Beta_CD3_raw_counts", "Alpha_Beta_CD3_raw_counts_2"]
    assert samples[0].metadata["source_stem"] == "Alpha/Beta_CD3+_raw_counts"
    assert samples[1].metadata["source_stem"] == "Alpha/Beta_CD3*_raw_counts"


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

    summary_files = list(out_dir.glob("summary_*.csv"))
    assert summary_files, "Expected a sanitised summary export"
    summary_path = summary_files[0]
    summary_df = pd.read_csv(summary_path)
    assert counts_path.stem in summary_df["stem"].tolist()

    assert not (out_dir / "plots").exists()
    assert not (out_dir / "counts").exists()
    assert not (out_dir / "curves").exists()
    assert not (out_dir / "aligned_curves").exists()

    results_files = list(out_dir.glob("results_*.json"))
    assert results_files, "Expected a sanitised results manifest"
    results_path = results_files[0]
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

    zip_candidates = list(out_dir.glob("before_after_alignment_*.zip"))
    assert zip_candidates, "Expected sanitised combined zip export"
    zip_path = zip_candidates[0]

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


def test_group_marker_exports_multiple_ridges(tmp_path):
    expr = pd.DataFrame(
        {
            "CD3": [1.0, 1.2, 1.1, 4.0, 4.2, 4.1],
            "CD19": [3.0, 3.3, 3.1, 6.5, 6.7, 6.6],
        }
    )
    meta = pd.DataFrame(
        {
            "sample": ["S1"] * 3 + ["S2"] * 3,
        }
    )

    expr_path = tmp_path / "expr.csv"
    meta_path = tmp_path / "meta.csv"
    expr.to_csv(expr_path, index=False)
    meta.to_csv(meta_path, index=False)

    options = BatchOptions(apply_arcsinh=False, align=True, max_peaks=2, group_by_marker=True)
    samples, meta_info = collect_dataset_samples(expr_path, meta_path, options)

    batch = run_batch(samples, options)
    assert not batch.interrupted
    assert batch.group_by_marker is True

    out_dir = tmp_path / "grouped_outputs"
    save_outputs(batch, out_dir, run_metadata=meta_info)

    zip_candidates = list(out_dir.glob("before_after_alignment_*.zip"))
    assert zip_candidates
    zip_path = zip_candidates[0]

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "before_alignment/before_alignment_ridge_CD19.png" in names
        assert "before_alignment/before_alignment_ridge_CD3.png" in names
        assert "after_alignment/aligned_ridge_CD19.png" in names
        assert "after_alignment/aligned_ridge_CD3.png" in names
        assert "before_alignment/before_alignment_ridge.png" not in names
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

    summary_candidates = list(out_dir.glob("summary_*.csv"))
    assert summary_candidates
    summary_path = summary_candidates[0]
    results_candidates = list(out_dir.glob("results_*.json"))
    assert results_candidates
    results_path = results_candidates[0]
    with results_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest.get("interrupted") is True
    assert manifest["samples"], "Partial samples should still be exported"


def test_run_batch_retries_timed_out_workers(monkeypatch):
    import peak_valley.batch as batch_mod

    options = BatchOptions(apply_arcsinh=False, workers=2, worker_timeout=0.1, worker_retries=1)
    sample_a = SampleInput(
        stem="SampleA",
        counts=np.array([0.1, 0.9]),
        metadata={"sample": "S1", "marker": "CD3"},
        arcsinh_signature=options.arcsinh_signature(),
    )
    sample_b = SampleInput(
        stem="SampleB",
        counts=np.array([0.2, 0.8]),
        metadata={"sample": "S2", "marker": "CD19"},
        arcsinh_signature=options.arcsinh_signature(),
    )

    call_counts: dict[str, int] = {}

    def fake_process(sample, opts, overrides, gpt_client):
        call_counts[sample.stem] = call_counts.get(sample.stem, 0) + 1
        if sample.stem == "SampleA" and call_counts[sample.stem] == 1:
            time.sleep(0.2)
        return SampleResult(
            stem=sample.stem,
            peaks=[0.5],
            valleys=[0.2],
            xs=np.array([0.0, 1.0]),
            ys=np.array([0.4, 0.6]),
            counts=sample.counts,
            params={"bw": 0.1, "prom": 0.05},
            quality=0.95,
            metadata=sample.metadata,
            source_name=sample.source_name,
            arcsinh_signature=sample.arcsinh_signature,
        )

    monkeypatch.setattr(batch_mod, "process_sample", fake_process)

    batch = run_batch([sample_a, sample_b], options)

    assert not batch.interrupted
    assert len(batch.samples) == 2
    assert call_counts["SampleA"] >= 2


def test_run_batch_reports_exceeded_retries(monkeypatch, capsys):
    import peak_valley.batch as batch_mod

    options = BatchOptions(
        apply_arcsinh=False,
        workers=2,
        worker_timeout=0.05,
        worker_retries=1,
    )
    slow_sample = SampleInput(
        stem="SlowOne",
        counts=np.array([0.1, 0.9]),
        metadata={"sample": "S1"},
        arcsinh_signature=options.arcsinh_signature(),
        order=0,
    )
    fast_sample = SampleInput(
        stem="FastOne",
        counts=np.array([0.2, 0.8]),
        metadata={"sample": "S2"},
        arcsinh_signature=options.arcsinh_signature(),
        order=1,
    )

    attempts: dict[str, int] = {}

    def fake_process(sample, opts, overrides, gpt_client):
        attempts[sample.stem] = attempts.get(sample.stem, 0) + 1
        if sample.stem == "SlowOne":
            time.sleep(options.worker_timeout + 0.05)
        return SampleResult(
            stem=sample.stem,
            peaks=[],
            valleys=[],
            xs=np.array([]),
            ys=np.array([]),
            counts=sample.counts,
            params={},
            quality=0.0,
            metadata=sample.metadata,
            source_name=sample.source_name,
            arcsinh_signature=sample.arcsinh_signature,
        )

    monkeypatch.setattr(batch_mod, "process_sample", fake_process)

    batch = run_batch([slow_sample, fast_sample], options)

    err = capsys.readouterr().err
    assert "giving up" in err
    assert batch.interrupted is True
    assert batch.failed_samples == ["SlowOne"]
    assert [res.stem for res in batch.samples] == ["FastOne"]
    assert attempts["SlowOne"] == options.worker_retries + 1


def test_alignment_normalizes_landmarks(monkeypatch):
    import peak_valley.batch as batch_mod

    def fake_kde(
        counts,
        n_peaks,
        prominence,
        bandwidth,
        min_width,
        grid_size,
        drop_frac,
        min_x_sep,
        curvature_thresh,
        turning_peak,
        first_valley,
    ):
        xs = np.linspace(-4.0, 4.0, 81)
        gauss_left = np.exp(-0.5 * ((xs + 3.0) / 0.4) ** 2)
        gauss_right = np.exp(-0.5 * ((xs - 3.0) / 0.4) ** 2)
        ys = gauss_left + gauss_right
        return np.array([-3.0, 3.0]), np.array([0.0]), xs, ys

    monkeypatch.setattr(batch_mod, "kde_peaks_valleys", fake_kde)

    options = BatchOptions(
        apply_arcsinh=False,
        align=True,
        target_landmarks=[-2.0, 0.0, 2.0],
    )

    sample = SampleInput(
        stem="Sample1",
        counts=np.linspace(-4.0, 4.0, 81),
        metadata={"sample": "S1", "marker": "CD3"},
        arcsinh_signature=options.arcsinh_signature(),
    )

    batch = run_batch([sample], options)
    assert not batch.interrupted
    assert batch.aligned_landmarks is not None

    result = batch.samples[0]
    assert result.aligned_counts is not None
    assert result.aligned_peaks is not None
    assert result.aligned_valleys is not None
    assert result.aligned_landmark_positions is not None

    warp = build_warp_function(np.array([-3.0, 0.0, 3.0]), np.array([-2.0, 0.0, 2.0]))
    expected_counts = warp(np.asarray(sample.counts, float))

    np.testing.assert_allclose(result.aligned_counts, expected_counts, atol=1e-9)
    np.testing.assert_allclose(result.aligned_peaks, np.array([-2.0, 2.0]), atol=1e-9)
    np.testing.assert_allclose(result.aligned_valleys, np.array([0.0]), atol=1e-9)
    np.testing.assert_allclose(
        result.aligned_landmark_positions,
        np.array([-2.0, 0.0, 2.0]),
        atol=1e-9,
    )
    np.testing.assert_allclose(batch.aligned_landmarks[0], [-2.0, 0.0, 2.0], atol=1e-9)


def _dummy_result(marker: str, sample: str) -> SampleResult:
    return SampleResult(
        stem=f"{sample}_{marker}",
        peaks=[0.5],
        valleys=[0.2],
        xs=np.array([0.0, 1.0]),
        ys=np.array([0.4, 0.6]),
        counts=np.array([0.1, 0.9]),
        params={"bw": 0.1, "prom": 0.05},
        quality=0.95,
        metadata={"sample": sample, "marker": marker},
    )


def test_save_outputs_uses_marker_slug(tmp_path):
    batch = BatchResults(samples=[_dummy_result("CD3", "S1")])
    save_outputs(batch, tmp_path)

    summary_files = {p.name for p in tmp_path.glob("summary_*.csv")}
    results_files = {p.name for p in tmp_path.glob("results_*.json")}

    assert "summary_CD3.csv" in summary_files
    assert "results_CD3.json" in results_files


def test_save_outputs_dedupes_marker_slugs(tmp_path):
    batch = BatchResults(
        samples=[
            _dummy_result("CD3", "S1"),
            _dummy_result("CD19", "S2"),
            _dummy_result("CD45", "S3"),
        ]
    )

    save_outputs(batch, tmp_path)
    first_summary = tmp_path / "summary_CD3-p2.csv"
    assert first_summary.exists()

    save_outputs(batch, tmp_path)
    second_summary = tmp_path / "summary_CD3_CD19-p1.csv"
    assert second_summary.exists()

    first_results = tmp_path / "results_CD3-p2.json"
    second_results = tmp_path / "results_CD3_CD19-p1.json"
    assert first_results.exists()
    assert second_results.exists()
