"""Batch-processing utilities for the Peak & Valley detector.

This module powers the command-line workflow which mirrors the Streamlit
experience: users may analyse multiple samples in bulk, optionally leverage
GPT-assisted parameter suggestions, enforce marker consistency, align
distributions, and export all intermediate and final artefacts.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .alignment import align_distributions
from .consistency import enforce_marker_consistency
from .data_io import arcsinh_transform, load_combined_csv, read_counts
from .gpt_adapter import (
    ask_gpt_bandwidth,
    ask_gpt_peak_count,
    ask_gpt_prominence,
)
from .kde_detector import kde_peaks_valleys, quick_peak_estimate
from .quality import stain_quality

try:  # optional dependency during tests
    from openai import OpenAI
except Exception:  # pragma: no cover - OpenAI not available in tests
    OpenAI = None  # type: ignore


@dataclass
class SampleInput:
    """Description of one sample to analyse."""

    stem: str
    counts: np.ndarray
    metadata: dict[str, Any]
    arcsinh_signature: tuple[bool, float, float, float]
    source_name: str | None = None
    order: int = 0
    cell_indices: Optional[np.ndarray] = None


@dataclass
class SampleResult:
    """Processed information for a sample."""

    stem: str
    peaks: list[float]
    valleys: list[float]
    xs: np.ndarray
    ys: np.ndarray
    counts: np.ndarray
    params: dict[str, Any]
    quality: float
    metadata: dict[str, Any]
    source_name: str | None = None
    arcsinh_signature: tuple[bool, float, float, float] = (True, 1.0, 0.2, 0.0)
    aligned_counts: Optional[np.ndarray] = None
    aligned_density: Optional[tuple[np.ndarray, np.ndarray]] = None
    cell_indices: Optional[np.ndarray] = None


@dataclass
class BatchOptions:
    """Global configuration for a batch run."""

    apply_arcsinh: bool = True
    arcsinh_a: float = 1.0
    arcsinh_b: float = 1 / 5
    arcsinh_c: float = 0.0

    # detector configuration
    n_peaks: Optional[int] = None
    n_peaks_auto: bool = False
    max_peaks: int = 3
    bandwidth: str | float = "scott"
    bandwidth_auto: bool = False
    prominence: float = 0.05
    prominence_auto: bool = False
    min_width: int = 0
    curvature: float = 0.0001
    turning_points: bool = False
    min_separation: float = 0.7
    grid_size: int = 20_000
    valley_drop: float = 10.0  # percent of peak height
    first_valley: str = "slope"  # or "drop"

    apply_consistency: bool = True
    consistency_tol: float = 0.5

    align: bool = False
    align_mode: str = "negPeak_valley_posPeak"
    target_landmarks: Optional[Sequence[float]] = None

    workers: int = 1

    # GPT integration
    gpt_model: Optional[str] = None

    def arcsinh_signature(self) -> tuple[bool, float, float, float]:
        return (
            bool(self.apply_arcsinh),
            float(self.arcsinh_a),
            float(self.arcsinh_b),
            float(self.arcsinh_c),
        )


@dataclass
class BatchResults:
    """Return value for a batch run."""

    samples: list[SampleResult]
    aligned_landmarks: Optional[np.ndarray] = None
    alignment_mode: Optional[str] = None
    target_landmarks: Optional[Sequence[float]] = None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not math.isnan(value):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _bandwidth_config(value: Any, default: str | float, auto_flag: bool) -> tuple[str | float, bool]:
    """Return (bandwidth, use_gpt)."""

    if value is None:
        return default, auto_flag

    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in {"auto", "gpt"}:
            return default, True
        try:
            return float(stripped), False
        except ValueError:
            return stripped, False

    if isinstance(value, (int, float)):
        return float(value), False

    return default, auto_flag


def _boolean_override(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "on"}:
            return True
        if lowered in {"false", "no", "0", "off"}:
            return False
    return default


def _merge_overrides(
    stem: str,
    meta: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not overrides:
        return {}

    merged: dict[str, Any] = {}
    for bucket, key in (
        ("global", None),
        ("samples", meta.get("sample")),
        ("markers", meta.get("marker")),
        ("batches", meta.get("batch")),
        ("stems", stem),
    ):
        if bucket not in overrides:
            continue
        if key is None:
            entry = overrides.get(bucket)
        else:
            entry = overrides[bucket].get(key)
        if isinstance(entry, Mapping):
            merged.update({k: v for k, v in entry.items() if v is not None})

    return merged


def _resolve_parameters(
    options: BatchOptions,
    overrides: Mapping[str, Any],
    gpt_client: OpenAI | None,
    counts: np.ndarray,
    marker: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return detector parameters and a debug dictionary."""

    params: dict[str, Any] = {}
    debug: dict[str, Any] = {}

    # --- basic numeric overrides -----------------------------------------
    max_peaks = _coerce_int(overrides.get("max_peaks"))
    if max_peaks is None or max_peaks <= 0:
        max_peaks = options.max_peaks
    params["max_peaks"] = max(1, int(max_peaks))

    min_width = overrides.get("min_width")
    params["min_width"] = _coerce_int(min_width)
    if params["min_width"] is None:
        params["min_width"] = int(options.min_width)

    params["curvature"] = (
        _coerce_float(overrides.get("curvature"))
        if overrides.get("curvature") is not None
        else float(options.curvature)
    )

    params["turning_points"] = _boolean_override(
        overrides.get("turning_points"), options.turning_points
    )

    min_sep_val = _coerce_float(overrides.get("min_separation"))
    params["min_separation"] = (
        float(min_sep_val) if min_sep_val is not None else float(options.min_separation)
    )

    grid_val = _coerce_int(overrides.get("max_grid"))
    grid_use = grid_val if grid_val is not None else options.grid_size
    params["grid_size"] = max(4000, int(grid_use))

    drop_override = _coerce_float(overrides.get("valley_drop"))
    params["valley_drop"] = (
        float(drop_override) if drop_override is not None else float(options.valley_drop)
    )

    first_mode = overrides.get("first_valley")
    if isinstance(first_mode, str) and first_mode.strip().lower().startswith("valley"):
        params["first_valley"] = "drop"
    elif isinstance(first_mode, str) and first_mode.strip().lower().startswith("slope"):
        params["first_valley"] = "slope"
    else:
        params["first_valley"] = options.first_valley

    # --- prominence -------------------------------------------------------
    prom_override = overrides.get("prom")
    prom_default = options.prominence
    prom_val = _coerce_float(prom_override)
    prom_auto = options.prominence_auto
    if isinstance(prom_override, str) and prom_override.strip().lower() in {"auto", "gpt"}:
        prom_auto = True
    elif prom_override is not None and prom_val is None:
        # fallback for non-numeric strings – treat as default
        prom_val = prom_default
    elif prom_val is not None:
        prom_auto = False
    params["prominence"] = prom_default if prom_val is None else float(prom_val)
    params["prominence_auto"] = bool(prom_auto)

    # --- bandwidth --------------------------------------------------------
    bw_override = overrides.get("bw")
    bw_value, bw_auto = _bandwidth_config(
        bw_override,
        options.bandwidth,
        options.bandwidth_auto,
    )
    params["bandwidth"] = bw_value
    params["bandwidth_auto"] = bw_auto

    # --- n_peaks ----------------------------------------------------------
    n_override = overrides.get("n_peaks")
    n_auto = options.n_peaks_auto
    n_fixed = options.n_peaks

    if isinstance(n_override, str) and n_override.strip().lower() in {"auto", "gpt"}:
        n_auto = True
        n_fixed = None
    elif n_override is not None:
        coerced = _coerce_int(n_override)
        if coerced is not None:
            n_fixed = coerced
            n_auto = False

    if n_fixed is not None and n_fixed <= 0:
        n_fixed = None

    params["n_peaks"] = n_fixed
    params["n_peaks_auto"] = bool(n_auto)

    # --- GPT powered bits -------------------------------------------------
    if params["bandwidth_auto"] or params["prominence_auto"] or params["n_peaks_auto"]:
        if gpt_client is None:
            debug["gpt_warning"] = "GPT client unavailable; falling back to defaults"

    bw_use = params["bandwidth"]
    if params["bandwidth_auto"] and gpt_client is not None:
        expected = (
            params["n_peaks"]
            if params["n_peaks"] is not None
            else params["max_peaks"]
        )
        try:
            bw_use = ask_gpt_bandwidth(
                gpt_client,
                options.gpt_model or "gpt-4o-mini",
                counts,
                peak_amount=expected,
                default="scott",
            )
        except Exception as exc:  # pragma: no cover - depends on API
            debug["gpt_bandwidth_error"] = str(exc)
            bw_use = options.bandwidth
    params["bandwidth_effective"] = bw_use

    prom_use = params["prominence"]
    if params["prominence_auto"] and gpt_client is not None:
        try:
            prom_use = ask_gpt_prominence(
                gpt_client,
                options.gpt_model or "gpt-4o-mini",
                counts,
                default=prom_use,
            )
        except Exception as exc:  # pragma: no cover - depends on API
            debug["gpt_prominence_error"] = str(exc)
            prom_use = params["prominence"]
    params["prominence_effective"] = float(prom_use)

    n_use = params["n_peaks"]
    if params["n_peaks_auto"]:
        if gpt_client is not None:
            try:
                n_use = ask_gpt_peak_count(
                    gpt_client,
                    options.gpt_model or "gpt-4o-mini",
                    params["max_peaks"],
                    counts_full=counts,
                    marker_name=marker,
                )
            except Exception as exc:  # pragma: no cover - depends on API
                debug["gpt_peak_error"] = str(exc)
                n_use = None
        if n_use is None:
            n_est, confident = quick_peak_estimate(
                counts,
                params["prominence_effective"],
                bw_use,
                params["min_width"] or None,
                params["grid_size"],
            )
            n_use = n_est if confident else None

    if n_use is None:
        n_use = params["max_peaks"]
    else:
        n_use = min(int(n_use), params["max_peaks"])
    params["n_peaks_effective"] = int(max(1, n_use))

    return params, debug


def _postprocess_valleys(
    peaks: list[float],
    valleys: list[float],
    xs: np.ndarray,
    ys: np.ndarray,
    drop_frac: float,
) -> list[float]:
    if len(peaks) == 1 and not valleys:
        p = peaks[0]
        p_idx = int(np.searchsorted(xs, p))
        p_idx = min(max(p_idx, 0), len(xs) - 1)
        y_pk = ys[p_idx]
        mask = np.where(ys[p_idx:] < drop_frac * y_pk)[0]
        if mask.size:
            valleys = [float(xs[p_idx + mask[0]])]
    return valleys


def process_sample(
    sample: SampleInput,
    options: BatchOptions,
    overrides: Mapping[str, Any] | None,
    gpt_client: OpenAI | None = None,
) -> SampleResult:
    """Run the detector for a single sample."""

    counts = np.asarray(sample.counts, float).ravel()
    cell_idx = None
    if sample.cell_indices is not None:
        cell_idx = np.asarray(sample.cell_indices, dtype=int).ravel()
    finite_mask = np.isfinite(counts)
    counts = counts[finite_mask]
    if cell_idx is not None and cell_idx.shape[0] == finite_mask.shape[0]:
        cell_idx = cell_idx[finite_mask]
    if counts.size == 0:
        return SampleResult(
            stem=sample.stem,
            peaks=[],
            valleys=[],
            xs=np.array([]),
            ys=np.array([]),
            counts=counts,
            params={},
            quality=float("nan"),
            metadata=dict(sample.metadata),
            source_name=sample.source_name,
            arcsinh_signature=sample.arcsinh_signature,
            cell_indices=cell_idx,
        )

    merged_overrides = _merge_overrides(sample.stem, sample.metadata, overrides)
    params, debug = _resolve_parameters(
        options,
        merged_overrides,
        gpt_client,
        counts,
        sample.metadata.get("marker"),
    )

    min_width = params["min_width"] if params["min_width"] else None
    curvature = params["curvature"]
    if curvature is not None and curvature <= 0:
        curvature = None

    drop_frac = params["valley_drop"] / 100.0

    peaks, valleys, xs, ys = kde_peaks_valleys(
        counts,
        params["n_peaks_effective"],
        params["prominence_effective"],
        params["bandwidth_effective"],
        min_width,
        params["grid_size"],
        drop_frac=drop_frac,
        min_x_sep=params["min_separation"],
        curvature_thresh=curvature,
        turning_peak=params["turning_points"],
        first_valley=params["first_valley"],
    )

    valleys = _postprocess_valleys(peaks, valleys, xs, ys, drop_frac)

    quality = float(stain_quality(counts, peaks, valleys))

    details = {
        "bw": params["bandwidth_effective"],
        "prom": params["prominence_effective"],
        "n_peaks": params["n_peaks_effective"],
        "max_peaks": params["max_peaks"],
        "min_width": params["min_width"],
        "curvature": curvature,
        "turning_points": params["turning_points"],
        "min_separation": params["min_separation"],
        "grid_size": params["grid_size"],
        "valley_drop": params["valley_drop"],
        "first_valley": params["first_valley"],
    }
    if debug:
        details["debug"] = debug

    return SampleResult(
        stem=sample.stem,
        peaks=list(map(float, peaks)),
        valleys=list(map(float, valleys)),
        xs=np.asarray(xs, float),
        ys=np.asarray(ys, float),
        counts=counts,
        params=details,
        quality=quality,
        metadata=dict(sample.metadata),
        source_name=sample.source_name,
        arcsinh_signature=sample.arcsinh_signature,
        cell_indices=cell_idx,
    )


def run_batch(
    samples: Iterable[SampleInput],
    options: BatchOptions,
    overrides: Mapping[str, Any] | None = None,
    gpt_client: OpenAI | None = None,
) -> BatchResults:
    """Process all samples and optionally align them."""

    ordered = sorted(samples, key=lambda s: s.order)
    order_map = {s.stem: idx for idx, s in enumerate(ordered)}
    results: list[SampleResult] = []

    if options.workers and options.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=options.workers) as pool:
            future_map = {
                pool.submit(process_sample, sample, options, overrides, gpt_client): sample
                for sample in ordered
            }
            for future in as_completed(future_map):
                res = future.result()
                results.append(res)
    else:
        for sample in ordered:
            results.append(process_sample(sample, options, overrides, gpt_client))

    results.sort(key=lambda r: order_map.get(r.stem, 0))

    if options.apply_consistency and len(results) > 1:
        info_map = {
            r.stem: {
                "peaks": list(r.peaks),
                "valleys": list(r.valleys),
                "xs": r.xs.tolist(),
                "ys": r.ys.tolist(),
                "marker": r.metadata.get("marker"),
            }
            for r in results
        }
        enforce_marker_consistency(info_map, tol=options.consistency_tol)
        for res in results:
            info = info_map.get(res.stem)
            if not info:
                continue
            res.peaks = list(map(float, info.get("peaks", [])))
            res.valleys = list(map(float, info.get("valleys", [])))
            res.quality = float(stain_quality(res.counts, res.peaks, res.valleys))

    aligned_landmarks: Optional[np.ndarray] = None

    if options.align and results:
        counts_list = [r.counts for r in results]
        peaks_list = [r.peaks for r in results]
        valleys_list = [r.valleys for r in results]
        density = [(r.xs, r.ys) for r in results]

        alignment = align_distributions(
            counts_list,
            peaks_list,
            valleys_list,
            align_type=options.align_mode,
            target_landmark=options.target_landmarks,
            density_grids=density,
        )

        aligned_counts, aligned_landmarks, _warp_funs, warped_density = alignment

        for res, counts_aligned, warped in zip(
            results, aligned_counts, warped_density  # type: ignore[misc]
        ):
            res.aligned_counts = np.asarray(counts_aligned, float)
            if warped is not None:
                xs_w, ys_w = warped
                res.aligned_density = (np.asarray(xs_w, float), np.asarray(ys_w, float))

    return BatchResults(
        samples=results,
        aligned_landmarks=aligned_landmarks,
        alignment_mode=options.align_mode if options.align else None,
        target_landmarks=options.target_landmarks,
    )


# ---------------------------------------------------------------------------
# Sample collection helpers

def _ensure_iterable(value: Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    items = [str(v) for v in value if str(v)]
    return items if items else None


def collect_counts_files(
    paths: Sequence[str | Path],
    options: BatchOptions,
    *,
    header_row: int = -1,
    skip_rows: int = 0,
) -> list[SampleInput]:
    """Create :class:`SampleInput` entries from raw counts CSV files."""

    samples: list[SampleInput] = []
    arcsinh_sig = options.arcsinh_signature()

    for order, path in enumerate(paths):
        path_obj = Path(path)
        counts, meta = read_counts(path_obj, header_row, skip_rows)
        if options.apply_arcsinh and not meta.get("arcsinh", False):
            counts = arcsinh_transform(
                counts,
                a=options.arcsinh_a,
                b=options.arcsinh_b,
                c=options.arcsinh_c,
            )

        metadata = {
            "sample": meta.get("sample") or path_obj.stem,
            "marker": meta.get("marker") or meta.get("protein_name"),
            "batch": meta.get("batch"),
            "protein_name": meta.get("protein_name"),
        }

        samples.append(
            SampleInput(
                stem=path_obj.stem,
                counts=np.asarray(counts, float),
                metadata=metadata,
                arcsinh_signature=arcsinh_sig,
                source_name=str(path_obj),
                order=order,
            )
        )

    return samples


def collect_dataset_samples(
    expression_file: str | Path,
    metadata_file: str | Path,
    options: BatchOptions,
    *,
    markers: Sequence[str] | None = None,
    samples_filter: Sequence[str] | None = None,
    batches: Sequence[str | None] | None = None,
) -> tuple[list[SampleInput], dict[str, Any]]:
    """Prepare samples from an expression + metadata dataset."""

    expr_df, expr_sources = load_combined_csv(expression_file, low_memory=False)
    meta_df, meta_sources = load_combined_csv(metadata_file, low_memory=False)

    if "sample" not in meta_df.columns:
        raise ValueError("Metadata CSV must contain a 'sample' column")

    available_markers = [c for c in expr_df.columns if c not in meta_df.columns]
    markers_sel = markers if markers else available_markers

    sample_values = meta_df["sample"].astype(str).tolist()
    sample_sel = samples_filter if samples_filter else sorted(set(sample_values))

    batch_column = "batch" if "batch" in meta_df.columns else None
    if batches is not None:
        batch_sel = [None if (isinstance(b, float) and math.isnan(b)) else b for b in batches]
    else:
        batch_sel = None

    arcsinh_sig = options.arcsinh_signature()

    prepared: list[SampleInput] = []

    order = 0
    for sample_name in sample_sel:
        mask = meta_df["sample"].astype(str).eq(str(sample_name))
        if not mask.any():
            continue

        if batch_column:
            raw_batches = meta_df.loc[mask, batch_column]
            batches_here = []
            for val in raw_batches.unique():
                clean = None if (pd.isna(val)) else val
                if batch_sel is not None and clean not in batch_sel:
                    continue
                batches_here.append(clean)
            if not batches_here:
                continue
        else:
            batches_here = [None]

        for batch_value in batches_here:
            if batch_column:
                batch_mask = meta_df[batch_column].eq(batch_value)
                if batch_value is None:
                    batch_mask = meta_df[batch_column].isna()
                cell_idx = meta_df.index[mask & batch_mask].tolist()
            else:
                cell_idx = meta_df.index[mask].tolist()

            if not cell_idx:
                continue

            for marker in markers_sel:
                if marker not in expr_df.columns:
                    continue
                values = expr_df.loc[cell_idx, marker].to_numpy(dtype=float, copy=True)
                if options.apply_arcsinh:
                    values = arcsinh_transform(
                        values,
                        a=options.arcsinh_a,
                        b=options.arcsinh_b,
                        c=options.arcsinh_c,
                    )

                stem_parts = [str(sample_name)]
                if batch_value is not None:
                    stem_parts.append(str(batch_value))
                stem_parts.append(str(marker))
                stem_parts.append("raw_counts")
                stem = "_".join(stem_parts)

                metadata = {
                    "sample": sample_name,
                    "marker": marker,
                    "batch": batch_value,
                }

                prepared.append(
                    SampleInput(
                        stem=stem,
                        counts=np.asarray(values, float),
                        metadata=metadata,
                        arcsinh_signature=arcsinh_sig,
                        source_name=str(expression_file),
                        order=order,
                        cell_indices=np.asarray(cell_idx, dtype=int),
                    )
                )
                order += 1

    run_meta = {
        "expression_sources": expr_sources,
        "metadata_sources": meta_sources,
        "expression_path": str(expression_file),
        "metadata_path": str(metadata_file),
    }

    return prepared, run_meta


def _jsonify(value: Any) -> Any:
    """Convert numpy/pandas objects into JSON-serialisable Python values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, MappingABC):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    return value


def results_to_dict(batch: BatchResults) -> dict[str, Any]:
    """Convert results into a JSON-serialisable structure."""

    payload: dict[str, Any] = {
        "samples": [],
    }

    if batch.aligned_landmarks is not None:
        payload["aligned_landmarks"] = _jsonify(batch.aligned_landmarks)
    if batch.alignment_mode:
        payload["alignment_mode"] = batch.alignment_mode
    if batch.target_landmarks is not None:
        payload["target_landmarks"] = _jsonify(batch.target_landmarks)

    for res in batch.samples:
        sample_payload = {
            "stem": res.stem,
            "peaks": [float(p) for p in res.peaks],
            "valleys": [float(v) for v in res.valleys],
            "quality": float(res.quality),
            "metadata": _jsonify(res.metadata),
            "params": _jsonify(res.params),
            "arcsinh": {
                "applied": bool(res.arcsinh_signature[0]),
                "a": float(res.arcsinh_signature[1]),
                "b": float(res.arcsinh_signature[2]),
                "c": float(res.arcsinh_signature[3]),
            },
            "source_name": res.source_name,
        }
        if res.aligned_counts is not None:
            sample_payload["aligned_counts"] = [
                float(x) if math.isfinite(float(x)) else None
                for x in res.aligned_counts
            ]
        payload["samples"].append(sample_payload)

    return payload


def export_summary(batch: BatchResults) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for res in batch.samples:
        meta = dict(res.metadata)
        row = {
            "stem": res.stem,
            "sample": meta.get("sample"),
            "marker": meta.get("marker"),
            "batch": meta.get("batch"),
            "n_peaks": len(res.peaks),
            "peaks": "; ".join(f"{p:.5g}" for p in res.peaks),
            "valleys": "; ".join(f"{v:.5g}" for v in res.valleys),
            "quality": res.quality,
        }
        row.update({
            "bandwidth": res.params.get("bw"),
            "prominence": res.params.get("prom"),
        })
        rows.append(row)

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["stem", "sample", "marker", "batch", "n_peaks", "peaks", "valleys", "quality"])


def counts_to_string(array: np.ndarray | None) -> str:
    if array is None or array.size == 0:
        return "[]"
    values: list[str] = []
    for val in np.asarray(array).ravel():
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            values.append("null")
        else:
            values.append(format(float(val), ".15g"))
    return "[" + "; ".join(values) + "]"


def save_outputs(
    batch: BatchResults,
    output_dir: str | Path,
    *,
    run_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist summary files, per-sample exports and optional alignment."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plots_dir = out / "plots"
    curves_dir = out / "curves"
    counts_dir = out / "counts"
    aligned_dir = out / "aligned_curves"
    plots_dir.mkdir(exist_ok=True)
    curves_dir.mkdir(exist_ok=True)
    counts_dir.mkdir(exist_ok=True)
    if batch.aligned_landmarks is not None:
        aligned_dir.mkdir(exist_ok=True)

    import matplotlib.pyplot as plt

    summary_df = export_summary(batch)
    summary_df.to_csv(out / "summary.csv", index=False)

    processed_rows: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []

    for res in batch.samples:
        stem = res.stem
        xs = res.xs
        ys = res.ys

        df_curve = pd.DataFrame({"x": xs, "density": ys})
        df_curve.to_csv(curves_dir / f"{stem}.csv", index=False)

        df_counts = pd.DataFrame({"normalized_counts": res.counts})
        df_counts.to_csv(counts_dir / f"{stem}.csv", index=False)

        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
        ax.plot(xs, ys, color="tab:blue")
        ax.fill_between(xs, 0, ys, color="tab:blue", alpha=0.2)
        for p in res.peaks:
            ax.axvline(p, color="tab:red", linestyle="--", linewidth=1)
        for v in res.valleys:
            ax.axvline(v, color="tab:green", linestyle=":", linewidth=1)
        ax.set_title(stem)
        ax.set_xlabel("Arcsinh counts")
        ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{stem}.png")
        plt.close(fig)

        meta = res.metadata
        processed_rows.append(
            {
                "stem": stem,
                "sample": meta.get("sample"),
                "marker": meta.get("marker"),
                "batch": meta.get("batch"),
                "normalized_counts": counts_to_string(res.counts),
            }
        )

        if res.aligned_counts is not None:
            aligned_rows.append(
                {
                    "stem": stem,
                    "sample": meta.get("sample"),
                    "marker": meta.get("marker"),
                    "batch": meta.get("batch"),
                    "aligned_normalized_counts": counts_to_string(res.aligned_counts),
                }
            )
            if res.aligned_density is not None:
                xs_a, ys_a = res.aligned_density
                pd.DataFrame({"x": xs_a, "density": ys_a}).to_csv(
                    aligned_dir / f"{stem}.csv",
                    index=False,
                )

    pd.DataFrame(processed_rows).to_csv(out / "processed_counts.csv", index=False)
    if aligned_rows:
        pd.DataFrame(aligned_rows).to_csv(out / "aligned_counts.csv", index=False)
    if batch.aligned_landmarks is not None:
        pd.DataFrame(batch.aligned_landmarks).to_csv(
            out / "aligned_landmarks.csv", index=False, header=False
        )

    manifest = results_to_dict(batch)
    if run_metadata:
        manifest["run_metadata"] = _jsonify(run_metadata)
    with open(out / "results.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    _write_combined_zip(
        batch,
        out,
        run_metadata=run_metadata,
    )


def _write_combined_zip(
    batch: BatchResults,
    output_dir: Path,
    *,
    run_metadata: Mapping[str, Any] | None,
) -> None:
    exports = _dataset_exports(batch, run_metadata)
    if exports is None:
        return

    before_meta = exports["meta"]
    before_expr = exports["expr_before"]
    if before_meta is None and before_expr is None and exports["raw_ridge"] is None:
        return

    zip_path = output_dir / "before_after_alignment.zip"
    has_after = exports["expr_after"] is not None or exports["aligned_ridge"] is not None

    with zipfile.ZipFile(zip_path, "w") as archive:
        if before_meta is not None:
            archive.writestr("before_alignment/cell_metadata_combined.csv", before_meta)
        if before_expr is not None:
            archive.writestr("before_alignment/expression_matrix_combined.csv", before_expr)
        if exports["raw_ridge"] is not None:
            archive.writestr("before_alignment/before_alignment_ridge.png", exports["raw_ridge"])
        if exports["aligned_ridge"] is not None:
            archive.writestr("after_alignment/aligned_ridge.png", exports["aligned_ridge"])
        if has_after and exports["meta_after"] is not None:
            archive.writestr("after_alignment/cell_metadata_combined.csv", exports["meta_after"])
        if exports["expr_after"] is not None:
            archive.writestr("after_alignment/expression_matrix_aligned.csv", exports["expr_after"])


def _dataset_exports(
    batch: BatchResults,
    run_metadata: Mapping[str, Any] | None,
) -> dict[str, bytes | None] | None:
    if not run_metadata:
        return None

    expr_path = run_metadata.get("expression_path")
    meta_path = run_metadata.get("metadata_path")
    if not expr_path or not meta_path:
        return None

    try:
        expr_df, _ = load_combined_csv(expr_path, low_memory=False)
        meta_df, _ = load_combined_csv(meta_path, low_memory=False)
    except Exception:
        return None

    dataset_results = [res for res in batch.samples if res.cell_indices is not None]
    if not dataset_results:
        return None

    union_indices: list[int] = []
    seen_indices: set[int] = set()
    markers_order: list[str] = []
    for res in dataset_results:
        marker = res.metadata.get("marker")
        if marker and marker not in markers_order:
            markers_order.append(str(marker))
        idx_array = np.asarray(res.cell_indices, dtype=int)
        for idx in idx_array.tolist():
            if idx not in seen_indices:
                seen_indices.add(idx)
                union_indices.append(idx)

    if not union_indices:
        return None

    meta_subset = meta_df.loc[union_indices].copy()
    meta_csv = meta_subset.reset_index(drop=True).to_csv(index=False).encode()

    markers_available = [c for c in expr_df.columns if c not in meta_df.columns]
    ordered_markers = [m for m in markers_available if m in markers_order]
    for marker in markers_order:
        if marker not in ordered_markers:
            ordered_markers.append(marker)

    expr_before = pd.DataFrame(index=union_indices, columns=ordered_markers, dtype=float)
    expr_after = pd.DataFrame(index=union_indices, columns=ordered_markers, dtype=float)
    has_aligned = any(res.aligned_counts is not None for res in dataset_results)

    for res in dataset_results:
        marker = res.metadata.get("marker")
        if marker is None:
            continue
        marker = str(marker)
        if marker not in expr_before.columns:
            expr_before[marker] = np.nan
            expr_after[marker] = np.nan
        idx_array = np.asarray(res.cell_indices, dtype=int)
        expr_before.loc[idx_array, marker] = np.asarray(res.counts, float)
        values_after = (
            np.asarray(res.aligned_counts, float)
            if res.aligned_counts is not None
            else np.asarray(res.counts, float)
        )
        expr_after.loc[idx_array, marker] = values_after

    expr_before = expr_before.loc[union_indices]
    expr_after = expr_after.loc[union_indices]

    if "cell_id" in expr_df.columns:
        cell_ids = expr_df.loc[union_indices, "cell_id"].to_numpy()
        expr_before.insert(0, "cell_id", cell_ids)
        expr_after.insert(0, "cell_id", cell_ids)

    expr_before_csv = expr_before.reset_index(drop=True).to_csv(index=False).encode()
    expr_after_csv = expr_after.reset_index(drop=True).to_csv(index=False).encode()
    if not has_aligned:
        expr_after_csv = None

    raw_ridge = _ridge_plot_png(batch.samples, aligned=False)
    aligned_ridge = _ridge_plot_png(batch.samples, aligned=True) if has_aligned else None

    return {
        "meta": meta_csv,
        "meta_after": meta_csv,
        "expr_before": expr_before_csv,
        "expr_after": expr_after_csv,
        "raw_ridge": raw_ridge,
        "aligned_ridge": aligned_ridge,
    }


def _ridge_plot_png(samples: Sequence[SampleResult], *, aligned: bool) -> bytes | None:
    curves: list[tuple[str, np.ndarray, np.ndarray, list[float], list[float]]] = []
    for res in samples:
        xs: np.ndarray
        ys: np.ndarray
        if aligned:
            if res.aligned_density is None:
                continue
            xs = np.asarray(res.aligned_density[0], float)
            ys = np.asarray(res.aligned_density[1], float)
        else:
            xs = np.asarray(res.xs, float)
            ys = np.asarray(res.ys, float)
        if xs.size == 0 or ys.size == 0:
            continue
        curves.append((res.stem, xs, ys, list(res.peaks), list(res.valleys)))

    if not curves:
        return None

    x_min = min(float(xs.min()) for _, xs, _, _, _ in curves)
    x_max = max(float(xs.max()) for _, xs, _, _, _ in curves)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return None

    if x_max == x_min:
        pad = 0.05 * (abs(x_min) if x_min != 0 else 1.0)
    else:
        pad = 0.05 * (x_max - x_min)

    y_max = max(float(ys.max()) for _, _, ys, _, _ in curves)
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = 1.0
    gap = 1.2 * y_max

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 0.8 * max(len(curves), 1)), dpi=150, sharex=True)

    for idx, (stem, xs, ys, peaks, valleys) in enumerate(curves):
        offset = idx * gap
        ax.plot(xs, ys + offset, color="black", lw=1)
        ax.fill_between(xs, offset, ys + offset, color="#FFA50088", lw=0)
        ymax_local = max(float(ys.max()), 1e-6)
        for p in peaks:
            try:
                peak_val = float(p)
            except (TypeError, ValueError):
                continue
            if np.isfinite(peak_val):
                ax.vlines(peak_val, offset, offset + ymax_local, color="black", lw=0.8)
        for v in valleys:
            try:
                valley_val = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(valley_val):
                ax.vlines(
                    valley_val,
                    offset,
                    offset + ymax_local,
                    color="grey",
                    lw=0.8,
                    linestyles=":",
                )
        ax.text(
            x_min,
            offset + 0.5 * y_max,
            stem,
            ha="right",
            va="center",
            fontsize=7,
        )

    ax.set_yticks([])
    ax.set_xlim(x_min - pad, x_max + pad)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()

