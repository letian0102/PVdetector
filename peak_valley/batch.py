"""Batch-processing utilities for the Peak & Valley detector.

This module powers the command-line workflow which mirrors the Streamlit
experience: users may analyse multiple samples in bulk, optionally leverage
GPT-assisted parameter suggestions, enforce marker consistency, align
distributions, and export all intermediate and final artefacts.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path
import signal
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

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
from scipy.stats import gaussian_kde

from .kde_detector import (
    _mostly_small_discrete,
    _normalise_bandwidth,
    kde_peaks_valleys,
    quick_peak_estimate,
)
from .quality import stain_quality
from .roughness import find_bw_for_roughness


DEFAULT_GPT_MODEL = "o4-mini"

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
    aligned_peaks: Optional[list[float]] = None
    aligned_valleys: Optional[list[float]] = None
    aligned_landmark_positions: Optional[np.ndarray] = None
    cell_indices: Optional[np.ndarray] = None


@dataclass
class BatchOptions:
    """Global configuration for a batch run."""

    apply_arcsinh: bool = True
    arcsinh_a: float = 1.0
    arcsinh_b: float = 1 / 5
    arcsinh_c: float = 0.0
    export_plots: bool = False

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
    min_separation: float = 0.5
    grid_size: int = 20_000
    valley_drop: float = 10.0  # percent of peak height
    first_valley: str = "slope"  # or "drop"

    apply_consistency: bool = False
    consistency_tol: float = 0.5

    sample_timeout: float = 10.0

    align: bool = False
    align_mode: str = "negPeak_valley_posPeak"
    target_landmarks: Optional[Sequence[float]] = None
    group_by_marker: bool = False

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
    interrupted: bool = False
    group_by_marker: bool = False
    failed_samples: list[str] = field(default_factory=list)


class BatchProgress(Protocol):
    """Callback interface for reporting batch progress."""

    def start(self, total: int) -> None:
        ...

    def advance(self, stem: str, completed: int, total: int) -> None:
        ...

    def finish(self, completed: int, total: int, interrupted: bool) -> None:
        ...

    def result(self, result: "SampleResult") -> None:
        ...


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
        max(0.0, float(min_sep_val))
        if min_sep_val is not None
        else max(0.0, float(options.min_separation))
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
                options.gpt_model or DEFAULT_GPT_MODEL,
                counts,
                peak_amount=expected,
                default="scott",
            )
        except Exception as exc:  # pragma: no cover - depends on API
            debug["gpt_bandwidth_error"] = str(exc)
            bw_use = options.bandwidth
    params["bandwidth_effective"] = bw_use

    if isinstance(params["bandwidth_effective"], str):
        bw_label = params["bandwidth_effective"].strip().lower()
        if bw_label == "roughness":
            try:
                params["bandwidth_effective"] = find_bw_for_roughness(counts)
                debug["bandwidth_method"] = "roughness"
            except Exception as exc:
                debug["roughness_error"] = str(exc)
                params["bandwidth_effective"] = options.bandwidth

        # Normalise any string bandwidth to something ``gaussian_kde`` accepts.
        bw_eff = params["bandwidth_effective"]
        if isinstance(bw_eff, str):
            try:
                params["bandwidth_effective"] = float(bw_eff)
            except ValueError:
                label = bw_eff.strip().lower()
                if label not in {"scott", "silverman"}:
                    debug["bandwidth_warning"] = (
                        f"Unsupported bandwidth '{bw_eff}', falling back to 'scott'"
                    )
                    params["bandwidth_effective"] = "scott"

    prom_use = params["prominence"]
    if params["prominence_auto"] and gpt_client is not None:
        try:
            prom_use = ask_gpt_prominence(
                gpt_client,
                options.gpt_model or DEFAULT_GPT_MODEL,
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
                    options.gpt_model or DEFAULT_GPT_MODEL,
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


def _effective_bandwidth(counts: np.ndarray, bw: str | float) -> float:
    """Return the scalar KDE bandwidth actually applied.

    Mirrors the logic inside :func:`kde_peaks_valleys` so presets such as
    ``"scott"``/``"silverman"`` resolve to the concrete numerical bandwidth
    used during estimation.
    """

    x = np.asarray(counts, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")

    x_kde = x
    if x_kde.size > 10_000:
        x_kde = np.random.choice(x_kde, 10_000, replace=False)

    bw_use = _normalise_bandwidth(bw)
    if bw_use == "roughness":
        try:
            bw_use = _normalise_bandwidth(find_bw_for_roughness(x_kde))
        except Exception:
            bw_use = "scott"

    try:
        kde = gaussian_kde(x_kde, bw_method=bw_use)
    except ValueError:
        bw_use = "scott"
        kde = gaussian_kde(x_kde, bw_method=bw_use)

    if _mostly_small_discrete(x_kde):
        kde.set_bandwidth(kde.factor * 4.0)

    if x_kde.size <= 1:
        return 0.0

    sample_std = float(np.sqrt(float(np.var(kde.dataset, ddof=1))))
    return float(kde.factor * sample_std)


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

    peaks, valleys, xs, ys, bw_scalar = kde_peaks_valleys(
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
        return_bandwidth=True,
    )

    valleys = _postprocess_valleys(peaks, valleys, xs, ys, drop_frac)

    quality = float(stain_quality(counts, peaks, valleys))

    details = {
        "bw": bw_scalar,
        "bw_label": params["bandwidth_effective"],
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
    progress: BatchProgress | None = None,
) -> BatchResults:
    """Process all samples and optionally align them."""

    def _run_sample_with_timeout(sample: SampleInput) -> SampleResult:
        timeout = float(options.sample_timeout)
        if timeout <= 0:
            return process_sample(sample, options, overrides, gpt_client)

        def _raise_timeout(signum, frame):  # pragma: no cover - invoked by signal
            raise TimeoutError(
                f"Sample '{sample.stem}' exceeded the {timeout:.3f}-second processing timeout"
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            return process_sample(sample, options, overrides, gpt_client)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    worker_count = options.workers if options.sample_timeout <= 0 else 1

    ordered = sorted(samples, key=lambda s: s.order)
    order_map = {s.stem: idx for idx, s in enumerate(ordered)}
    results: list[SampleResult] = []
    seen_stems: set[str] = set()
    total = len(ordered)
    completed = 0
    interrupted = False
    failed_samples: list[str] = []

    if progress is not None:
        try:
            progress.start(total)
        except Exception:
            progress = None

    def _append_result(res: SampleResult) -> None:
        nonlocal completed, progress
        if res.stem in seen_stems:
            return
        results.append(res)
        seen_stems.add(res.stem)
        completed += 1
        if progress is not None:
            try:
                progress.advance(res.stem, completed, total)
                result_cb = getattr(progress, "result", None)
                if callable(result_cb):
                    result_cb(res)
            except Exception:
                progress = None

    def _record_failure(stem: str) -> None:
        nonlocal completed, progress
        if stem in seen_stems or stem in failed_samples:
            return
        failed_samples.append(stem)
        completed += 1
        if progress is not None:
            try:
                progress.advance(stem, completed, total)
            except Exception:
                progress = None

    aligned_landmarks: Optional[np.ndarray] = None

    try:
        if worker_count and worker_count > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            error: BaseException | None = None
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                future_map = {
                    pool.submit(_run_sample_with_timeout, sample): sample
                    for sample in ordered
                }
                try:
                    for future in as_completed(future_map):
                        sample = future_map[future]
                        try:
                            res = future.result()
                        except KeyboardInterrupt:
                            interrupted = True
                            break
                        except TimeoutError:
                            _record_failure(sample.stem)
                            continue
                        except BaseException as exc:  # capture other errors to re-raise later
                            error = exc
                            interrupted = True
                            break
                        _append_result(res)
                except KeyboardInterrupt:
                    interrupted = True
                finally:
                    if interrupted:
                        for future, sample in future_map.items():
                            if future.done():
                                try:
                                    res = future.result()
                                except KeyboardInterrupt:
                                    continue
                                except BaseException:
                                    continue
                                _append_result(res)
                            else:
                                future.cancel()
            if error is not None:
                raise error
        else:
            for sample in ordered:
                if interrupted:
                    break
                try:
                    res = _run_sample_with_timeout(sample)
                except KeyboardInterrupt:
                    interrupted = True
                    break
                except TimeoutError:
                    _record_failure(sample.stem)
                    continue
                _append_result(res)

        results.sort(key=lambda r: order_map.get(r.stem, 0))

        if not interrupted and options.apply_consistency and len(results) > 1:
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
            try:
                enforce_marker_consistency(info_map, tol=options.consistency_tol)
            except KeyboardInterrupt:
                interrupted = True
            else:
                for res in results:
                    info = info_map.get(res.stem)
                    if not info:
                        continue
                    res.peaks = list(map(float, info.get("peaks", [])))
                    res.valleys = list(map(float, info.get("valleys", [])))
                    res.quality = float(stain_quality(res.counts, res.peaks, res.valleys))

        if not interrupted and options.align and results:
            if options.group_by_marker:
                grouped: dict[str, list[SampleResult]] = {}
                for res in results:
                    marker = res.metadata.get("marker")
                    key = str(marker) if marker else "Default"
                    grouped.setdefault(key, []).append(res)

                landmark_rows: dict[str, np.ndarray] = {}
                landmark_cols: int | None = None

                for group_results in grouped.values():
                    counts_list = [r.counts for r in group_results]
                    peaks_list = [r.peaks for r in group_results]
                    valleys_list = [r.valleys for r in group_results]
                    density = [(r.xs, r.ys) for r in group_results]

                    try:
                        alignment = align_distributions(
                            counts_list,
                            peaks_list,
                            valleys_list,
                            align_type=options.align_mode,
                            target_landmark=options.target_landmarks,
                            density_grids=density,
                        )
                    except KeyboardInterrupt:
                        interrupted = True
                        break
                    else:
                        (
                            aligned_counts_grp,
                            aligned_landmarks_grp,
                            warp_funs,
                            warped_density,
                        ) = alignment

                        for idx, (res, counts_aligned, warped) in enumerate(
                            zip(group_results, aligned_counts_grp, warped_density)  # type: ignore[misc]
                        ):
                            res.aligned_counts = np.asarray(counts_aligned, float)
                            if warped is not None:
                                xs_w, ys_w = warped
                                res.aligned_density = (
                                    np.asarray(xs_w, float),
                                    np.asarray(ys_w, float),
                                )
                            else:
                                res.aligned_density = None
                            warp_fn = warp_funs[idx] if idx < len(warp_funs) else None
                            if warp_fn is not None:
                                if res.peaks:
                                    peaks_arr = np.asarray(res.peaks, float)
                                    res.aligned_peaks = [float(v) for v in warp_fn(peaks_arr)]
                                else:
                                    res.aligned_peaks = []
                                if res.valleys:
                                    valleys_arr = np.asarray(res.valleys, float)
                                    res.aligned_valleys = [float(v) for v in warp_fn(valleys_arr)]
                                else:
                                    res.aligned_valleys = []
                            else:
                                res.aligned_peaks = []
                                res.aligned_valleys = []
                            if (
                                aligned_landmarks_grp is not None
                                and idx < len(aligned_landmarks_grp)
                            ):
                                row = np.asarray(aligned_landmarks_grp[idx], float)
                                res.aligned_landmark_positions = row
                                landmark_rows[res.stem] = row
                                if landmark_cols is None:
                                    landmark_cols = row.shape[0]

                if not interrupted and landmark_cols is not None:
                    aligned_landmarks = np.full((len(results), landmark_cols), np.nan, float)
                    for idx, res in enumerate(results):
                        row = landmark_rows.get(res.stem)
                        if row is not None and row.shape[0] == landmark_cols:
                            aligned_landmarks[idx] = row
            else:
                counts_list = [r.counts for r in results]
                peaks_list = [r.peaks for r in results]
                valleys_list = [r.valleys for r in results]
                density = [(r.xs, r.ys) for r in results]

                try:
                    alignment = align_distributions(
                        counts_list,
                        peaks_list,
                        valleys_list,
                        align_type=options.align_mode,
                        target_landmark=options.target_landmarks,
                        density_grids=density,
                    )
                except KeyboardInterrupt:
                    interrupted = True
                else:
                    aligned_counts_all, aligned_landmarks_all, warp_funs, warped_density = alignment

                    for idx, (res, counts_aligned, warped) in enumerate(
                        zip(results, aligned_counts_all, warped_density)  # type: ignore[misc]
                    ):
                        res.aligned_counts = np.asarray(counts_aligned, float)
                        if warped is not None:
                            xs_w, ys_w = warped
                            res.aligned_density = (
                                np.asarray(xs_w, float),
                                np.asarray(ys_w, float),
                            )
                        else:
                            res.aligned_density = None
                        warp_fn = warp_funs[idx] if idx < len(warp_funs) else None
                        if warp_fn is not None:
                            if res.peaks:
                                peaks_arr = np.asarray(res.peaks, float)
                                res.aligned_peaks = [float(v) for v in warp_fn(peaks_arr)]
                            else:
                                res.aligned_peaks = []
                            if res.valleys:
                                valleys_arr = np.asarray(res.valleys, float)
                                res.aligned_valleys = [float(v) for v in warp_fn(valleys_arr)]
                            else:
                                res.aligned_valleys = []
                        else:
                            res.aligned_peaks = []
                            res.aligned_valleys = []
                        if aligned_landmarks_all is not None and idx < len(aligned_landmarks_all):
                            res.aligned_landmark_positions = np.asarray(
                                aligned_landmarks_all[idx],
                                float,
                            )

                    aligned_landmarks = aligned_landmarks_all
    finally:
        if progress is not None:
            try:
                progress.finish(completed, total, interrupted)
            except Exception:
                pass

    return BatchResults(
        samples=results,
        aligned_landmarks=aligned_landmarks,
        alignment_mode=options.align_mode if (options.align and not interrupted) else None,
        target_landmarks=options.target_landmarks,
        interrupted=interrupted,
        group_by_marker=bool(options.group_by_marker),
        failed_samples=failed_samples,
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
    used_stems: set[str] = set()

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

        raw_stem = meta.get("stem") or path_obj.stem
        safe_stem = _unique_stem(raw_stem, used=used_stems)
        metadata.setdefault("source_stem", raw_stem)

        samples.append(
            SampleInput(
                stem=safe_stem,
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
    exclude_markers: Sequence[str] | None = None,
    exclude_samples: Sequence[str] | None = None,
    batches: Sequence[str | None] | None = None,
) -> tuple[list[SampleInput], dict[str, Any]]:
    """Prepare samples from an expression + metadata dataset."""

    expr_df, expr_sources = load_combined_csv(expression_file, low_memory=False)
    meta_df, meta_sources = load_combined_csv(metadata_file, low_memory=False)

    if "sample" not in meta_df.columns:
        raise ValueError("Metadata CSV must contain a 'sample' column")

    available_markers = [c for c in expr_df.columns if c not in meta_df.columns]
    markers_sel = markers if markers else available_markers
    if exclude_markers:
        marker_excludes = {str(m).lower() for m in exclude_markers}
        markers_sel = [m for m in markers_sel if str(m).lower() not in marker_excludes]

    sample_values = meta_df["sample"].astype(str).tolist()
    sample_sel = samples_filter if samples_filter else sorted(set(sample_values))
    if exclude_samples:
        sample_excludes = {str(s).lower() for s in exclude_samples}
        sample_sel = [s for s in sample_sel if str(s).lower() not in sample_excludes]

    batch_column = "batch" if "batch" in meta_df.columns else None
    if batches is not None:
        batch_sel = [None if (isinstance(b, float) and math.isnan(b)) else b for b in batches]
    else:
        batch_sel = None

    arcsinh_sig = options.arcsinh_signature()

    prepared: list[SampleInput] = []
    used_stems: set[str] = set()

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
                raw_stem = "_".join(stem_parts)
                stem = _unique_stem(raw_stem, used=used_stems)

                metadata = {
                    "sample": sample_name,
                    "marker": marker,
                    "batch": batch_value,
                    "source_stem": raw_stem,
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


def _sanitize_group_label(label: str | None) -> str:
    """Return a filesystem-safe label for ridge plot exports."""

    if label is None:
        return "group"

    text = str(label).strip()
    if not text:
        return "group"

    clean = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    clean = clean.strip("._-")
    return clean or "group"


def _sanitize_stem_value(value: str) -> str:
    """Return a filesystem-safe value suitable for sample stems."""

    text = str(value).strip()
    if not text:
        return "sample"

    clean = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    clean = re.sub(r"_{2,}", "_", clean)
    clean = clean.strip("._-")
    return clean or "sample"


def _unique_stem(raw_value: str, *, used: set[str]) -> str:
    """Sanitise ``raw_value`` and ensure the resulting stem is unique."""

    base = _sanitize_stem_value(raw_value)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _export_slug(
    batch: BatchResults,
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> str:
    """Build a concise, filesystem-safe slug describing this batch run."""

    out = Path(output_dir)

    markers: list[str] = []
    fallbacks: list[str] = []
    seen: set[str] = set()

    for res in batch.samples:
        meta = res.metadata

        marker = meta.get("marker")
        if marker not in (None, ""):
            safe_marker = _sanitize_stem_value(str(marker))
            if safe_marker and safe_marker not in seen:
                seen.add(safe_marker)
                markers.append(safe_marker)
                continue

        sample = meta.get("sample")
        if sample not in (None, ""):
            safe_sample = _sanitize_stem_value(str(sample))
            if safe_sample and safe_sample not in seen:
                seen.add(safe_sample)
                fallbacks.append(safe_sample)
                continue

        stem = _sanitize_stem_value(res.stem)
        if stem and stem not in seen:
            seen.add(stem)
            fallbacks.append(stem)

    descriptors = markers or fallbacks or ["batch"]

    def _candidates() -> Iterable[str]:
        base = descriptors[0]
        if len(descriptors) == 1:
            yield base
        else:
            second = descriptors[1]
            if len(descriptors) == 2:
                yield f"{base}-p1"
                yield f"{base}_{second}"
            else:
                yield f"{base}-p{len(descriptors) - 1}"
                yield f"{base}_{second}-p{len(descriptors) - 2}"
                yield f"{base}_{second}"
            yield base

        combined = "_".join(descriptors[:3])
        combined_safe = _sanitize_stem_value(combined)
        if combined_safe:
            yield combined_safe

        digest_source = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:6]
        digest_base = combined_safe or base or "batch"
        yield f"{digest_base}_{digest}"

    def _slug_available(slug: str) -> bool:
        targets = [
            out / f"summary_{slug}.csv",
            out / f"results_{slug}.json",
            out / f"before_after_alignment_{slug}.zip",
        ]
        return not any(target.exists() for target in targets)

    tried: set[str] = set()
    for candidate in _candidates():
        slug = _sanitize_stem_value(candidate)
        if not slug or slug in tried:
            continue
        tried.add(slug)
        if _slug_available(slug):
            return slug

    fallback_base = _sanitize_stem_value(descriptors[0]) or "batch"
    suffix = 2
    while True:
        slug = f"{fallback_base}-{suffix}"
        if slug not in tried and _slug_available(slug):
            return slug
        tried.add(slug)
        suffix += 1


def results_to_dict(batch: BatchResults) -> dict[str, Any]:
    """Convert results into a JSON-serialisable structure."""

    payload: dict[str, Any] = {
        "samples": [],
    }
    payload["group_by_marker"] = bool(batch.group_by_marker)

    payload["interrupted"] = bool(batch.interrupted)
    payload["failed_samples"] = list(batch.failed_samples)

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
        if res.aligned_peaks is not None:
            sample_payload["aligned_peaks"] = [float(p) for p in res.aligned_peaks]
        if res.aligned_valleys is not None:
            sample_payload["aligned_valleys"] = [float(v) for v in res.aligned_valleys]
        if res.aligned_landmark_positions is not None:
            sample_payload["aligned_landmarks"] = [
                float(x) if np.isfinite(float(x)) else None
                for x in np.asarray(res.aligned_landmark_positions, float)
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


def save_outputs(
    batch: BatchResults,
    output_dir: str | Path,
    *,
    run_metadata: Mapping[str, Any] | None = None,
    export_plots: bool = False,
) -> None:
    """Persist batch outputs mirroring the Streamlit downloads."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_df = export_summary(batch)

    error_file = out / "error_samples.txt"
    with error_file.open("w", encoding="utf-8") as fh:
        for stem in batch.failed_samples:
            fh.write(f"{stem}\n")

    manifest = results_to_dict(batch)
    if run_metadata:
        manifest["run_metadata"] = _jsonify(run_metadata)

    slug = _export_slug(batch, manifest, out)

    summary_name = f"summary_{slug}.csv"
    summary_df.to_csv(out / summary_name, index=False)

    if export_plots:
        plots_dir = out / "plots"
        plots_dir.mkdir(exist_ok=True)

        import matplotlib.pyplot as plt

        for res in batch.samples:
            xs = np.asarray(res.xs, float)
            ys = np.asarray(res.ys, float)
            if xs.size == 0 or ys.size == 0:
                continue

            fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
            ax.plot(xs, ys, color="tab:blue")
            ax.fill_between(xs, 0, ys, color="tab:blue", alpha=0.2)
            for p in res.peaks:
                ax.axvline(p, color="tab:red", linestyle="--", linewidth=1)
            for v in res.valleys:
                ax.axvline(v, color="tab:green", linestyle=":", linewidth=1)
            ax.set_title(res.stem)
            ax.set_xlabel("Arcsinh counts")
            ax.set_ylabel("Density")
            fig.tight_layout()
            fig.savefig(plots_dir / f"{res.stem}.png")
            plt.close(fig)

    with open(out / f"results_{slug}.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    _write_combined_zip(
        batch,
        out,
        slug=slug,
        run_metadata=run_metadata,
    )


def _write_combined_zip(
    batch: BatchResults,
    output_dir: Path,
    *,
    slug: str,
    run_metadata: Mapping[str, Any] | None,
) -> None:
    exports = _dataset_exports(batch, run_metadata)
    if exports is None:
        return

    before_meta = exports["meta"]
    before_expr = exports["expr_before"]
    if before_meta is None and before_expr is None and exports["raw_ridge"] is None:
        return

    zip_path = output_dir / f"before_after_alignment_{slug}.zip"
    has_after = exports["expr_after"] is not None or exports["aligned_ridge"] is not None

    with zipfile.ZipFile(zip_path, "w") as archive:
        def _write_ridge_entries(base: str, data: bytes | dict[str, bytes] | None) -> None:
            if data is None:
                return
            if isinstance(data, dict):
                items = sorted(data.items(), key=lambda item: (str(item[0]).lower(), str(item[0])))
                if len(items) == 1 and str(items[0][0]) in {"Default", "default", "None", ""}:
                    archive.writestr(f"{base}.png", items[0][1])
                    return
                for group, blob in items:
                    safe = _sanitize_group_label(str(group) if group is not None else "Default")
                    archive.writestr(f"{base}_{safe}.png", blob)
                return
            archive.writestr(f"{base}.png", data)

        if before_meta is not None:
            archive.writestr("before_alignment/cell_metadata_combined.csv", before_meta)
        if before_expr is not None:
            archive.writestr("before_alignment/expression_matrix_combined.csv", before_expr)
        _write_ridge_entries("before_alignment/before_alignment_ridge", exports["raw_ridge"])
        _write_ridge_entries("after_alignment/aligned_ridge", exports["aligned_ridge"])
        if has_after and exports["meta_after"] is not None:
            archive.writestr("after_alignment/cell_metadata_combined.csv", exports["meta_after"])
        if exports["expr_after"] is not None:
            archive.writestr("after_alignment/expression_matrix_aligned.csv", exports["expr_after"])


def _dataset_exports(
    batch: BatchResults,
    run_metadata: Mapping[str, Any] | None,
) -> dict[str, bytes | dict[str, bytes] | None] | None:
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

    raw_ridge = _ridge_plot_png(
        batch.samples,
        aligned=False,
        group_by_marker=batch.group_by_marker,
    )
    aligned_ridge = (
        _ridge_plot_png(
            batch.samples,
            aligned=True,
            group_by_marker=batch.group_by_marker,
        )
        if has_aligned
        else None
    )

    return {
        "meta": meta_csv,
        "meta_after": meta_csv,
        "expr_before": expr_before_csv,
        "expr_after": expr_after_csv,
        "raw_ridge": raw_ridge,
        "aligned_ridge": aligned_ridge,
    }


def _ridge_plot_png(
    samples: Sequence[SampleResult],
    *,
    aligned: bool,
    group_by_marker: bool = False,
) -> bytes | dict[str, bytes] | None:
    if group_by_marker:
        grouped: dict[str, list[SampleResult]] = {}
        for res in samples:
            marker = res.metadata.get("marker")
            key = str(marker) if marker else "Default"
            grouped.setdefault(key, []).append(res)

        outputs: dict[str, bytes] = {}
        for group_name in sorted(grouped):
            png = _ridge_plot_png(
                grouped[group_name],
                aligned=aligned,
                group_by_marker=False,
            )
            if isinstance(png, bytes):
                outputs[group_name] = png
        return outputs or None

    curves: list[tuple[str, np.ndarray, np.ndarray, list[float], list[float], float]] = []
    for res in samples:
        xs: np.ndarray
        ys: np.ndarray
        if aligned:
            if res.aligned_density is None:
                continue
            xs = np.asarray(res.aligned_density[0], float)
            ys = np.asarray(res.aligned_density[1], float)
            peaks_for_plot = res.aligned_peaks if res.aligned_peaks is not None else res.peaks
            valleys_for_plot = (
                res.aligned_valleys if res.aligned_valleys is not None else res.valleys
            )
        else:
            xs = np.asarray(res.xs, float)
            ys = np.asarray(res.ys, float)
            peaks_for_plot = res.peaks
            valleys_for_plot = res.valleys
        if xs.size == 0 or ys.size == 0:
            continue
        height = max(float(np.nanmax(ys)), 1e-6)
        curves.append(
            (
                res.stem,
                xs,
                ys,
                list(peaks_for_plot),
                list(valleys_for_plot),
                height,
            )
        )

    if not curves:
        return None

    trimmed_bounds: list[tuple[float, float]] = []
    for _, xs, _, _, _, _ in curves:
        finite_xs = xs[np.isfinite(xs)]
        if finite_xs.size == 0:
            continue
        low, high = np.nanquantile(finite_xs, [0.01, 0.99])
        trimmed_bounds.append((float(low), float(high)))

    if trimmed_bounds:
        x_min = min(low for low, _ in trimmed_bounds)
        x_max = max(high for _, high in trimmed_bounds)
    else:
        x_min = min(float(xs.min()) for _, xs, _, _, _, _ in curves)
        x_max = max(float(xs.max()) for _, xs, _, _, _, _ in curves)

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return None

    if x_max == x_min:
        pad = 0.05 * (abs(x_min) if x_min != 0 else 1.0)
    else:
        pad = 0.05 * (x_max - x_min)

    offsets: list[float] = []
    current_offset = 0.0
    for _, _, _, _, _, height in curves:
        offsets.append(current_offset)
        step = max(height, 1e-6) * 1.2
        current_offset += step

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 0.8 * max(len(curves), 1)), dpi=150, sharex=True)

    text_transform = ax.get_yaxis_transform()

    for offset, (stem, xs, ys, peaks, valleys, height) in zip(offsets, curves):
        ax.plot(xs, ys + offset, color="black", lw=1)
        ax.fill_between(xs, offset, ys + offset, color="#FFA50088", lw=0)
        ymax_local = max(height, 1e-6)
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
            -0.02,
            offset + 0.5 * ymax_local,
            stem,
            ha="right",
            va="center",
            fontsize=7,
            transform=text_transform,
            clip_on=False,
        )

    ax.set_yticks([])
    total_height = offsets[-1] + curves[-1][5]
    spacing = float(np.median(np.diff(offsets))) if len(offsets) > 1 else float(total_height)
    y_margin = 0.35 * spacing if spacing > 0 else 0.0
    ax.set_ylim(-y_margin, total_height + y_margin)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.margins(x=0, y=0)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()

