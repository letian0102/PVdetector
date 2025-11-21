# app.py  – GPT-assisted bandwidth detector with
#           live incremental results + per-sample overrides
from __future__ import annotations
import io, zipfile, re, math, time, os
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from pathlib import Path
import warnings
from queue import Empty, Queue
from threading import Thread

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI, AuthenticationError

try:  # NumPy 2.1+
    from numpy import get_array_backend as _np_get_array_backend  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    _np_get_array_backend = None


def get_array_backend(*arrays, **kwargs):
    """Return the array backend associated with ``arrays`` (NumPy fallback)."""

    if _np_get_array_backend is None:
        return np
    return _np_get_array_backend(*arrays, **kwargs)

from peak_valley.quality import stain_quality

from peak_valley import (
    arcsinh_transform, read_counts, load_combined_csv,
    kde_peaks_valleys, quick_peak_estimate,
    fig_to_png, enforce_marker_consistency,
)
from peak_valley.cli_import import (
    derive_min_separation,
    parse_peak_positions,
)
from peak_valley.gpt_adapter import (
    ask_gpt_peak_count, ask_gpt_prominence, ask_gpt_bandwidth,
)
from peak_valley.batch import (
    BatchOptions,
    BatchResults,
    SampleInput,
    SampleResult,
    run_batch,
)
from peak_valley.alignment import align_distributions, fill_landmark_matrix

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.cluster._kmeans",
)

# ────────────────────────── Streamlit page & state ──────────────────────────
st.set_page_config("Peak & Valley Detector", None, layout="wide")

DEFAULT_MIN_SEPARATION = 0.5
DEFAULT_WORKERS = max(1, min(4, (os.cpu_count() or 1)))
st.title("Peak & Valley Detector — CSV *or* full dataset")
st.warning(
    "**Heads-up:** if you refresh or close this page, all of your uploaded data and results will be lost."
)

st.markdown(
    (
        "<a href='Magic%20Book.html' target='_blank'>"
        "<button style='position: fixed; top: 10px; right: 10px; z-index: 1000;'>"
        "User Manual"
        "</button></a>"
    ),
    unsafe_allow_html=True,
)

st.session_state.setdefault("active_sample", None)
st.session_state.setdefault("active_subtab", {})   # stem → "plot" / "params" / "manual"

st.session_state.setdefault("paused", False)
st.session_state.setdefault("invalid_api_key", False)

for key, default in {
    "results":     {},         # stem → {"peaks":…, "valleys":…, "xs":…, "ys":…}
    "results_raw": {},         # stem → raw counts (np.ndarray)
    "results_raw_meta": {},    # stem → {"arcsinh": tuple, "protein_name": str, …}
    "fig_pngs":    {},         # stem.png → png bytes (latest)
    "params":      {},         # stem → {"bw":…, "prom":…, "n_peaks":…}
    "dirty":       {},         # stem → True if user edited params *or* positions
    "dirty_reason": {},        # stem → set[str] recording why a sample is dirty
    "pre_overrides": {},       # stem → per-sample overrides (persistent)
      "group_assignments": {},   # stem → group name
      "group_overrides": {"Default": {}},
      "group_new_name": "",
    "group_new_name_reset": False,
    "group_assignments_mode": "Manual selection",
    "cached_uploads": [],
    "generated_csvs": [],
    "generated_meta": {},
    "sel_markers": [], "sel_samples": [], "sel_batches": [],
    "expr_df": None, "meta_df": None,
    "expr_name": None, "meta_name": None,
    "combined_expr_df": None,
    "combined_meta_df": None,
    "combined_expr_bytes": None,
    "combined_meta_bytes": None,
    "combined_expr_sources": [],
    "combined_meta_sources": [],
    "combined_expr_name": None,
    "combined_meta_name": None,
    "combine_expr_upload": None,
    "combine_meta_upload": None,
    "cli_summary_df": None,
    "cli_summary_name": None,
    "cli_summary_selection": [],
    "cli_import_status": None,
    "cli_filter_text": "",
    "cli_counts_native": False,
    "cli_summary_lookup": {},
    "cli_positions_cache": {},
    "cli_positions_pending": [],
    "cli_positions_fixed": set(),
    "cli_group_mode": "none",
    "cli_group_new_name": "",
    "align_group_markers": False,
    "mode_selector": "Counts CSV files",
    "mode_selector_target": None,
      # incremental‑run machinery
      "pending":     [],         # list[io.BytesIO] still to process
      "total_todo":  0,
      "run_active":  False,
      "batch_queue": None,
      "batch_thread": None,
      "batch_run_id": 0,
      "batch_error": None,
      "batch_progress": {"completed": 0, "total": 0},
      "workers": DEFAULT_WORKERS,
    "aligned_counts":    None,
    "aligned_landmarks": None,
    "aligned_results": {},   # stem → {"peaks":…, "valleys":…, "xs":…, "ys":…}
    "aligned_fig_pngs": {},  # stem_aligned.png → bytes
    "aligned_ridge_png":    None,
    "apply_consistency": False,  # enforce marker consistency across samples
    "raw_ridge_png": None,
    "excluded_markers": [],
    "excluded_samples": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


pending_mode_selector = st.session_state.pop("mode_selector_target", None)
if pending_mode_selector is not None:
    st.session_state["mode_selector"] = pending_mode_selector


def _keyify(label: str) -> str:
    """Sanitize labels so they are safe to use in Streamlit widget keys."""
    return re.sub(r"[^0-9A-Za-z_]+", "_", label)


def _normalize_label(value) -> str | None:
    """Return a stripped string label or ``None`` when empty."""

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()

    return text or None


def _normalize_label_casefold(value) -> str | None:
    """Normalize ``value`` and compare case-insensitively."""

    base = _normalize_label(value)
    if base is None:
        return None
    return base.casefold()


def _marker_lookup_variants(value) -> list[str]:
    """Return string variants that can help match marker column names."""

    if value is None:
        return []

    text = str(value)
    variants: list[str] = []

    def _add(option: str | None) -> None:
        if option and option not in variants:
            variants.append(option)

    _add(text)
    stripped = text.strip()
    _add(stripped)

    lowered = stripped.casefold()
    _add(lowered)

    squeezed = re.sub(r"[\s_-]+", "", lowered)
    _add(squeezed)

    canonical = re.sub(r"[^0-9a-z]+", "", lowered)
    _add(canonical)

    return [variant for variant in variants if variant]


def _build_expr_marker_lookup(expr_df: pd.DataFrame) -> dict[str, object]:
    """Return a mapping of marker-name variants to expression columns."""

    lookup: dict[str, object] = {}
    for column in expr_df.columns:
        for key in _marker_lookup_variants(column):
            if key not in lookup:
                lookup[key] = column
    return lookup


def _resolve_expr_marker(marker: object, lookup: dict[str, object]) -> object | None:
    """Resolve ``marker`` to an expression column using ``lookup`` variants."""

    for key in _marker_lookup_variants(marker):
        column = lookup.get(key)
        if column is not None:
            return column
    return None


def _summarize_sources(label: str, sources: list[str]) -> str | None:
    """Return a short human-readable summary for combined CSV parts."""

    if not sources:
        return None

    display = ", ".join(Path(src).name for src in sources[:3])
    if len(sources) > 3:
        display += ", …"

    plural = "part" if len(sources) == 1 else "parts"
    return f"{label} combined from {len(sources)} {plural}: {display}"


def _summarize_overrides(overrides: dict[str, object]) -> list[str]:
    """Return a list of human-readable summaries for override values."""
    summary: list[str] = []

    if "n_peaks" in overrides:
        summary.append(f"{int(overrides['n_peaks'])} peaks")
    if "max_peaks" in overrides:
        summary.append(f"max peaks {int(overrides['max_peaks'])}")
    if "bw" in overrides:
        bw_val = overrides["bw"]
        if isinstance(bw_val, (int, float)):
            summary.append(f"bandwidth {float(bw_val):.2f}")
        else:
            summary.append(f"bandwidth {bw_val}")
    if "prom" in overrides:
        summary.append(f"prominence {float(overrides['prom']):.2f}")
    if "min_width" in overrides:
        summary.append(f"min width {int(overrides['min_width'])}")
    if "curvature" in overrides:
        summary.append(f"curvature ≥ {float(overrides['curvature']):.4f}")
    if "turning_points" in overrides:
        summary.append(
            "turning pts→peaks" if overrides["turning_points"] else "turning pts off"
        )
    if "min_separation" in overrides:
        summary.append(f"min sep {float(overrides['min_separation']):.1f}")
    if "max_grid" in overrides:
        summary.append(f"grid {int(overrides['max_grid'])}")
    if "valley_drop" in overrides:
        summary.append(f"valley drop {float(overrides['valley_drop']):.0f}%")
    if "first_valley" in overrides:
        val = str(overrides["first_valley"]).lower()
        summary.append(f"first valley: {val}")

    return summary


def _clean_stem_label(value: object) -> str | None:
    """Return a stripped stem label or ``None`` when empty/invalid."""

    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    return cleaned or None


def _clean_cli_positions(values) -> list[float]:
    """Return finite float values extracted from a CLI summary entry."""

    cleaned: list[float] = []
    if not values:
        return cleaned

    try:
        iterator = list(values)
    except TypeError:
        return cleaned

    for value in iterator:
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            cleaned.append(num)

    return cleaned


def _release_cli_positions(stem: str) -> None:
    """Allow ``stem`` to diverge from its imported CLI positions."""

    fixed_raw = st.session_state.get("cli_positions_fixed")
    if not fixed_raw:
        return

    fixed = set(fixed_raw) if not isinstance(fixed_raw, set) else set(fixed_raw)
    if stem in fixed:
        fixed.discard(stem)
        st.session_state.cli_positions_fixed = fixed


def _restore_cli_positions(stems: Iterable[str] | None = None) -> None:
    """Reapply imported CLI landmark positions for ``stems``."""

    if not st.session_state.get("cli_counts_native"):
        return

    fixed_raw = st.session_state.get("cli_positions_fixed")
    if not fixed_raw:
        return

    if stems is None:
        if isinstance(fixed_raw, (set, list, tuple)):
            stems_iter = list(fixed_raw)
        else:
            stems_iter = []
    else:
        stems_iter = list(stems)

    if not stems_iter:
        return

    cache = st.session_state.get("cli_positions_cache")
    if not isinstance(cache, dict) or not cache:
        cache = st.session_state.get("cli_summary_lookup") or {}

    for stem in stems_iter:
        entry = cache.get(stem)
        if not isinstance(entry, dict):
            continue

        peaks = _clean_cli_positions(entry.get("peaks"))
        valleys = _clean_cli_positions(entry.get("valleys"))
        if not peaks and not valleys:
            continue

        info = st.session_state.results.get(stem)
        if not isinstance(info, dict):
            continue

        if peaks:
            info["peaks"] = peaks.copy()
        if valleys:
            info["valleys"] = valleys.copy()

        counts = st.session_state.results_raw.get(stem)
        if counts is not None:
            info["quality"] = float(stain_quality(counts, info["peaks"], info["valleys"]))

        params = st.session_state.params.get(stem)
        if isinstance(params, dict) and info.get("peaks"):
            params["n_peaks"] = len(info["peaks"])

        pk_key, vl_key = f"{stem}__pk_list", f"{stem}__vl_list"
        st.session_state[pk_key] = info["peaks"].copy()
        st.session_state[vl_key] = info["valleys"].copy()


def _apply_cli_positions(
    stem: str,
    peaks: list[float],
    valleys: list[float],
) -> tuple[list[float], list[float], bool]:
    """Replace detected peak/valley positions with CLI imports when available."""

    stem_clean = _clean_stem_label(stem) or (str(stem).strip() if stem is not None else None)
    if not st.session_state.get("cli_counts_native"):
        return peaks, valleys, False

    pending_raw = st.session_state.get("cli_positions_pending")
    fixed_raw = st.session_state.get("cli_positions_fixed")

    def _pending_labels(values) -> list[str]:
        labels: list[str] = []
        try:
            iterator = values if isinstance(values, (list, tuple, set)) else [values]
        except Exception:
            iterator = []
        for value in iterator:
            cleaned = _clean_stem_label(value) or (str(value).strip() if value is not None else None)
            if cleaned:
                labels.append(cleaned)
        return labels

    pending_iter = []
    if pending_raw:
        pending_iter = _pending_labels(pending_raw)
    fixed = set(fixed_raw) if isinstance(fixed_raw, (set, list, tuple)) else set()

    should_apply = False
    if stem_clean and stem_clean in pending_iter:
        should_apply = True
        st.session_state.cli_positions_pending = [s for s in pending_iter if s != stem_clean]
    elif stem_clean and stem_clean in fixed:
        should_apply = True

    if not should_apply:
        return peaks, valleys, False

    lookup = st.session_state.get("cli_summary_lookup")
    if not isinstance(lookup, dict) or not lookup:
        cache_lookup = st.session_state.get("cli_positions_cache")
        if isinstance(cache_lookup, dict) and cache_lookup:
            lookup = cache_lookup
        else:
            return peaks, valleys, False

    entry = lookup.get(stem_clean)
    if not isinstance(entry, dict):
        return peaks, valleys, False

    new_peaks = _clean_cli_positions(entry.get("peaks"))
    new_valleys = _clean_cli_positions(entry.get("valleys"))

    applied = False
    if new_peaks:
        peaks = new_peaks
        applied = True
    if new_valleys:
        valleys = new_valleys
        applied = True

    if applied:
        cache = st.session_state.get("cli_positions_cache")
        if isinstance(cache, dict):
            updated = dict(cache)
        else:
            updated = {}
        updated[stem_clean] = {
            "peaks": tuple(peaks),
            "valleys": tuple(valleys),
        }
        st.session_state.cli_positions_cache = updated

        fixed_raw = st.session_state.get("cli_positions_fixed")
        if isinstance(fixed_raw, set):
            fixed = set(fixed_raw)
        elif isinstance(fixed_raw, (list, tuple)):
            fixed = set(fixed_raw)
        else:
            fixed = set()
        fixed.add(stem_clean)
        st.session_state.cli_positions_fixed = fixed

    return peaks, valleys, applied


def _mark_sample_dirty(stem: str, reason: str) -> None:
    """Mark ``stem`` as needing a rerun and record the triggering reason."""

    if stem not in st.session_state.results:
        return

    _release_cli_positions(stem)

    st.session_state.dirty[stem] = True
    current = set(st.session_state.dirty_reason.get(stem, set()))
    current.add(reason)
    st.session_state.dirty_reason[stem] = current


def _update_pre_override(stem: str, overrides: dict[str, object]) -> None:
    """Update per-sample overrides and mark samples dirty when they change."""

    prev = dict(st.session_state.pre_overrides.get(stem, {}))
    prev_clean = {k: v for k, v in prev.items() if v is not None}
    new_clean = {k: v for k, v in overrides.items() if v is not None}

    changed: bool
    if new_clean:
        changed = prev_clean != new_clean
        if changed or stem not in st.session_state.pre_overrides:
            st.session_state.pre_overrides[stem] = new_clean
    else:
        changed = bool(prev_clean)
        if stem in st.session_state.pre_overrides:
            st.session_state.pre_overrides.pop(stem, None)

    if changed:
        _mark_sample_dirty(stem, "override")


def _render_override_controls(
    base_key: str,
    *,
    prev: dict[str, object],
    bw_label: str,
    prom_mode: str,
    prom_default: float | None,
    run_defaults: dict[str, object],
) -> dict[str, object]:
    """Render override widgets and return the resulting override dict."""

    key_prefix = _keyify(base_key)
    overrides: dict[str, object] = {}
    _ = prom_mode  # reserved for future behavior differences

    # ── Peaks / bandwidth / prominence ──────────────────────────────────
    col_peak, col_bw, col_prom = st.columns(3)

    peak_options = [None, 1, 2, 3, 4, 5, 6]
    peak_prev = _coerce_int(prev.get("n_peaks"))
    peak_index = peak_options.index(peak_prev) if peak_prev in peak_options else 0
    n_choice = col_peak.selectbox(
        "Peaks",
        options=peak_options,
        index=peak_index,
        format_func=lambda v: "Use run setting" if v is None else f"{int(v)}",
        key=f"{key_prefix}_pre_peak",
        help="Choose a fixed number of peaks for this item.",
    )
    if n_choice is not None:
        overrides["n_peaks"] = int(n_choice)

    bw_options: list[float | str | None] = [
        None,
        "scott",
        "silverman",
        0.5,
        0.8,
        1.0,
        "custom",
    ]

    bw_prev = prev.get("bw")
    bw_custom_default = 1.0
    if isinstance(bw_prev, (int, float)):
        if bw_prev in (0.5, 0.8, 1.0):
            bw_index = bw_options.index(float(bw_prev))
        else:
            bw_index = bw_options.index("custom")
        bw_custom_default = float(bw_prev)
    elif isinstance(bw_prev, str):
        if bw_prev in ("scott", "silverman"):
            bw_index = bw_options.index(bw_prev)
        else:
            try:
                bw_custom_default = float(bw_prev)
                bw_index = bw_options.index("custom")
            except ValueError:
                bw_index = 0
    else:
        bw_index = 0

    def _fmt_bw(opt):
        if opt is None:
            return f"Use run setting ({bw_label})"
        if opt == "custom":
            return "Custom numeric"
        if isinstance(opt, str):
            return opt.title()
        return f"{opt:.2f}"

    bw_choice = col_bw.selectbox(
        "Bandwidth",
        options=bw_options,
        index=bw_index,
        format_func=_fmt_bw,
        key=f"{key_prefix}_pre_bw_choice",
        help="Pick a bandwidth rule or numeric value for this item.",
    )

    if bw_choice == "custom":
        bw_value = col_bw.slider(
            "Custom bandwidth",
            min_value=0.01,
            max_value=5.0,
            value=float(bw_custom_default),
            step=0.01,
            key=f"{key_prefix}_pre_bw_val",
            help="Manual KDE bandwidth just for this item.",
        )
        overrides["bw"] = float(bw_value)
    elif bw_choice is not None:
        overrides["bw"] = bw_choice

    prom_prev = _coerce_float(prev.get("prom"))
    prom_toggle = col_prom.checkbox(
        "Custom prominence",
        value=(prom_prev is not None),
        key=f"{key_prefix}_pre_prom_toggle",
        help="Enable to override the prominence threshold.",
    )
    if prom_toggle:
        if prom_prev is not None:
            prom_start = float(prom_prev)
        elif prom_default is not None:
            prom_start = float(prom_default)
        else:
            prom_start = 0.05
        prom_override = col_prom.slider(
            "Prominence",
            min_value=0.00,
            max_value=0.30,
            value=float(prom_start),
            step=0.01,
            key=f"{key_prefix}_pre_prom_val",
            help="Set the minimum drop from peak to valley.",
        )
        overrides["prom"] = float(prom_override)

    # ── Advanced parameters ──────────────────────────────────────────────
    col_cap, col_width, col_curv = st.columns(3)

    max_prev = _coerce_int(prev.get("max_peaks"))
    max_options = [None, 1, 2, 3, 4, 5, 6]
    max_index = max_options.index(max_prev) if max_prev in max_options else 0
    max_choice = col_cap.selectbox(
        "Max peaks (auto cap)",
        options=max_options,
        index=max_index,
        format_func=lambda v: "Use run setting" if v is None else f"{int(v)}",
        key=f"{key_prefix}_max_peaks",
        help="Upper limit when GPT estimates the peak count.",
    )
    if max_choice is not None:
        overrides["max_peaks"] = int(max_choice)

    width_prev = _coerce_int(prev.get("min_width"))
    width_options = [None, 0, 1, 2, 3, 4, 5, 6]
    width_index = width_options.index(width_prev) if width_prev in width_options else 0
    width_choice = col_width.selectbox(
        "Min peak width",
        options=width_options,
        index=width_index,
        format_func=lambda v: "Use run setting" if v is None else f"{int(v)}",
        key=f"{key_prefix}_min_width",
        help="Smallest allowable width for detected peaks.",
    )
    if width_choice is not None:
        overrides["min_width"] = int(width_choice)

    curv_prev = _coerce_float(prev.get("curvature"))
    curv_toggle = col_curv.checkbox(
        "Custom curvature",
        value=(curv_prev is not None),
        key=f"{key_prefix}_curv_toggle",
        help="Override the curvature threshold for peaks.",
    )
    if curv_toggle:
        default_curv = (
            curv_prev
            if curv_prev is not None
            else float(run_defaults.get("curvature", 0.0001))
        )
        curv_val = col_curv.slider(
            "Curvature threshold",
            min_value=0.0000,
            max_value=0.0050,
            value=float(default_curv),
            step=0.0001,
            format="%.4f",
            key=f"{key_prefix}_curv_val",
            help="Filter out peaks with curvature below this value; 0 disables.",
        )
        overrides["curvature"] = float(curv_val)

    col_turn, col_sep, col_grid = st.columns(3)

    turn_prev_raw = prev.get("turning_points")
    if isinstance(turn_prev_raw, str):
        turn_prev = True if turn_prev_raw.lower().startswith("y") else False if turn_prev_raw.lower().startswith("n") else None
    elif isinstance(turn_prev_raw, bool):
        turn_prev = turn_prev_raw
    else:
        turn_prev = None
    turn_options = [None, True, False]
    def _fmt_turn(opt):
        if opt is None:
            return "Use run setting"
        return "Yes" if opt else "No"

    turn_choice = col_turn.selectbox(
        "Treat turning points as peaks",
        options=turn_options,
        index=turn_options.index(turn_prev) if turn_prev in turn_options else 0,
        format_func=_fmt_turn,
        key=f"{key_prefix}_turning",
        help="Count concave-down turning points as peaks (Yes) or ignore them (No).",
    )
    if turn_choice is not None:
        overrides["turning_points"] = bool(turn_choice)

    sep_prev = _coerce_float(prev.get("min_separation"))
    sep_toggle = col_sep.checkbox(
        "Custom min separation",
        value=(sep_prev is not None),
        key=f"{key_prefix}_sep_toggle",
        help="Override the minimum distance between peaks.",
    )
    if sep_toggle:
        default_sep = (
            sep_prev
            if sep_prev is not None
            else float(run_defaults.get("min_separation", DEFAULT_MIN_SEPARATION))
        )
        sep_val = col_sep.slider(
            "Min peak separation",
            min_value=0.0,
            max_value=10.0,
            value=float(default_sep),
            step=0.1,
            key=f"{key_prefix}_sep_val",
            help="Minimum distance required between peaks.",
        )
        overrides["min_separation"] = max(0.0, float(sep_val))

    grid_prev = _coerce_int(prev.get("max_grid"))
    grid_toggle = col_grid.checkbox(
        "Custom max grid",
        value=(grid_prev is not None),
        key=f"{key_prefix}_grid_toggle",
        help="Override the maximum number of KDE grid points.",
    )
    if grid_toggle:
        default_grid = (
            grid_prev
            if grid_prev is not None
            else int(run_defaults.get("max_grid", 20000))
        )
        grid_val = col_grid.slider(
            "Max KDE grid",
            min_value=4_000,
            max_value=40_000,
            value=int(default_grid),
            step=1_000,
            key=f"{key_prefix}_grid_val",
            help="Number of grid points for KDE; higher is slower but finer.",
        )
        overrides["max_grid"] = int(grid_val)

    col_drop, col_first, _ = st.columns(3)

    drop_prev = _coerce_float(prev.get("valley_drop"))
    drop_toggle = col_drop.checkbox(
        "Custom valley drop",
        value=(drop_prev is not None),
        key=f"{key_prefix}_drop_toggle",
        help="Override the drop required to mark a valley.",
    )
    if drop_toggle:
        default_drop = (
            drop_prev
            if drop_prev is not None
            else float(run_defaults.get("valley_drop", 10))
        )
        drop_val = col_drop.slider(
            "Valley drop (% of peak)",
            min_value=1,
            max_value=50,
            value=int(default_drop),
            step=1,
            key=f"{key_prefix}_drop_val",
            help="Required drop from peak height to qualify as a valley.",
        )
        overrides["valley_drop"] = float(drop_val)

    first_prev = prev.get("first_valley")
    if isinstance(first_prev, str) and first_prev not in ("Slope change", "Valley drop"):
        first_prev = None
    first_options = [None, "Slope change", "Valley drop"]
    first_choice = col_first.selectbox(
        "First valley method",
        options=first_options,
        index=first_options.index(first_prev) if first_prev in first_options else 0,
        key=f"{key_prefix}_first_valley",
        help="Choose how to determine the first valley after a peak.",
    )
    if first_choice is not None:
        overrides["first_valley"] = first_choice

    return overrides

# ── helper #1 : aligned ZIP  (plots + aligned counts) ───────────────
def _refresh_raw_ridge() -> None:
    """Re-compute the stacked RAW ridge plot and store it in session_state."""
    if not st.session_state.results:
        st.session_state.raw_ridge_png = None
        return

    use_groups = bool(st.session_state.get("align_group_markers"))
    stems = _ordered_stems_for_results(use_groups=use_groups)
    st.session_state.raw_ridge_png = _ridge_plot_for_stems(
        stems, st.session_state.results
    )


def _mark_raw_ridge_stale() -> None:
    """Flag the cached raw ridge plot so it will be regenerated lazily."""
    st.session_state.raw_ridge_png = None


def _current_arcsinh_signature() -> tuple[bool, float, float, float]:
    """Return a tuple describing the current arcsinh configuration."""
    apply_arc = bool(st.session_state.get("apply_arcsinh", True))
    return (
        apply_arc,
        float(st.session_state.get("arcsinh_a", 1.0)),
        float(st.session_state.get("arcsinh_b", 1 / 5)),
        float(st.session_state.get("arcsinh_c", 0.0)),
    )


def _stem_offsets_for_plot(
    stems: list[str], group_for: dict[str, str], gap: float
) -> tuple[list[float], int]:
    """Return stacked offsets for ``stems`` with gaps between marker groups."""

    offsets: list[float] = []
    offset = 0.0
    breaks = 0
    last_group: str | None = None

    for stem in stems:
        group = group_for.get(stem, "Default")
        if last_group is not None and group != last_group:
            offset += 0.5 * gap
            breaks += 1
        offsets.append(offset)
        offset += gap
        last_group = group

    return offsets, breaks


def _ridge_plot_for_stems(
    stems: list[str],
    results_map: dict[str, dict[str, object]],
) -> bytes | None:
    """Return a stacked ridge plot PNG for the provided stems."""

    curve_info: list[tuple[str, np.ndarray, np.ndarray, list[float], list[float], float]] = []

    for stem in stems:
        info = results_map.get(stem)
        if not info:
            continue

        xs = np.asarray(info.get("xs", []), float)
        ys = np.asarray(info.get("ys", []), float)
        if xs.size == 0 or ys.size == 0:
            continue

        peaks = list(info.get("peaks", []))
        valleys = list(info.get("valleys", []))
        height = max(float(np.nanmax(ys)), 1e-6)
        curve_info.append((stem, xs, ys, peaks, valleys, height))

    if not curve_info:
        return None

    x_min = min(float(xs.min()) for _, xs, _, _, _, _ in curve_info)
    x_max = max(float(xs.max()) for _, xs, _, _, _, _ in curve_info)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return None

    if x_max == x_min:
        span = abs(x_min) if x_min != 0 else 1.0
        pad = 0.05 * span
    else:
        pad = 0.05 * (x_max - x_min)

    offsets: list[float] = []
    current_offset = 0.0
    for _, _, _, _, _, height in curve_info:
        offsets.append(current_offset)
        current_offset += max(height, 1e-6) * 1.2

    fig, ax = plt.subplots(
        figsize=(6, 0.8 * max(len(curve_info), 1)),
        dpi=150,
        sharex=True,
    )

    for offset, (stem, xs, ys, peaks, valleys, height) in zip(offsets, curve_info):
        ax.plot(xs, ys + offset, color="black", lw=1)
        ax.fill_between(xs, offset, ys + offset, color="#FFA50088", lw=0)

        ymax = max(height, 1e-6)

        for peak_val in peaks:
            try:
                peak_float = float(peak_val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(peak_float):
                ax.vlines(peak_float, offset, offset + ymax, color="black", lw=0.8)

        for valley_val in valleys:
            try:
                valley_float = float(valley_val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(valley_float):
                ax.vlines(
                    valley_float,
                    offset,
                    offset + ymax,
                    color="grey",
                    lw=0.8,
                    linestyles=":",
                )

        ax.text(
            x_min,
            offset + 0.5 * ymax,
            stem,
            ha="right",
            va="center",
            fontsize=7,
        )

    ax.set_yticks([])
    ax.set_xlim(x_min - pad, x_max + pad)
    fig.tight_layout()

    png = fig_to_png(fig)
    plt.close(fig)
    return png


def _group_stems_with_results() -> dict[str, list[str]]:
    """Return a mapping of group → stems for processed results."""

    assignments = st.session_state.get("group_assignments", {}) or {}
    use_marker_groups = bool(st.session_state.get("align_group_markers"))
    groups: dict[str, list[str]] = {}

    for stem in st.session_state.results:
        group = assignments.get(stem)

        if use_marker_groups and (not isinstance(group, str) or not group or group == "Default"):
            marker = _stem_marker(stem)
            group = _normalize_label_casefold(marker) or _normalize_label(marker)

        if not isinstance(group, str) or not group:
            group = "Default"

        groups.setdefault(group, []).append(stem)

    return groups


def _ordered_stems_for_results(*, use_groups: bool) -> list[str]:
    """Return stems ordered by group (Default first) or as-is when grouping is off."""

    stems = list(st.session_state.results)
    if not use_groups:
        return stems

    groups = _group_stems_with_results()

    def _group_sort(item: tuple[str, list[str]]) -> tuple[int, str]:
        name, _ = item
        return (0, "") if name == "Default" else (1, name.lower())

    ordered: list[str] = []
    for _, group_stems in sorted(groups.items(), key=_group_sort):
        ordered.extend(group_stems)

    for stem in stems:
        if stem not in ordered:
            ordered.append(stem)

    return ordered


def _align_results_by_group(*, align_mode: str, target_vec: list[float] | None) -> None:
    """Align results, respecting marker-based grouping when enabled."""

    if not st.session_state.results:
        st.error("Run the detector first to align landmarks.")
        st.stop()

    use_groups = bool(st.session_state.get("align_group_markers"))
    group_map = (
        _group_stems_with_results() if use_groups else {"Default": list(st.session_state.results)}
    )

    def _group_sort(item: tuple[str, list[str]]) -> tuple[int, str]:
        name, _ = item
        return (0, "") if name == "Default" else (1, name.lower())

    xs_map: dict[str, np.ndarray] = {}
    ys_map: dict[str, np.ndarray] = {}
    aligned_counts: dict[str, np.ndarray] = {}
    aligned_results: dict[str, dict[str, object]] = {}
    landmark_rows: dict[str, list[float]] = {}

    all_xmin = np.inf
    all_xmax = -np.inf
    all_ymax = 0.0

    for _, stems in sorted(group_map.items(), key=_group_sort):
        if not stems:
            continue

        peaks_all = [st.session_state.results[s]["peaks"] for s in stems]
        valleys_all = [st.session_state.results[s]["valleys"] for s in stems]
        counts_all = [st.session_state.results_raw[s] for s in stems]

        landmark_mat = fill_landmark_matrix(
            peaks=peaks_all,
            valleys=valleys_all,
            align_type=align_mode,
            midpoint_type="valley",
        )

        if target_vec is None:
            target_landmark = np.nanmedian(landmark_mat, axis=0).tolist()
        else:
            need = _cols_for_align_mode(align_mode)
            if len(target_vec) != len(need):
                st.error(
                    f"Need {len(need)} values for '{align_mode}', but got {len(target_vec)}."
                )
                st.stop()
            target_landmark = target_vec

        warped, warped_lm, warp_funs = align_distributions(
            counts_all,
            peaks_all,
            valleys_all,
            align_type=align_mode,
            landmark_matrix=landmark_mat,
            target_landmark=target_landmark,
        )

        warped = list(warped)
        warp_funs = list(warp_funs)
        warped_lm = [list(row) for row in warped_lm]

        neg_peaks = [lm[0] for lm in warped_lm]
        target_neg = float(np.nanmean(neg_peaks)) if neg_peaks else 0.0

        for idx, stem in enumerate(stems):
            if idx >= len(warp_funs):
                peaks = peaks_all[idx]
                offset = (target_neg - peaks[0]) if peaks else 0.0
                warp_funs.append(lambda x, o=offset: x + o)
                warped.append(counts_all[idx] + offset)
                warped_lm.append([
                    (peaks[0] + offset) if peaks else np.nan,
                    np.nan,
                    np.nan,
                ])

            xs_raw = np.asarray(st.session_state.results[stem].get("xs", []), float)
            ys_raw = np.asarray(st.session_state.results[stem].get("ys", []), float)
            warp_fn = warp_funs[idx]

            xs_aligned = warp_fn(xs_raw)
            xs_map[stem] = xs_aligned
            ys_aligned = np.asarray(aligned_counts.get(stem, ys_raw), float)
            ys_map[stem] = ys_aligned
            if xs_aligned.size:
                all_xmin = min(all_xmin, float(xs_aligned.min()))
                all_xmax = max(all_xmax, float(xs_aligned.max()))
            if ys_aligned.size:
                all_ymax = max(all_ymax, float(ys_aligned.max()))

            peaks_aligned = warp_fn(np.asarray(st.session_state.results[stem]["peaks"]))
            valleys_aligned = warp_fn(np.asarray(st.session_state.results[stem]["valleys"]))
            aligned_counts[stem] = np.asarray(warped[idx], float)

            try:
                sample_ymax = float(np.nanmax(ys_aligned))
            except ValueError:
                sample_ymax = all_ymax

            sample_xmin = float(xs_aligned.min()) if xs_aligned.size else all_xmin
            sample_xmax = float(xs_aligned.max()) if xs_aligned.size else all_xmax
            if not np.isfinite(sample_xmin) or not np.isfinite(sample_xmax):
                sample_xmin, sample_xmax = all_xmin, all_xmax

            if sample_xmax == sample_xmin:
                span = abs(sample_xmin) if sample_xmin != 0 else 1.0
                pad_local = 0.05 * span
            else:
                pad_local = 0.05 * (sample_xmax - sample_xmin)

            if not np.isfinite(sample_ymax) or sample_ymax <= 0:
                sample_ymax = all_ymax if np.isfinite(all_ymax) and all_ymax > 0 else 1.0

            png = _plot_png_fixed(
                f"{stem} (aligned)",
                xs_raw,
                ys_aligned,
                peaks_aligned[~np.isnan(peaks_aligned)],
                valleys_aligned[~np.isnan(valleys_aligned)],
                (sample_xmin - pad_local, sample_xmax + pad_local),
                sample_ymax,
            )

            st.session_state.aligned_fig_pngs[f"{stem}_aligned.png"] = png
            aligned_results[stem] = {
                "peaks": peaks_aligned.round(4).tolist(),
                "valleys": valleys_aligned.round(4).tolist(),
                "xs": xs_aligned.tolist(),
                "ys": ys_aligned.tolist(),
            }
            landmark_rows[stem] = list(warped_lm[idx])

    if not np.isfinite(all_xmin) or not np.isfinite(all_xmax):
        all_xmin, all_xmax = 0.0, 1.0
    elif all_xmax == all_xmin:
        d = abs(all_xmin) if all_xmin != 0 else 1.0
        all_xmin -= 0.05 * d
        all_xmax += 0.05 * d

    span = all_xmax - all_xmin
    pad = 0.05 * span
    if not np.isfinite(all_ymax):
        all_ymax = 1.0

    st.session_state.aligned_counts = aligned_counts
    st.session_state.aligned_landmarks = [
        landmark_rows.get(stem, []) for stem in st.session_state.results
    ]
    st.session_state.aligned_results = aligned_results
    st.session_state.aligned_ridge_png = None

    ordered_stems = _ordered_stems_for_results(use_groups=use_groups)
    gap = 1.2 * all_ymax
    fig, ax = plt.subplots(
        figsize=(6, 0.8 * len(ordered_stems)),
        dpi=150,
        sharex=True,
    )

    for i, stem in enumerate(ordered_stems):
        xs = xs_map.get(stem, np.asarray([]))
        ys = ys_map.get(stem, np.asarray([]))
        offset = i * gap

        ax.plot(xs, ys + offset, color="black", lw=1)
        ax.fill_between(xs, offset, ys + offset, color="#FFA50088", lw=0)

        info = aligned_results.get(stem)
        if info:
            for p in info["peaks"]:
                ax.vlines(p, offset, offset + ys.max(), color="black", lw=0.8)
            for v in info["valleys"]:
                ax.vlines(v, offset, offset + ys.max(), color="grey", lw=0.8, linestyles=":")

        ax.text(all_xmin - pad, offset + 0.5 * all_ymax, stem, ha="right", va="center", fontsize=7)

    ax.set_yticks([])
    ax.set_xlim(all_xmin - pad, all_xmax + pad)
    fig.tight_layout()

    st.session_state.aligned_ridge_png = fig_to_png(fig)
    plt.close(fig)


def _ensure_raw_ridge_png() -> bytes | None:
    """Return the cached raw ridge plot, recomputing it when needed."""

    if st.session_state.get("raw_ridge_png") is None:
        _refresh_raw_ridge()
    return st.session_state.get("raw_ridge_png")


def _serializable_counts(array: np.ndarray | None) -> list[float | None]:
    """Convert a numeric array to a JSON-serialisable list of floats/None."""

    if array is None:
        return []

    arr = np.asarray(array, float).ravel()
    serialisable: list[float | None] = []
    for value in arr:
        if np.isnan(value):
            serialisable.append(None)
        elif np.isfinite(value):
            serialisable.append(float(value))
        else:
            serialisable.append(None)
    return serialisable


def _counts_csv_field(array: np.ndarray | None) -> str:
    """Return a CSV-safe string representation of the ``array`` values."""

    values = _serializable_counts(array)
    if not values:
        return "[]"

    parts: list[str] = []
    for value in values:
        if value is None:
            parts.append("null")
        else:
            parts.append(format(value, ".15g"))

    return "[" + "; ".join(parts) + "]"


def _sample_metadata(stem: str) -> dict[str, str]:
    """Collect exported metadata for ``stem`` from cached dataset info."""

    meta_map = st.session_state.get("generated_meta", {}) or {}
    meta = meta_map.get(stem, {})

    def _clean(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value)

    batch_val = meta.get("batch_label", meta.get("batch"))

    return {
        "stem": stem,
        "sample": _clean(meta.get("sample", "")),
        "marker": _clean(meta.get("marker", "")),
        "batch": _clean(batch_val),
    }


def _stem_marker(stem: str) -> str | None:
    """Best-effort lookup of the marker associated with ``stem``."""

    meta_map = st.session_state.get("generated_meta", {}) or {}
    meta = meta_map.get(stem)
    if isinstance(meta, dict):
        marker = meta.get("marker") or meta.get("protein_name")
        if marker:
            marker_str = str(marker).strip()
            if marker_str:
                return marker_str

    raw_meta_map = st.session_state.get("results_raw_meta", {}) or {}
    raw_meta = raw_meta_map.get(stem)
    if isinstance(raw_meta, dict):
        marker = raw_meta.get("marker") or raw_meta.get("protein_name")
        if marker:
            marker_str = str(marker).strip()
            if marker_str:
                return marker_str

    summary_df = st.session_state.get("cli_summary_df")
    if isinstance(summary_df, pd.DataFrame) and "marker" in summary_df.columns:
        try:
            mask = summary_df.get("stem").astype(str).eq(str(stem))
            if mask.any():
                marker_val = summary_df.loc[mask, "marker"].iloc[0]
                if isinstance(marker_val, str):
                    marker_str = marker_val.strip()
                    if marker_str:
                        return marker_str
        except Exception:
            pass

    return None


def _processed_counts_csv() -> bytes:
    """Return a CSV (bytes) with all processed, pre-alignment counts."""

    rows: list[dict[str, object]] = []
    for stem in st.session_state.results:
        counts = st.session_state.results_raw.get(stem)
        if counts is None:
            continue
        meta = _sample_metadata(stem)
        meta["normalized_counts"] = _counts_csv_field(counts)
        rows.append(meta)

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=["stem", "sample", "marker", "batch", "normalized_counts"])

    return df.to_csv(index=False).encode()


def _aligned_counts_csv() -> bytes:
    """Return a CSV (bytes) with aligned counts for every processed sample."""

    aligned_counts = st.session_state.get("aligned_counts") or {}
    rows: list[dict[str, object]] = []
    for stem in st.session_state.results:
        counts = aligned_counts.get(stem)
        if counts is None:
            continue
        meta = _sample_metadata(stem)
        meta["aligned_normalized_counts"] = _counts_csv_field(counts)
        rows.append(meta)

    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(
            columns=["stem", "sample", "marker", "batch", "aligned_normalized_counts"]
        )

    return df.to_csv(index=False).encode()


def _dataset_tables(use_aligned: bool = False) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Build metadata/expression tables mirroring the combined CSV format."""

    meta_df = st.session_state.get("meta_df")
    expr_df = st.session_state.get("expr_df")
    if meta_df is None or expr_df is None:
        return None

    meta_map = st.session_state.get("generated_meta", {}) or {}
    if not meta_map:
        return None

    data_map = (
        st.session_state.get("aligned_counts")
        if use_aligned
        else st.session_state.get("results_raw")
    ) or {}
    if not data_map:
        return None

    stems = [
        stem
        for stem in st.session_state.get("results", {})
        if stem in data_map and stem in meta_map
    ]
    if not stems:
        return None

    marker_seen: list[str] = []
    union_indices: set = set()
    series_entries: list[tuple[str, pd.Series]] = []

    for stem in stems:
        info = meta_map.get(stem, {})
        marker = info.get("marker")
        sample = info.get("sample")
        batch = info.get("batch")
        if not marker:
            continue
        counts = data_map.get(stem)
        if counts is None:
            continue
        indices = info.get("cell_indices")
        if indices and not isinstance(indices, list):
            indices = list(indices)
        if not indices:
            indices = _cell_indices_for(meta_df, sample, batch)
        if not indices:
            continue
        arr = np.asarray(counts, dtype=float).ravel()
        if arr.size == 0:
            continue
        if arr.size != len(indices):
            limit = min(arr.size, len(indices))
            arr = arr[:limit]
            indices = list(indices)[:limit]
        series = pd.Series(arr, index=pd.Index(indices))
        series_entries.append((marker, series))
        union_indices.update(series.index.tolist())
        if marker not in marker_seen:
            marker_seen.append(marker)

    if not series_entries or not union_indices:
        return None

    mask = meta_df.index.isin(union_indices)
    if not mask.any():
        return None
    meta_subset = meta_df.loc[mask].copy()

    expr_columns = [c for c in expr_df.columns if c != "cell_id"]
    marker_order = [c for c in expr_columns if c in marker_seen]
    for marker in marker_seen:
        if marker not in marker_order:
            marker_order.append(marker)

    expr_table = pd.DataFrame(index=meta_subset.index, columns=marker_order, dtype=float)
    for marker, series in series_entries:
        expr_table.loc[series.index, marker] = series.values

    expr_table = expr_table.reindex(meta_subset.index)

    if "cell_id" in expr_df.columns:
        cell_ids = expr_df.loc[meta_subset.index, "cell_id"]
        expr_table.insert(0, "cell_id", cell_ids.values)

    return meta_subset.reset_index(drop=True), expr_table.reset_index(drop=True)


def _dataset_csv_exports(*, aligned: bool = False) -> tuple[bytes | None, bytes | None]:
    """Return (metadata_csv, expression_csv) for dataset-mode exports."""

    tables = _dataset_tables(use_aligned=aligned)
    if not tables:
        return None, None
    meta_tbl, expr_tbl = tables
    return meta_tbl.to_csv(index=False).encode(), expr_tbl.to_csv(index=False).encode()


def _raw_meta_for(stem: str) -> dict[str, object]:
    """Return metadata captured while parsing ``stem``'s source CSV."""

    meta_map = st.session_state.get("results_raw_meta", {}) or {}
    info = meta_map.get(stem)
    if isinstance(info, tuple):  # legacy cache
        info = {"arcsinh": info}
    return dict(info or {})


def _sample_export_filename(stem: str, info: dict[str, object], *, aligned: bool) -> str:
    """Derive a filename for a per-sample export."""

    source = info.get("source_name")
    if source:
        name = Path(str(source)).name
    else:
        name = f"{stem}.csv"

    path = Path(name)
    suffix = path.suffix or ".csv"
    base = path.stem if path.suffix else name
    if aligned:
        base = f"{base}_aligned"
    return f"{base}{suffix}"


def _counts_column_csv(protein: str | None, counts: np.ndarray, fallback: str) -> bytes:
    """Return CSV bytes with the marker name in the first row and counts below."""

    header = (protein or "").strip()
    if not header:
        header = fallback

    arr = np.asarray(counts, dtype=float).ravel()
    column: list[object] = [header]
    column.extend(arr.tolist())

    series = pd.Series(column, dtype="object")
    return series.to_csv(index=False, header=False).encode()


def _sample_csv_exports(*, aligned: bool = False) -> list[tuple[str, bytes]]:
    """Build per-sample CSV exports for before/after alignment downloads."""

    data_map = (
        st.session_state.get("aligned_counts")
        if aligned
        else st.session_state.get("results_raw")
    ) or {}
    if not data_map:
        return []

    exports: list[tuple[str, bytes]] = []
    for stem in st.session_state.get("results", {}):
        counts = data_map.get(stem)
        if counts is None:
            continue
        info = _raw_meta_for(stem)
        filename = _sample_export_filename(stem, info, aligned=aligned)
        csv_bytes = _counts_column_csv(info.get("protein_name"), counts, Path(filename).stem)
        exports.append((filename, csv_bytes))
    return exports


def _protein_label() -> str:
    """Return a filesystem-safe protein/marker label for export filenames."""

    markers: set[str] = set()
    meta_map = st.session_state.get("generated_meta", {}) or {}
    for meta in meta_map.values():
        marker = meta.get("marker")
        if marker:
            markers.add(str(marker))

    if not markers:
        stems = list(st.session_state.results)
        for stem in stems:
            core = stem
            if stem.endswith("_raw_counts"):
                core = stem[: -len("_raw_counts")]
            parts = [p for p in core.split("_") if p]
            if parts:
                markers.add(parts[-1])

    if not markers:
        sel_markers = st.session_state.get("sel_markers")
        if isinstance(sel_markers, list) and sel_markers:
            markers.update(str(m) for m in sel_markers if m)

    if len(markers) == 1:
        base = markers.pop()
    elif len(markers) > 1:
        base = "multiple_proteins"
    else:
        expr_name = st.session_state.get("expr_name")
        base = Path(expr_name).stem if expr_name else "PeakValleyResults"

    safe = re.sub(r"[^0-9A-Za-z_-]+", "_", str(base)).strip("_")
    return safe or "PeakValleyResults"


def _before_alignment_zip_name() -> str:
    return f"{_protein_label()}_before_alignment.zip"


def _final_alignment_zip_name() -> str:
    return f"{_protein_label()}.zip"


def _make_before_alignment_zip() -> bytes:
    """Bundle the pre-alignment ridge plot and dataset exports into a ZIP."""

    raw_png = _ensure_raw_ridge_png()
    meta_csv, expr_csv = _dataset_csv_exports(aligned=False)
    sample_exports = _sample_csv_exports(aligned=False)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        if raw_png:
            z.writestr("before_alignment_ridge.png", raw_png)
        for filename, csv_bytes in sample_exports:
            z.writestr(filename, csv_bytes)
        if meta_csv and expr_csv:
            z.writestr("cell_metadata_combined.csv", meta_csv)
            z.writestr("expression_matrix_combined.csv", expr_csv)
    return out.getvalue()


def _make_final_alignment_zip() -> bytes:
    """Bundle ridge plots plus aligned dataset exports into a ZIP."""

    raw_png = _ensure_raw_ridge_png()
    aligned_png = st.session_state.get("aligned_ridge_png")
    aligned_csv = _aligned_counts_csv()
    meta_aligned_csv, expr_aligned_csv = _dataset_csv_exports(aligned=True)
    meta_csv, expr_csv = _dataset_csv_exports(aligned=False)
    aligned_sample_exports = _sample_csv_exports(aligned=True)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        if raw_png:
            z.writestr("before_alignment_ridge.png", raw_png)
        if aligned_png:
            z.writestr("aligned_ridge.png", aligned_png)
        for filename, csv_bytes in aligned_sample_exports:
            z.writestr(filename, csv_bytes)
        meta_bytes = meta_aligned_csv or meta_csv
        if expr_aligned_csv:
            if meta_bytes:
                z.writestr("cell_metadata_combined.csv", meta_bytes)
            z.writestr("expression_matrix_aligned.csv", expr_aligned_csv)
        elif meta_bytes and expr_csv:
            z.writestr("cell_metadata_combined.csv", meta_bytes)
            z.writestr("expression_matrix_combined.csv", expr_csv)
        z.writestr("aligned_data.csv", aligned_csv)
    return out.getvalue()


def _combined_alignment_zip_name() -> str:
    return f"{_protein_label()}_before_after_alignment.zip"


def _make_combined_alignment_zip() -> bytes:
    """Bundle before- and after-alignment dataset exports into a single ZIP."""

    meta_before, expr_before = _dataset_csv_exports(aligned=False)
    meta_after, expr_after = _dataset_csv_exports(aligned=True)
    sample_before = _sample_csv_exports(aligned=False)
    sample_after = _sample_csv_exports(aligned=True)
    raw_png = _ensure_raw_ridge_png()
    aligned_png = st.session_state.get("aligned_ridge_png")
    aligned_csv = _aligned_counts_csv()

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        if meta_before and expr_before:
            z.writestr("before_alignment/cell_metadata_combined.csv", meta_before)
            z.writestr("before_alignment/expression_matrix_combined.csv", expr_before)
        for filename, csv_bytes in sample_before:
            z.writestr(f"before_alignment/{filename}", csv_bytes)
        if raw_png:
            z.writestr("before_alignment/before_alignment_ridge.png", raw_png)
        if expr_after:
            meta_bytes = meta_after or meta_before
            if meta_bytes:
                z.writestr("after_alignment/cell_metadata_combined.csv", meta_bytes)
            z.writestr("after_alignment/expression_matrix_aligned.csv", expr_after)
        for filename, csv_bytes in sample_after:
            z.writestr(f"after_alignment/{filename}", csv_bytes)
        if aligned_png:
            z.writestr("after_alignment/aligned_ridge.png", aligned_png)
        if aligned_csv:
            z.writestr("after_alignment/aligned_data.csv", aligned_csv)
    return out.getvalue()


def _cell_indices_for(
    meta_df: pd.DataFrame,
    sample: str,
    batch: str | None,
) -> list[int]:
    """Return the DataFrame indices for ``sample`` and optional ``batch``."""

    sample_norm = _normalize_label_casefold(sample)
    if sample_norm is None:
        return []

    sample_series = meta_df["sample"].map(_normalize_label_casefold)
    mask = sample_series.eq(sample_norm)

    if "batch" in meta_df.columns:
        batch_norm = _normalize_label_casefold(batch) if batch is not None else None
        batch_series = meta_df["batch"].map(_normalize_label_casefold)
        if batch_norm is None:
            mask &= batch_series.isna()
        else:
            mask &= batch_series.eq(batch_norm)
    return meta_df.index[mask].tolist()


def _sync_generated_counts(
    sel_m: list[str],
    sel_s: list[str],
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sel_b: list[str] | None = None,
    *,
    summary_rows: Iterable | None = None,
) -> None:
    """Refresh cached CSVs to match the current marker/sample selection.

    The ``stem`` for each generated CSV uniquely identifies the batch, sample,
    and marker combination (``<sample>_<batch>_<marker>_raw_counts``) unless
    explicit stems are supplied via ``summary_rows``.
    """
    use_batches = "batch" in meta_df.columns
    sel_batch_clean = [None if pd.isna(b) else b for b in (sel_b or [])]
    sel_batch_set = set(sel_batch_clean)

    marker_lookup = _build_expr_marker_lookup(expr_df)
    missing_markers: set[str] = set()
    desired: dict[str, tuple[str, str, str | None]] = {}
    index_cache: dict[tuple[str, str | None], list[int]] = {}

    def _cached_indices(sample: str, batch: str | None) -> list[int]:
        key = (sample, batch)
        if key not in index_cache:
            index_cache[key] = _cell_indices_for(meta_df, sample, batch)
        return index_cache[key]

    if summary_rows is not None:
        for row in summary_rows:
            stem_raw = getattr(row, "stem", None)
            if not isinstance(stem_raw, str):
                continue
            stem = stem_raw.strip()
            if not stem:
                continue

            sample = _normalize_label(getattr(row, "sample", None))
            marker_raw = getattr(row, "marker", None)
            marker_label = _normalize_label(marker_raw)
            batch_val = getattr(row, "batch", None) if use_batches else None
            batch_clean = None if pd.isna(batch_val) else batch_val

            if not sample or not marker_label:
                continue
            if sel_batch_set and batch_clean not in sel_batch_set:
                continue

            cell_idx = _cached_indices(sample, batch_clean)
            if not cell_idx:
                continue

            column_label = _resolve_expr_marker(marker_label, marker_lookup)
            if column_label is None:
                missing_markers.add(marker_label)
                continue
            resolved_marker = _normalize_label(column_label) or marker_label
            desired[stem] = (sample, resolved_marker, batch_clean)
    else:
        for s in sel_s:
            sample = _normalize_label(s)
            if not sample:
                continue
            batches = [None]
            if use_batches:
                raw_batches = meta_df.loc[
                    meta_df["sample"].map(_normalize_label) == sample, "batch"
                ].unique()
                cleaned: list[str | None] = []
                for raw in raw_batches:
                    clean = None if pd.isna(raw) else raw
                    if sel_batch_set and clean not in sel_batch_set:
                        continue
                    cleaned.append(clean)
                if not cleaned:
                    continue
                batches = cleaned
            for b in batches:
                cell_idx = _cached_indices(sample, b)
                if not cell_idx:
                    continue
                for m in sel_m:
                    marker_label = _normalize_label(m)
                    if not marker_label:
                        continue
                    column_label = _resolve_expr_marker(marker_label, marker_lookup)
                    if column_label is None:
                        missing_markers.add(marker_label)
                        continue
                    resolved_marker = _normalize_label(column_label) or marker_label
                    stem = (
                        f"{sample}_{resolved_marker}_raw_counts"
                        if b is None
                        else f"{sample}_{b}_{resolved_marker}_raw_counts"
                    )
                    desired[stem] = (sample, resolved_marker, b)

    # remove stale combinations
    st.session_state.generated_csvs = [
        (stem, bio) for stem, bio in st.session_state.generated_csvs
        if stem in desired
    ]
    prev_meta = st.session_state.get("generated_meta", {}) or {}
    updated_meta: dict[str, dict[str, object]] = {}
    for stem, (s, m, b) in desired.items():
        entry = dict(prev_meta.get(stem, {}))
        entry.update(
            {
                "sample": s,
                "marker": m,
                "batch": b,
                "batch_label": _format_batch_label(b),
            }
        )
        if not entry.get("cell_indices"):
            entry["cell_indices"] = list(_cached_indices(s, b))
        updated_meta[stem] = entry
    st.session_state.generated_meta = updated_meta

    # drop outdated results
    keep = set(desired)
    for bucket in ("results", "results_raw", "results_raw_meta", "params", "dirty",
                   "aligned_results"):
        current = st.session_state.get(bucket) or {}
        st.session_state[bucket] = {k: v for k, v in current.items() if k in keep}
    st.session_state.fig_pngs = {
        fn: png for fn, png in st.session_state.fig_pngs.items()
        if fn.split(".")[0] in keep
    }
    st.session_state.aligned_fig_pngs = {
        fn: png for fn, png in st.session_state.aligned_fig_pngs.items()
        if fn.rsplit("_aligned", 1)[0] in keep
    }
    st.session_state.aligned_counts = None
    st.session_state.aligned_landmarks = None
    st.session_state.aligned_ridge_png = None
    _mark_raw_ridge_stale()

    existing = {stem for stem, _ in st.session_state.generated_csvs}
    to_generate: list[tuple[str, str, str, str | None, str, dict[str, object]]]
    to_generate = []

    apply_arcsinh = bool(st.session_state.get("apply_arcsinh", True))
    arcsinh_a = float(st.session_state.get("arcsinh_a", 1.0))
    arcsinh_b = float(st.session_state.get("arcsinh_b", 1 / 5))
    arcsinh_c = float(st.session_state.get("arcsinh_c", 0.0))
    cli_native = bool(st.session_state.get("cli_counts_native"))
    worker_count = max(1, int(st.session_state.get("workers", 1)))

    for stem, (s, m, b) in desired.items():
        if stem in existing:
            continue
        entry = st.session_state.generated_meta.setdefault(stem, {})
        entry.update(
            {
                "sample": s,
                "marker": m,
                "batch": b,
                "batch_label": _format_batch_label(b),
            }
        )
        indices = entry.get("cell_indices") or _cached_indices(s, b)
        if not indices:
            continue
        column_label = _resolve_expr_marker(m, marker_lookup)
        if column_label is None:
            missing_markers.add(m)
            continue
        entry["cell_indices"] = list(indices)
        to_generate.append((stem, s, m, b, column_label, entry))

    def _build_counts(args: tuple[str, str, str, str | None, str, dict[str, object]]):
        stem, sample, marker, batch, column_label, entry = args
        vals = expr_df.loc[entry["cell_indices"], column_label]

        if cli_native:
            counts = vals.to_numpy(dtype=float, copy=True)
            arcsinh_applied = True
        elif apply_arcsinh:
            counts = arcsinh_transform(vals, a=arcsinh_a, b=arcsinh_b, c=arcsinh_c)
            arcsinh_applied = True
        else:
            counts = vals.astype(float)
            arcsinh_applied = False

        bio = io.BytesIO()
        counts_df = pd.DataFrame({marker: np.asarray(counts).ravel()})
        counts_df.to_csv(bio, index=False)
        bio.seek(0)
        bio.name = f"{stem}.csv"
        setattr(bio, "marker", marker)
        setattr(bio, "arcsinh", arcsinh_applied)
        setattr(bio, "sample", sample)
        setattr(bio, "batch", batch)
        return stem, bio, entry

    generated: list[tuple[str, io.BytesIO, dict[str, object]]] = []
    if worker_count > 1 and len(to_generate) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            for item in pool.map(_build_counts, to_generate):
                generated.append(item)
    else:
        for args in to_generate:
            generated.append(_build_counts(args))

    st.session_state.generated_csvs.extend((stem, bio) for stem, bio, _ in generated)
    for stem, _, entry in generated:
        st.session_state.generated_meta[stem] = entry

    if missing_markers:
        st.warning(
            "Skipped markers missing from the expression matrix: "
            + ", ".join(sorted(missing_markers))
            + "."
        )


# ─────────────────────────── CLI import helpers ──────────────────────────────

def _clear_cli_import() -> None:
    """Remove CLI import state without touching current results."""

    for key in ("cli_summary_df", "cli_summary_name", "cli_import_status"):
        st.session_state[key] = None
    st.session_state["cli_summary_selection"] = []
    st.session_state["cli_filter_text"] = ""
    st.session_state["cli_counts_native"] = False
    st.session_state["cli_summary_lookup"] = {}
    st.session_state["cli_positions_cache"] = {}
    st.session_state["cli_positions_pending"] = []
    st.session_state["cli_positions_fixed"] = set()
    st.session_state["cli_group_mode"] = "none"
    st.session_state["cli_group_new_name"] = ""
    st.session_state["align_group_markers"] = False


def _load_cli_import(
    summary_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    summary_name: str,
    expr_name: str,
    meta_name: str,
) -> None:
    """Store imported CLI outputs in session state and prime selectors."""

    summary_df = summary_df.copy()
    if "stem" in summary_df.columns:
        summary_df["stem"] = summary_df["stem"].map(_clean_stem_label)

    st.session_state.expr_df = expr_df
    st.session_state.meta_df = meta_df
    st.session_state.expr_name = expr_name
    st.session_state.meta_name = meta_name

    st.session_state.cli_summary_df = summary_df
    st.session_state.cli_summary_name = summary_name
    st.session_state.cli_import_status = (
        f"Loaded {len(summary_df)} entries from {summary_name}"
    )
    st.session_state.cli_counts_native = True
    st.session_state.align_group_markers = False

    stems = [s for s in summary_df.get("stem", []) if isinstance(s, str) and s]
    valid_stems = set(stems)
    st.session_state.cli_summary_selection = stems.copy()

    st.session_state.generated_csvs.clear()
    st.session_state.generated_meta = {}

    existing_overrides = {
        stem: dict(values)
        for stem, values in st.session_state.get("pre_overrides", {}).items()
        if stem in valid_stems and isinstance(values, dict)
    }
    st.session_state.group_assignments = {stem: "Default" for stem in valid_stems}

    try:
        baseline_sep = float(st.session_state.get("Min peak separation", DEFAULT_MIN_SEPARATION))
    except Exception:
        baseline_sep = DEFAULT_MIN_SEPARATION

    summary_lookup: dict[str, dict[str, object]] = {}
    positions_cache: dict[str, dict[str, object]] = {}
    updated_overrides: dict[str, dict[str, object]] = {}
    for row in summary_df.itertuples(index=False):
        stem = _clean_stem_label(getattr(row, "stem", None))
        if not stem:
            continue

        overrides = dict(existing_overrides.get(stem, {}))

        row_peaks = parse_peak_positions(getattr(row, "peaks", None))
        row_valleys = parse_peak_positions(getattr(row, "valleys", None))
        summary_lookup[stem] = {
            "peaks": tuple(row_peaks),
            "valleys": tuple(row_valleys),
        }
        positions_cache[stem] = summary_lookup[stem]

        n_val = getattr(row, "n_peaks", None)
        n_int = _coerce_int(n_val)
        if n_int is not None:
            overrides["n_peaks"] = n_int

        bw_val = _coerce_float(getattr(row, "bandwidth", None))
        if bw_val is not None and math.isfinite(bw_val):
            overrides["bw"] = float(bw_val)

        prom_val = _coerce_float(getattr(row, "prominence", None))
        if prom_val is not None and math.isfinite(prom_val):
            overrides["prom"] = float(prom_val)

        sep_val = _coerce_float(getattr(row, "min_separation", None))
        if sep_val is None:
            sep_val = derive_min_separation(row_peaks, baseline_sep)
        if sep_val is not None:
            sep_clean = max(0.0, float(sep_val))
            if abs(sep_clean - baseline_sep) <= 1e-6:
                overrides.pop("min_separation", None)
            else:
                overrides["min_separation"] = sep_clean

        if overrides:
            updated_overrides[stem] = overrides
        elif stem in existing_overrides:
            updated_overrides[stem] = {}

    st.session_state.pre_overrides = updated_overrides
    st.session_state.cli_summary_lookup = summary_lookup
    st.session_state.cli_positions_cache = positions_cache
    st.session_state.cli_positions_pending = list(summary_lookup)
    st.session_state.cli_positions_fixed = set(summary_lookup)
    st.session_state.cli_group_mode = "none"
    st.session_state.cli_group_new_name = ""

    samples = [str(s) for s in summary_df.get("sample", []) if isinstance(s, str)]
    markers = [str(m) for m in summary_df.get("marker", []) if isinstance(m, str)]
    batches_raw = summary_df.get("batch")
    if batches_raw is not None:
        batches = []
        for b in batches_raw:
            batches.append(None if pd.isna(b) else b)
        st.session_state.sel_batches = sorted({b for b in batches})
    else:
        st.session_state.sel_batches = []

    st.session_state.sel_samples = sorted(set(samples))
    st.session_state.sel_markers = sorted(set(markers))

    keep = valid_stems
    for bucket in (
        "results",
        "fig_pngs",
        "params",
        "dirty",
        "dirty_reason",
        "results_raw",
        "results_raw_meta",
        "aligned_results",
        "aligned_fig_pngs",
    ):
        data = st.session_state.get(bucket)
        if isinstance(data, dict):
            if bucket == "fig_pngs":
                st.session_state[bucket] = {
                    fn: png for fn, png in data.items() if fn.split(".")[0] in keep
                }
            elif bucket == "aligned_fig_pngs":
                st.session_state[bucket] = {
                    fn: png for fn, png in data.items() if fn.rsplit("_aligned", 1)[0] in keep
                }
            else:
                st.session_state[bucket] = {k: v for k, v in data.items() if k in keep}

    for key in list(st.session_state.keys()):
        if key.endswith("__pk_list") or key.endswith("__vl_list"):
            stem = key.split("__", 1)[0]
            if stem not in keep:
                st.session_state.pop(key, None)

    st.session_state.mode_selector_target = "Whole dataset"


def _cli_assign_groups(
    subset: pd.DataFrame,
    *,
    mode: str,
    new_group: str | None,
) -> bool:
    """Apply grouping rules for imported CLI samples.

    Returns ``True`` when grouping was applied successfully. A ``False`` return
    indicates that the operation should be aborted (e.g. missing information).
    """

    if not isinstance(subset, pd.DataFrame) or subset.empty:
        return True

    assignments_changed = False
    align_flag_before = bool(st.session_state.get("align_group_markers"))
    raw_ridge_before = st.session_state.get("raw_ridge_png")
    aligned_ridge_before = st.session_state.get("aligned_ridge_png")
    return_state = True

    assignments = st.session_state.get("group_assignments")
    if not isinstance(assignments, dict):
        assignments = {}
        st.session_state.group_assignments = assignments

    overrides = st.session_state.get("group_overrides")
    if not isinstance(overrides, dict):
        overrides = {"Default": {}}
        st.session_state.group_overrides = overrides

    stems = [str(s) for s in subset.get("stem", []) if isinstance(s, str) and s]

    st.session_state.align_group_markers = False

    if mode == "align_sample":
        if "sample" not in subset.columns:
            st.warning("The imported summary does not contain a 'sample' column to align by.")
            return False

        any_grouped = False
        for stem, sample in zip(stems, subset["sample"].tolist()):
            if not stem:
                continue
            if isinstance(sample, str):
                group_name = sample.strip()
            else:
                group_name = ""
            if not group_name:
                continue
            overrides.setdefault(group_name, {})
            if assignments.get(stem) != group_name:
                assignments[stem] = group_name
                if stem in st.session_state.results:
                    _mark_sample_dirty(stem, "group")
                assignments_changed = True
            any_grouped = True

        if not any_grouped:
            st.warning("No sample names were available to build group assignments.")
            return False

        st.session_state.group_overrides = overrides
        return_state = True

    if mode == "marker_groups":
        if "marker" not in subset.columns:
            st.warning("The imported summary does not contain a 'marker' column to group by.")
            return False

        any_grouped = False
        markers = subset["marker"].tolist()
        for stem, marker in zip(stems, markers):
            if not stem:
                continue
            label = marker if isinstance(marker, str) else ""
            clean_label = _normalize_label_casefold(label) or _normalize_label(label)
            if not clean_label:
                continue
            group_name = str(clean_label)
            overrides.setdefault(group_name, {})
            if assignments.get(stem) != group_name:
                assignments[stem] = group_name
                if stem in st.session_state.results:
                    _mark_sample_dirty(stem, "group")
                assignments_changed = True
            any_grouped = True

        if not any_grouped:
            st.warning("No marker names were available to build group assignments.")
            return False

        st.session_state.group_overrides = overrides
        st.session_state.align_group_markers = True
        return_state = True

    if mode == "new_group":
        clean = (new_group or "").strip()
        if not clean:
            st.warning("Enter a group name before grouping the selected samples together.")
            return False

        overrides.setdefault(clean, {})
        for stem in stems:
            if assignments.get(stem) != clean:
                assignments[stem] = clean
                if stem in st.session_state.results:
                    _mark_sample_dirty(stem, "group")
                assignments_changed = True
        st.session_state.group_overrides = overrides

    if (
        assignments_changed
        or bool(st.session_state.get("align_group_markers")) != align_flag_before
        or raw_ridge_before is not None
        or aligned_ridge_before is not None
    ):
        _mark_raw_ridge_stale()
        st.session_state.aligned_ridge_png = None
        st.session_state.aligned_counts = None
        st.session_state.aligned_landmarks = None

    return return_state


def _queue_cli_samples(
    stems: list[str],
    *,
    group_mode: str,
    new_group: str | None,
) -> None:
    """Queue selected CLI samples for processing."""

    summary_df = st.session_state.get("cli_summary_df")
    expr_df = st.session_state.get("expr_df")
    meta_df = st.session_state.get("meta_df")

    if summary_df is None or expr_df is None or meta_df is None:
        st.error("Load expression, metadata, and summary files before queuing samples.")
        return

    if not stems:
        st.warning("Select at least one sample to load.")
        return

    subset = summary_df[summary_df["stem"].astype(str).isin(stems)]
    if subset.empty:
        st.warning("No matching rows were found in the imported summary.")
        return

    marker_lookup = _build_expr_marker_lookup(expr_df)
    marker_map: dict[str, object] = {}
    missing_raw: list[str] = []
    skip_reasons: dict[str, str] = {}
    use_batches = "batch" in meta_df.columns
    for raw_marker in subset.get("marker", []):
        marker_label = _normalize_label(raw_marker)
        if not marker_label:
            continue
        column_label = _resolve_expr_marker(marker_label, marker_lookup)
        if column_label is None:
            missing_raw.append(marker_label)
            continue
        resolved_marker = _normalize_label(column_label) or marker_label
        marker_map.setdefault(resolved_marker, column_label)
    for _, row in subset.iterrows():
        stem = str(row.get("stem"))
        sample = _normalize_label(row.get("sample"))
        marker_raw = row.get("marker")
        marker_label = _normalize_label(marker_raw)
        batch_raw = row.get("batch") if use_batches else None
        batch_clean = None if pd.isna(batch_raw) else batch_raw

        if not stem or not sample or not marker_label:
            continue

        column_label = _resolve_expr_marker(marker_label, marker_lookup)
        if column_label is None:
            skip_reasons[stem] = (
                f"Marker '{marker_raw}' not found in the expression matrix"
            )
            continue

        indices = _cell_indices_for(meta_df, sample, batch_clean)
        if not indices:
            batch_note = "" if batch_clean is None else f" batch '{batch_clean}'"
            skip_reasons[stem] = (
                f"No cells for sample '{sample}'{batch_note} in the metadata"
            )

    markers = list(marker_map)
    missing_markers = sorted(set(missing_raw))
    samples = sorted({str(s) for s in subset.get("sample", []) if isinstance(s, str)})
    if "batch" in subset:
        raw_batches = subset["batch"].tolist()
        batches = sorted({None if pd.isna(b) else b for b in raw_batches})
    else:
        batches = []

    if missing_markers:
        st.warning(
            "The expression matrix is missing columns for the following markers: "
            + ", ".join(missing_markers)
            + ". They will be skipped."
        )

    if not markers:
        st.error(
            "None of the selected markers were found in the loaded expression matrix."
        )
        st.session_state.sel_markers = []
        st.session_state.sel_samples = samples
        st.session_state.sel_batches = batches
        return

    st.session_state.sel_markers = markers
    st.session_state.sel_samples = samples
    st.session_state.sel_batches = batches

    if not _cli_assign_groups(subset, mode=group_mode, new_group=new_group):
        return

    if st.session_state.get("cli_counts_native"):
        pending_raw = st.session_state.get("cli_positions_pending") or []
        if isinstance(pending_raw, list):
            pending_list = pending_raw.copy()
        else:
            pending_list = list(pending_raw)
        for stem in stems:
            if stem not in pending_list:
                pending_list.append(stem)
        st.session_state.cli_positions_pending = pending_list

    _sync_generated_counts(
        markers, samples, expr_df, meta_df, batches, summary_rows=subset.itertuples(index=False)
    )

    available_stems = {stem for stem, _ in st.session_state.generated_csvs}
    missing_stems = sorted({stem for stem in stems if stem not in available_stems})
    if missing_stems:
        preview = ", ".join(missing_stems[:5])
        suffix = "…" if len(missing_stems) > 5 else ""
        detail_parts: list[str] = []
        for stem in missing_stems[:3]:
            reason = skip_reasons.get(stem)
            if reason:
                detail_parts.append(f"{stem} ({reason})")
            else:
                detail_parts.append(stem)
        detail = "; ".join(detail_parts)
        extra = "" if detail else preview
        st.warning(
            "Skipped "
            f"{len(missing_stems)} sample(s) because counts could not be generated: "
            f"{detail or extra}{suffix}."
        )

    chosen = set(stems)
    files: list[io.BytesIO] = []
    for stem, bio in st.session_state.generated_csvs:
        if stem in chosen:
            bio.seek(0)
            files.append(bio)

    if not files:
        st.warning("Unable to locate generated counts for the selected samples.")
        return

    st.session_state.pending = files
    st.session_state.total_todo = len(files)
    st.session_state.run_active = True
    st.session_state.paused = False
    st.session_state.raw_ridge_png = None
    st.session_state.mode_selector_target = "Whole dataset"
    st.session_state.cli_summary_selection = stems
    st.rerun()


def _render_cli_import_section() -> None:
    """UI for importing CLI batch outputs and queueing samples."""

    st.markdown("---")
    with st.expander("Import CLI batch outputs", expanded=False):
        st.caption(
            "Upload the `summary.csv`, `expression_matrix_combined.csv`, and "
            "`cell_metadata_combined.csv` files generated by the CLI to review "
            "results and re-run specific samples."
        )

        col_summary, col_expr, col_meta = st.columns(3)
        summary_file = col_summary.file_uploader(
            "summary.csv", type=["csv"], key="cli_summary_upload"
        )
        expr_file = col_expr.file_uploader(
            "expression_matrix_combined.csv",
            type=["csv"],
            key="cli_expr_upload",
        )
        meta_file = col_meta.file_uploader(
            "cell_metadata_combined.csv",
            type=["csv"],
            key="cli_meta_upload",
        )

        load_col, clear_col = st.columns([3, 1])
        with load_col:
            if st.button("Load files", disabled=not (summary_file and expr_file and meta_file)):
                try:
                    summary_df = pd.read_csv(summary_file)
                    expr_df = pd.read_csv(expr_file, low_memory=False)
                    meta_df = pd.read_csv(meta_file, low_memory=False)
                except Exception as exc:
                    st.error(f"Failed to read uploaded files: {exc}")
                else:
                    _load_cli_import(
                        summary_df,
                        expr_df,
                        meta_df,
                        summary_name=summary_file.name,
                        expr_name=expr_file.name,
                        meta_name=meta_file.name,
                    )
                    st.rerun()

        with clear_col:
            if st.button("Clear import", disabled=st.session_state.cli_summary_df is None):
                _clear_cli_import()
                st.rerun()

        summary_df = st.session_state.get("cli_summary_df")
        if summary_df is None:
            return

        status = st.session_state.get("cli_import_status")
        if status:
            st.success(status)

        filter_text = st.text_input(
            "Filter imported rows",
            value=st.session_state.get("cli_filter_text", ""),
            key="cli_filter_input",
            placeholder="Search by stem, sample, or marker",
        )
        st.session_state.cli_filter_text = filter_text

        view_df = summary_df.copy()
        all_options = list(
            dict.fromkeys(
                str(stem)
                for stem in summary_df.get("stem", [])
                if isinstance(stem, str) and stem
            )
        )
        if filter_text:
            def _match(col: str) -> pd.Series:
                if col not in view_df.columns:
                    return pd.Series(False, index=view_df.index)
                return view_df[col].astype(str).str.contains(filter_text, case=False, na=False)

            mask = (
                view_df["stem"].astype(str).str.contains(filter_text, case=False, na=False)
                | _match("sample")
                | _match("marker")
            )
            view_df = view_df[mask]

        display_cols = [
            c
            for c in ["stem", "sample", "marker", "batch", "n_peaks", "peaks", "valleys", "quality"]
            if c in view_df.columns
        ]
        if display_cols:
            st.dataframe(view_df[display_cols], use_container_width=True, height=240)

        options = view_df["stem"].astype(str).tolist()
        defaults = [s for s in st.session_state.get("cli_summary_selection", []) if s in options]
        selection = st.multiselect(
            "Samples to queue",
            options,
            default=defaults,
            key="cli_summary_select",
            help="Choose the samples to re-run in the Streamlit editor.",
        )

        st.session_state.cli_summary_selection = selection

        group_choices = ["none", "align_sample", "marker_groups", "new_group"]
        group_labels = {
            "none": "No automatic grouping",
            "align_sample": "Align using sample column",
            "marker_groups": "Group by marker name",
            "new_group": "Group selected together",
        }
        default_mode = st.session_state.get("cli_group_mode", "none")
        try:
            default_index = group_choices.index(default_mode)
        except ValueError:
            default_index = 0

        group_mode = st.radio(
            "Post-import grouping",
            group_choices,
            index=default_index,
            key="cli_group_mode",
            format_func=lambda value: group_labels.get(value, value),
            help="Automatically assign selected samples to groups after loading.",
        )

        new_group_name = st.text_input(
            "Group name for selected samples",
            value=st.session_state.get("cli_group_new_name", ""),
            key="cli_group_new_name",
            disabled=group_mode != "new_group",
            help="Provide the group name used when grouping the selected samples together.",
        )

        buttons_col1, buttons_col2 = st.columns([1, 1])
        with buttons_col1:
            if st.button("Load selected samples", disabled=not selection):
                _queue_cli_samples(
                    selection,
                    group_mode=group_mode,
                    new_group=new_group_name,
                )
        with buttons_col2:
            if st.button("Load all samples", disabled=view_df.empty):
                _queue_cli_samples(
                    all_options,
                    group_mode=group_mode,
                    new_group=new_group_name,
                )
# ───────────────────────── helper: (re)plot a dataset ───────────────────────

def _plot_png_fixed(stem, xs, ys, peaks, valleys,
                    xlim, ylim) -> bytes:
    """Same as _plot_png but with externally supplied axes limits."""
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
    ax.plot(xs, ys, color="orange")
    ax.fill_between(xs, 0, ys, color="#FFA50088")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        try:
            y0, y1 = ylim
        except (TypeError, ValueError):
            y0, y1 = 0.0, float(ylim)
        ax.set_ylim(y0, y1)
    for p in peaks:   ax.axvline(p, color="black", ls="--", lw=1)
    for v in valleys: ax.axvline(v, color="grey",  ls=":",  lw=1)
    ax.set_yticks([]); ax.set_title(stem, fontsize=8)
    fig.tight_layout()
    out = fig_to_png(fig); plt.close(fig)
    return out

def _plot_png(stem: str, xs: np.ndarray, ys: np.ndarray,
              peaks: list[float], valleys: list[float]) -> bytes:
    """Return PNG bytes of the KDE plot with current peak/valley markers."""
    x_min = float(xs.min())
    x_max = float(xs.max())
    pad   = 0.05 * (x_max - x_min)
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
    ax.plot(xs, ys, color="skyblue"); ax.fill_between(xs, 0, ys, color="#87CEEB88")
    ax.set_xlim(x_min - pad, x_max + pad)
    for p in peaks:   ax.axvline(p, color="red",   ls="--", lw=1)
    for v in valleys: ax.axvline(v, color="green", ls=":",  lw=1)
    ax.set_xlabel("Arcsinh counts"); ax.set_ylabel("Density")
    ax.set_title(stem, fontsize=9); fig.tight_layout()
    out = fig_to_png(fig); plt.close(fig)
    return out

# ───────────────────── helper: inline editor + plot ─────────────────────────

def _manual_editor(stem: str):
    info  = st.session_state.results[stem]
    xs    = np.asarray(info["xs"]); ys = np.asarray(info["ys"])
    xmin0, xmax0 = float(xs.min()), float(xs.max())

    pk_key, vl_key = f"{stem}__pk_list", f"{stem}__vl_list"
    if pk_key not in st.session_state:
        st.session_state[pk_key] = info["peaks"].copy()
    if vl_key not in st.session_state:
        st.session_state[vl_key] = info["valleys"].copy()

    pk_list = st.session_state[pk_key]
    vl_list = st.session_state[vl_key]

    xmin = float(min([xmin0] + pk_list + vl_list))
    xmax = float(max([xmax0] + pk_list + vl_list))

    ### one single column for sliders + buttons
    colL, colR = st.columns([3, 5])

    # ---------- sliders & delete buttons (NO nested columns!) ----------
    with colL:
        st.subheader("Peaks")
        i = 0
        while i < len(pk_list):
            pk_list[i] = st.slider(
                f"Peak #{i+1}", xmin, xmax, float(pk_list[i]), 0.01,
                key=f"{stem}_pk_slider_{i}",
                help="Adjust the position of this peak."
            )
            if st.button(f"Delete peak #{i+1}", key=f"{stem}_pk_del_{i}"):
                pk_list.pop(i)
                st.session_state.raw_ridge_png = None
                st.rerun()
            else:
                i += 1

        if st.button("Add peak", key=f"{stem}_add_pk"):
            pk_list.append((xmin + xmax) / 2)
            st.rerun()

        st.divider()
        st.subheader("Valleys")
        i = 0
        while i < len(vl_list):
            vl_list[i] = st.slider(
                f"Valley #{i+1}", xmin, xmax, float(vl_list[i]), 0.01,
                key=f"{stem}_vl_slider_{i}",
                help="Adjust the position of this valley."
            )
            if st.button(f"Delete valley #{i+1}", key=f"{stem}_vl_del_{i}"):
                vl_list.pop(i)
                st.rerun()
            else:
                i += 1

        if st.button("Add valley", key=f"{stem}_add_vl"):
            vl_list.append((xmin + xmax) / 2)
            st.rerun()

        # keep sorted / unique
        pk_list[:] = sorted(dict.fromkeys(pk_list))
        vl_list[:] = sorted(dict.fromkeys(vl_list))

        # push back to main state
        st.session_state.results[stem]["peaks"]   = pk_list
        st.session_state.results[stem]["valleys"] = vl_list
        # st.session_state.dirty[stem] = False

    # ---------- live plot -------------------------------------------------
    with colR:
        png = _plot_png(stem, xs, ys, pk_list, vl_list)
        st.session_state.fig_pngs[f"{stem}.png"] = png
        st.image(png, use_container_width=True)

    # sync back to main result dict after UI elements update ----------------
    #st.session_state.results[stem]["peaks"] = pk_list.copy()
    #st.session_state.results[stem]["valleys"] = vl_list.copy()
    # st.session_state.dirty[stem] = False

    apply_key = f"{stem}_apply_edits"
    if st.button("Apply changes", key=apply_key):
        _release_cli_positions(stem)
        st.session_state.results[stem]["peaks"]   = pk_list.copy()
        st.session_state.results[stem]["valleys"] = vl_list.copy()
        _mark_raw_ridge_stale()                    # ← rebuild lazily
        st.rerun()


def _render_group_controls(
    stems: list[str], *,
    bw_label: str,
    prom_mode: str,
    prom_default: float | None,
    run_defaults: dict[str, object],
) -> None:
    """Render the group-assignment UI for the given stems."""

    stems_unique = [stem for stem in dict.fromkeys(stems) if stem]

    # Drop any lingering assignments for stems no longer in scope.
    for stem in list(st.session_state.group_assignments):
        if stem not in stems_unique:
            st.session_state.group_assignments.pop(stem, None)

    if not stems_unique:
        return

    previous_assignments = dict(st.session_state.group_assignments)
    st.session_state.group_overrides.setdefault("Default", {})

    st.markdown("---\n### Sample groups (optional)")

    marker_map = {stem: _stem_marker(stem) for stem in stems_unique}
    has_marker_groups = any(marker for marker in marker_map.values())

    mode_options = ["Manual selection"]
    if has_marker_groups:
        mode_options.append("By marker name")

    prev_mode = st.session_state.get("group_assignments_mode", "Manual selection")
    if prev_mode not in mode_options:
        prev_mode = "Manual selection"

    selection_index = mode_options.index(prev_mode)
    selected_mode = st.radio(
        "Grouping mode",
        mode_options,
        index=selection_index,
        key="group_assignments_mode",
        help="Choose whether to assign samples manually or group them automatically by marker name.",
    )
    auto_mode = selected_mode == "By marker name"

    col_name, col_add = st.columns([3, 1])

    if st.session_state.get("group_new_name_reset", False):
        st.session_state.group_new_name = ""
        st.session_state.group_new_name_reset = False

    new_group_name = col_name.text_input(
        "Create new group",
        key="group_new_name",
        placeholder="e.g. Treatment A",
        help="Groups let you tune detector settings once and apply them to multiple samples.",
        disabled=auto_mode,
    )
    if col_add.button("Add group", key="group_add_btn", disabled=auto_mode):
        clean = new_group_name.strip()
        if clean:
            if clean not in st.session_state.group_overrides:
                st.session_state.group_overrides[clean] = {}
            st.session_state.group_new_name_reset = True
        st.rerun()

    if auto_mode:
        for marker in marker_map.values():
            if marker:
                st.session_state.group_overrides.setdefault(str(marker), {})

    group_names = sorted(st.session_state.group_overrides)
    valid_groups = set(group_names)

    current_assignments: dict[str, str] = {}
    auto_defaulted: list[str] = []
    for stem in stems_unique:
        if auto_mode:
            marker = marker_map.get(stem)
            if marker:
                group = str(marker)
            else:
                group = "Default"
                auto_defaulted.append(stem)
        else:
            group = st.session_state.group_assignments.get(stem, "Default")
        if group not in valid_groups:
            group = "Default"
        current_assignments[stem] = group

    if auto_mode:
        st.info(
            "Samples are automatically grouped by their marker names. Manual assignment controls are disabled."
        )
        if auto_defaulted:
            st.caption(
                "Markers not found for: " + ", ".join(auto_defaulted) + " (kept in Default group)."
            )

    if len(group_names) > 1:
        st.caption(
            "Select the samples that belong to each group. Samples left unselected stay in the Default group."
        )

    group_selections: dict[str, list[str]] = {}
    for group_name in group_names:
        if group_name == "Default":
            continue
        selection_key = f"group_samples__{_keyify(group_name)}"
        default_selection = [
            stem for stem, grp in current_assignments.items() if grp == group_name
        ]
        if auto_mode:
            st.session_state[selection_key] = default_selection
            default_values = default_selection
        else:
            existing_selection = st.session_state.get(selection_key)
            if isinstance(existing_selection, list):
                sanitized_existing = [stem for stem in existing_selection if stem in stems_unique]
                if sanitized_existing != existing_selection:
                    st.session_state[selection_key] = sanitized_existing
                default_values = sanitized_existing
            else:
                default_values = default_selection

        selection = st.multiselect(
            f"Samples in {group_name}",
            options=stems_unique,
            default=default_values,
            key=selection_key,
            help="Choose the samples that should inherit this group's overrides.",
            disabled=auto_mode,
        )
        group_selections[group_name] = selection

    new_assignments = {stem: "Default" for stem in stems_unique}
    duplicate_samples: dict[str, list[str]] = {}

    for group_name in group_names:
        if group_name == "Default":
            continue
        for stem in group_selections.get(group_name, []):
            if stem not in new_assignments:
                continue
            previous_group = new_assignments.get(stem, "Default")
            if previous_group != "Default" and previous_group != group_name:
                duplicate_samples.setdefault(stem, [previous_group]).append(group_name)
            new_assignments[stem] = group_name

    changed_assignments = [
        stem
        for stem, group in new_assignments.items()
        if previous_assignments.get(stem, "Default") != group
    ]

    st.session_state.group_assignments = new_assignments
    for stem in changed_assignments:
        _mark_sample_dirty(stem, "group")

    if duplicate_samples:
        dup_messages = [
            f"{sample}: {', '.join(groups)}" for sample, groups in duplicate_samples.items()
        ]
        st.warning(
            "Some samples were assigned to multiple groups. The last group in the list above will be used for each sample: "
            + "; ".join(dup_messages)
        )

    default_samples = [stem for stem, group in new_assignments.items() if group == "Default"]
    if default_samples and len(group_names) > 1:
        st.caption(
            "Samples currently using the Default group: " + ", ".join(default_samples)
        )
    elif len(group_names) == 1:
        st.caption(
            "All samples currently use the Default group. Create a group above to assign samples."
        )

    if any(g != "Default" for g in group_names):
        st.caption("Group-level overrides apply before per-sample overrides.")

    for group_name in group_names:
        if group_name == "Default":
            continue

        with st.expander(f"Group: {group_name}", expanded=False):
            if st.button("Remove group", key=f"group_remove__{group_name}", disabled=auto_mode):
                affected = [
                    stem
                    for stem, grp in st.session_state.group_assignments.items()
                    if grp == group_name
                ]
                st.session_state.group_overrides.pop(group_name, None)
                for stem in affected:
                    st.session_state.group_assignments[stem] = "Default"
                    _mark_sample_dirty(stem, "group")
                group_selection_key = f"group_samples__{_keyify(group_name)}"
                if group_selection_key in st.session_state:
                    st.session_state.pop(group_selection_key)
                st.rerun()

            prev_group = dict(st.session_state.group_overrides.get(group_name, {}))
            group_overrides = _render_override_controls(
                f"group_{group_name}",
                prev=prev_group,
                bw_label=bw_label,
                prom_mode=prom_mode,
                prom_default=prom_default,
                run_defaults=run_defaults,
            )
            st.session_state.group_overrides[group_name] = group_overrides

            if group_overrides != prev_group:
                for stem, grp in st.session_state.group_assignments.items():
                    if grp == group_name:
                        _mark_sample_dirty(stem, "group")

            summary = _summarize_overrides(group_overrides)
            if summary:
                st.caption("Overrides: " + ", ".join(summary))
            else:
                st.caption("Using run setting for this group.")


def _render_pre_run_overrides(
    stems: list[str], *,
    bw_label: str,
    prom_mode: str,
    prom_default: float | None,
    run_defaults: dict[str, object],
) -> None:
    """Allow users to configure per-sample overrides before the first run."""

    st.markdown("---\n### Per-sample overrides (before run)")

    if not stems:
        st.caption(
            "Select CSV files in the sidebar to configure per-sample overrides."
        )
        return

    prom_label = (
        f"Manual ({prom_default:.2f})"
        if prom_mode == "Manual" and prom_default is not None
        else "GPT automatic"
    )

    st.caption(
        "Override the peak count, bandwidth, prominence, or advanced options for "
        "specific files before running the detector. Leave everything on “Use run "
        "setting” to inherit the global configuration or group defaults."
    )

    for stem in stems:
        overrides_active = bool(st.session_state.pre_overrides.get(stem))
        label = stem + (" ✳️" if overrides_active else "")

        with st.expander(label, expanded=False):
            if stem in st.session_state.results:
                st.info(
                    "This sample has already been processed. Use the **Parameters** "
                    "tab below to adjust it and re-run."
                )
                continue

            prev = {
                k: v
                for k, v in st.session_state.pre_overrides.get(stem, {}).items()
                if v is not None
            }

            group_name = st.session_state.group_assignments.get(stem, "Default")
            group_overrides = st.session_state.group_overrides.get(group_name, {})
            if group_name != "Default" or group_overrides:
                group_summary = _summarize_overrides(group_overrides)
                if group_summary:
                    st.caption(
                        f"Group **{group_name}** overrides: " + ", ".join(group_summary)
                    )
                else:
                    st.caption(f"Group **{group_name}** uses run settings.")

            overrides = _render_override_controls(
                stem,
                prev=prev,
                bw_label=bw_label,
                prom_mode=prom_mode,
                prom_default=prom_default,
                run_defaults=run_defaults,
            )

            if overrides:
                _update_pre_override(stem, overrides)
            elif prev:
                _update_pre_override(stem, {})

            summary = _summarize_overrides(overrides)
            if summary:
                st.caption("Overrides: " + ", ".join(summary))
            else:
                st.caption(
                    "Using run setting for peaks, bandwidth "
                    f"({bw_label}), prominence ({prom_label}), and advanced options."
                )


def _combined_overrides(stem: str, base: dict[str, object] | None = None) -> dict[str, object]:
    """Merge group-level overrides with per-sample overrides for a stem."""

    combined: dict[str, object] = {}
    group = st.session_state.group_assignments.get(stem)
    if group:
        for key, value in st.session_state.group_overrides.get(group, {}).items():
            if value is not None:
                combined[key] = value

    if base:
        for key, value in base.items():
            if value is not None:
                combined[key] = value

    return combined


def _coerce_int(value) -> int | None:
    """Best-effort conversion of a value to ``int`` (None if not possible)."""

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not pd.isna(value):
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


def _coerce_float(value) -> float | None:
    """Best-effort conversion of a value to ``float`` (None if not possible)."""

    if isinstance(value, (int, float, np.floating)):
        return float(value) if not pd.isna(value) else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _prepare_batch_overrides(stems: list[str]) -> dict[str, dict[str, object]]:
    """Collect combined overrides for each stem keyed by ``stems`` bucket."""

    overrides: dict[str, dict[str, object]] = {}
    for stem in stems:
        per_sample_over = dict(st.session_state.pre_overrides.get(stem, {}))
        reason_raw = st.session_state.dirty_reason.get(stem)
        reasons = set(reason_raw) if reason_raw else set()

        if st.session_state.dirty.get(stem, False) and (not reasons or "manual" in reasons):
            for key, value in st.session_state.params.get(stem, {}).items():
                if value is not None:
                    per_sample_over[key] = value

        overrides[stem] = _combined_overrides(stem, per_sample_over)

    return {"stems": overrides}


def _prepare_sample_inputs(
    files: list[io.BytesIO],
    arcsinh_sig: tuple[bool, float, float, float],
    header_row: int,
    skip_rows: int,
) -> list[SampleInput]:
    samples: list[SampleInput] = []

    for order, f in enumerate(files):
        f.seek(0)
        stem = Path(f.name).stem
        marker = getattr(f, "marker", None)
        cached_cnts = st.session_state.results_raw.get(stem)
        meta_entry_raw = st.session_state.results_raw_meta.get(stem) or {}
        if isinstance(meta_entry_raw, tuple):
            cached_sig = meta_entry_raw
            meta_entry = {"arcsinh": meta_entry_raw}
        else:
            meta_entry = dict(meta_entry_raw)
            cached_sig = meta_entry.get("arcsinh")

        if cached_cnts is not None and cached_sig == arcsinh_sig:
            cnts = cached_cnts
        else:
            cnts, meta = read_counts(f, header_row, skip_rows)
            if arcsinh_sig[0] and not getattr(f, "arcsinh", False):
                cnts = arcsinh_transform(
                    cnts,
                    a=arcsinh_sig[1],
                    b=arcsinh_sig[2],
                    c=arcsinh_sig[3],
                )
            meta_entry = {
                "arcsinh": arcsinh_sig,
                "protein_name": meta.get("protein_name"),
                "source_name": meta.get("source_name") or getattr(f, "name", None),
                "marker": meta.get("marker") or meta.get("protein_name"),
                "batch": meta.get("batch"),
            }
            st.session_state.results_raw[stem] = cnts
            st.session_state.results_raw_meta[stem] = meta_entry

        dataset_meta = st.session_state.generated_meta.get(stem) or {}

        metadata = {
            "sample": dataset_meta.get("sample") or stem,
            "marker": marker or dataset_meta.get("marker") or meta_entry.get("marker"),
            "batch": getattr(f, "batch", None) or dataset_meta.get("batch"),
            "protein_name": meta_entry.get("protein_name"),
        }

        samples.append(
            SampleInput(
                stem=stem,
                counts=np.asarray(cnts, float),
                metadata=metadata,
                arcsinh_signature=arcsinh_sig,
                source_name=getattr(f, "name", None),
                order=order,
                cell_indices=getattr(f, "cell_indices", None),
            )
        )

    return samples


def _apply_sample_result(res: SampleResult, *, aligned: bool = False) -> None:
    stem = res.stem
    xs = np.asarray(res.aligned_density[0] if aligned and res.aligned_density else res.xs, float)
    ys = np.asarray(res.aligned_density[1] if aligned and res.aligned_density else res.ys, float)
    peaks = (
        list(map(float, res.aligned_peaks))
        if aligned and res.aligned_peaks is not None
        else list(map(float, res.peaks))
    )
    valleys = (
        list(map(float, res.aligned_valleys))
        if aligned and res.aligned_valleys is not None
        else list(map(float, res.valleys))
    )

    quality = float(res.quality)
    cli_applied = False
    if not aligned:
        peaks, valleys, cli_applied = _apply_cli_positions(stem, peaks, valleys)
        if cli_applied:
            quality = float(stain_quality(res.counts, peaks, valleys))

    meta_entry = {
        "arcsinh": res.arcsinh_signature,
        "protein_name": res.metadata.get("protein_name"),
        "source_name": res.source_name,
        "marker": res.metadata.get("marker"),
        "batch": res.metadata.get("batch"),
    }

    st.session_state.results_raw[stem] = res.counts
    st.session_state.results_raw_meta[stem] = meta_entry

    target = st.session_state.aligned_results if aligned else st.session_state.results
    target[stem] = {
        "peaks": peaks,
        "valleys": valleys,
        "quality": quality,
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "marker": res.metadata.get("marker"),
    }

    png_key = f"{stem}_aligned.png" if aligned else f"{stem}.png"
    plotter = _plot_png_fixed if aligned else _plot_png
    if xs.size and ys.size:
        if aligned:
            xlim = (float(xs.min()), float(xs.max())) if xs.size else None
            ylim = float(np.nanmax(ys)) if ys.size else None
            st.session_state.aligned_fig_pngs[png_key] = plotter(
                stem if aligned else stem,
                xs,
                ys,
                peaks,
                valleys,
                xlim,
                (0, ylim) if ylim is not None else None,
            )
        else:
            st.session_state.fig_pngs[png_key] = plotter(stem, xs, ys, peaks, valleys)

    params = {
        "bw": res.params.get("bw"),
        "prom": res.params.get("prom"),
        "n_peaks": res.params.get("n_peaks"),
    }
    if cli_applied and peaks:
        params["n_peaks"] = len(peaks)
    st.session_state.params[stem] = params
    st.session_state.dirty[stem] = False
    st.session_state.dirty_reason.pop(stem, None)
    st.session_state[f"{stem}__pk_list"] = peaks.copy()
    st.session_state[f"{stem}__vl_list"] = valleys.copy()
    _mark_raw_ridge_stale()


def _apply_aligned_outputs(batch: BatchResults) -> None:
    if not batch.alignment_mode:
        st.session_state.aligned_counts = None
        st.session_state.aligned_landmarks = None
        st.session_state.aligned_results = {}
        st.session_state.aligned_fig_pngs = {}
        st.session_state.aligned_ridge_png = None
        return

    st.session_state.aligned_landmarks = batch.aligned_landmarks
    st.session_state.aligned_results = {}
    st.session_state.aligned_fig_pngs = {}
    st.session_state.aligned_counts = {}
    st.session_state.aligned_ridge_png = None

    for res in batch.samples:
        if res.aligned_counts is not None:
            st.session_state.aligned_counts[res.stem] = res.aligned_counts
        _apply_sample_result(res, aligned=True)


def _run_batch_async(
    run_id: int,
    samples: list[SampleInput],
    options: BatchOptions,
    overrides: dict[str, dict[str, object]],
    gpt_client: OpenAI | None,
    queue_obj: Queue,
) -> None:
    class QueueProgress:
        def start(self, total: int) -> None:
            queue_obj.put({"type": "start", "total": total, "run_id": run_id})

        def advance(self, stem: str, completed: int, total: int) -> None:
            queue_obj.put(
                {
                    "type": "progress",
                    "stem": stem,
                    "completed": completed,
                    "total": total,
                    "run_id": run_id,
                }
            )

        def finish(self, completed: int, total: int, interrupted: bool) -> None:
            queue_obj.put(
                {
                    "type": "finish",
                    "completed": completed,
                    "total": total,
                    "interrupted": interrupted,
                    "run_id": run_id,
                }
            )

        def result(self, sample: SampleResult) -> None:
            queue_obj.put({"type": "result", "sample": sample, "run_id": run_id})

    try:
        batch = run_batch(
            samples,
            options,
            overrides=overrides,
            gpt_client=gpt_client,
            progress=QueueProgress(),
        )
        queue_obj.put({"type": "batch_complete", "batch": batch, "run_id": run_id})
    except Exception as exc:  # pragma: no cover - depends on runtime/API
        queue_obj.put({"type": "error", "error": exc, "run_id": run_id})


def _process_batch_events() -> None:
    queue_obj: Queue | None = st.session_state.get("batch_queue")
    if queue_obj is None:
        return

    while True:
        try:
            event = queue_obj.get_nowait()
        except Empty:
            break

        if event.get("run_id") != st.session_state.batch_run_id:
            continue

        etype = event.get("type")
        if etype == "start":
            total = int(event.get("total", 0))
            st.session_state.batch_progress = {"completed": 0, "total": total}
            st.session_state.total_todo = total
        elif etype == "progress":
            completed = int(event.get("completed", 0))
            total = int(event.get("total", 0))
            st.session_state.batch_progress = {"completed": completed, "total": total}
        elif etype == "result":
            res = event.get("sample")
            if isinstance(res, SampleResult):
                _apply_sample_result(res)
                st.session_state.batch_progress["completed"] = max(
                    st.session_state.batch_progress.get("completed", 0),
                    len(st.session_state.results),
                )
        elif etype == "batch_complete":
            batch = event.get("batch")
            if isinstance(batch, BatchResults):
                for res in batch.samples:
                    _apply_sample_result(res)
                _apply_aligned_outputs(batch)
                st.session_state.run_active = False
                st.session_state.pending = []
                st.session_state.batch_thread = None
                st.session_state.batch_progress["completed"] = len(batch.samples)
                st.session_state.batch_progress["total"] = max(
                    st.session_state.batch_progress.get("total", len(batch.samples)),
                    len(batch.samples),
                )
                if st.session_state.results and st.session_state.get("raw_ridge_png") is None:
                    _refresh_raw_ridge()
                if batch.interrupted:
                    st.warning(
                        "Batch interrupted before completing all samples. Partial results are available."
                    )
                else:
                    st.success("All files processed!")
        elif etype == "finish":
            interrupted = bool(event.get("interrupted", False))
            if interrupted:
                st.session_state.run_active = False
                st.session_state.batch_thread = None
        elif etype == "error":
            exc = event.get("error")
            st.session_state.batch_error = exc
            st.session_state.run_active = False
            st.session_state.batch_thread = None


def _schedule_batch_rerun(poll_interval: float = 0.5) -> None:
    """Keep the Streamlit script polling for batch events while work runs."""

    thread: Thread | None = st.session_state.get("batch_thread")
    queue_obj: Queue | None = st.session_state.get("batch_queue")
    queue_pending = bool(queue_obj and not queue_obj.empty())
    run_active = bool(st.session_state.get("run_active"))

    if not (run_active or queue_pending):
        return

    if thread and thread.is_alive():
        time.sleep(max(poll_interval, 0.05))
    else:
        # Thread already finished or never started, but pending events might
        # still need to be drained.
        time.sleep(0.05)
    st.rerun()


def _format_batch_label(batch_val) -> str:
    """Return a human-friendly label for batch values, handling NaNs."""
    if pd.isna(batch_val):
        return "—"
    return str(batch_val)


def _enumerate_dataset_entries(
    sel_markers: list[str],
    sel_samples: list[str],
    meta_df: pd.DataFrame,
    sel_batches: list[str | None] | None = None,
) -> list[dict[str, object]]:
    """List dataset stems + metadata for the current marker/sample selection."""

    if not sel_markers or not sel_samples or meta_df is None:
        return []

    use_batches = "batch" in meta_df.columns
    sel_batch_set = set(sel_batches or [])
    combos: list[dict[str, object]] = []

    for sample in sel_samples:
        mask = meta_df["sample"].eq(sample)
        if not mask.any():
            continue

        batch_values: list[object]
        if use_batches:
            batches = meta_df.loc[mask, "batch"].unique().tolist()
            normed: list[object] = []
            for raw in batches:
                clean = None if pd.isna(raw) else raw
                if sel_batch_set and clean not in sel_batch_set:
                    continue
                normed.append(clean)
            if not normed:
                continue
            batch_values = normed
        else:
            batch_values = [None]

        for batch in batch_values:
            for marker in sel_markers:
                if batch is None:
                    stem = f"{sample}_{marker}_raw_counts"
                else:
                    stem = f"{sample}_{batch}_{marker}_raw_counts"
                combos.append(
                    {
                        "stem": stem,
                        "sample": sample,
                        "marker": marker,
                        "batch": batch,
                        "batch_label": _format_batch_label(batch),
                    }
                )

    return combos


def _render_dataset_overrides(
    combos: list[dict[str, object]], *,
    bw_label: str,
    prom_mode: str,
    prom_default: float | None,
) -> None:
    """UI for configuring overrides when working with full datasets."""

    st.markdown("---\n### Per-sample overrides (dataset)")

    if not combos:
        st.caption(
            "Select at least one sample and marker to configure per-sample overrides."
        )
        return

    prom_label = (
        f"Manual ({prom_default:.2f})"
        if prom_mode == "Manual" and prom_default is not None
        else "GPT automatic"
    )

    st.caption(
        "Use the table below to override the peak count, bandwidth, prominence, "
        "or advanced detection options for specific sample/marker combinations. "
        "Leave a cell blank to inherit the run-wide setting."
    )

    all_samples = sorted({entry["sample"] for entry in combos})
    search_term = st.text_input(
        "Filter samples", key="dataset_override_search", placeholder="Type to filter sample names"
    )
    if search_term:
        lowered = search_term.strip().lower()
        all_samples = [s for s in all_samples if lowered in s.lower()]

    if not all_samples:
        st.info("No samples match this filter.")
        return

    selected_sample = st.selectbox(
        "Sample to edit",
        all_samples,
        key="dataset_override_selected_sample",
        help="Choose which sample's combinations to edit.",
    )

    sample_combos = [c for c in combos if c["sample"] == selected_sample]
    if not sample_combos:
        st.info("No dataset entries for this sample.")
        return

    marker_options = sorted({c["marker"] for c in sample_combos})
    marker_key = f"dataset_override_markers__{selected_sample}"
    default_markers = st.session_state.get(marker_key)
    if default_markers:
        default_markers = [m for m in default_markers if m in marker_options]
    else:
        default_markers = marker_options
    marker_selection = st.multiselect(
        "Markers",
        marker_options,
        default=default_markers,
        key=marker_key,
        help="Limit the table to specific markers (leave empty to show all).",
    )
    if not marker_selection:
        marker_selection = marker_options

    has_batches = any(c["batch"] is not None for c in sample_combos)
    allowed_batches: set[object]
    if has_batches:
        batch_label_map: dict[str, set[object]] = {}
        for c in sample_combos:
            label = c["batch_label"]
            batch_label_map.setdefault(label, set()).add(c["batch"])
        batch_options = list(batch_label_map.keys())
        batch_key = f"dataset_override_batches__{selected_sample}"
        default_batches = st.session_state.get(batch_key)
        if default_batches:
            default_batches = [b for b in default_batches if b in batch_options]
        else:
            default_batches = batch_options
        batch_selection = st.multiselect(
            "Batches",
            batch_options,
            default=default_batches,
            key=batch_key,
            help="Filter dataset entries by batch (leave empty to include all).",
        )
        if not batch_selection:
            batch_selection = batch_options
        allowed_batches = set()
        for label in batch_selection:
            allowed_batches.update(batch_label_map.get(label, set()))
    else:
        allowed_batches = {None}

    show_overrides_only = st.checkbox(
        "Show only combinations with overrides",
        value=False,
        key=f"dataset_override_only__{selected_sample}",
    )

    filtered = [
        c
        for c in sample_combos
        if c["marker"] in marker_selection
        and (not has_batches or c["batch"] in allowed_batches)
        and (not show_overrides_only or c["stem"] in st.session_state.pre_overrides)
    ]

    if not filtered:
        st.info("No dataset entries match the current filters.")
        return

    rows = []
    for entry in filtered:
        overrides = st.session_state.pre_overrides.get(entry["stem"], {})
        bw_val = overrides.get("bw", "")
        if isinstance(bw_val, (int, float)) and not np.isnan(bw_val):
            bw_txt = f"{bw_val}"
        elif bw_val:
            bw_txt = str(bw_val)
        else:
            bw_txt = ""
        turning_val = overrides.get("turning_points")
        if turning_val is True:
            turn_txt = "Yes"
        elif turning_val is False:
            turn_txt = "No"
        else:
            turn_txt = ""
        rows.append(
            {
                "stem": entry["stem"],
                "Marker": entry["marker"],
                "Batch": entry["batch_label"],
                "Peaks": overrides.get("n_peaks"),
                "Max peaks": overrides.get("max_peaks"),
                "Bandwidth": bw_txt,
                "Prominence": overrides.get("prom"),
                "Min width": overrides.get("min_width"),
                "Curvature": overrides.get("curvature"),
                "Turning points": turn_txt,
                "Min separation": overrides.get("min_separation"),
                "Max grid": overrides.get("max_grid"),
                "Valley drop": overrides.get("valley_drop"),
                "First valley": overrides.get("first_valley"),
            }
        )

    df_table = pd.DataFrame(rows).set_index("stem")

    data_editor = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)

    if data_editor is None:
        st.info(
            "Streamlit is too old to edit overrides inline. Upgrade Streamlit to enable "
            "editing; showing a read-only table instead."
        )
        edited = df_table
        st.dataframe(df_table, use_container_width=True)
    else:
        edited = data_editor(
            df_table,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            column_config={
                "Marker": st.column_config.TextColumn("Marker", disabled=True),
                "Batch": st.column_config.TextColumn("Batch", disabled=True),
                "Peaks": st.column_config.NumberColumn(
                    "Peaks", min_value=1, max_value=6, step=1,
                    help="Override the expected peak count (leave blank for run setting).",
                ),
                "Max peaks": st.column_config.NumberColumn(
                    "Max peaks", min_value=1, max_value=6, step=1,
                    help="Override the GPT automatic peak cap (leave blank for run setting).",
                ),
                "Bandwidth": st.column_config.TextColumn(
                    "Bandwidth",
                    help="Enter a preset (scott, silverman, 0.5, …) or numeric bandwidth.",
                ),
                "Prominence": st.column_config.NumberColumn(
                    "Prominence",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.2f",
                    help="Minimum relative drop from peak to valley.",
                ),
                "Min width": st.column_config.NumberColumn(
                    "Min peak width", min_value=0, max_value=6, step=1,
                    help="Override the minimum width for detected peaks.",
                ),
                "Curvature": st.column_config.NumberColumn(
                    "Curvature threshold",
                    min_value=0.0,
                    max_value=0.005,
                    step=0.0001,
                    format="%.4f",
                    help="Override the curvature requirement for peaks (0 disables).",
                ),
                "Turning points": st.column_config.SelectboxColumn(
                    "Treat turning points",
                    options=["", "Yes", "No"],
                    help="Treat concave-down turning points as peaks (Yes/No).",
                ),
                "Min separation": st.column_config.NumberColumn(
                    "Min peak separation",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    format="%.1f",
                    help="Override the minimum distance between peaks.",
                ),
                "Max grid": st.column_config.NumberColumn(
                    "Max KDE grid",
                    min_value=4_000,
                    max_value=40_000,
                    step=1_000,
                    help="Override the KDE grid resolution (higher is finer but slower).",
                ),
                "Valley drop": st.column_config.NumberColumn(
                    "Valley drop (%)",
                    min_value=1,
                    max_value=50,
                    step=1,
                    help="Override the required drop from peak to valley in percent.",
                ),
                "First valley": st.column_config.SelectboxColumn(
                    "First valley method",
                    options=["", "Slope change", "Valley drop"],
                    help="Override how to identify the first valley after a peak.",
                ),
            },
            key=f"dataset_override_editor__{selected_sample}",
        )

    for stem, row in edited.iterrows():
        overrides: dict[str, object] = {}

        peaks_val = row.get("Peaks")
        if peaks_val not in (None, "") and not pd.isna(peaks_val):
            overrides["n_peaks"] = int(peaks_val)

        bw_raw = row.get("Bandwidth", "")
        if isinstance(bw_raw, str):
            bw_clean = bw_raw.strip()
            if bw_clean:
                try:
                    overrides["bw"] = float(bw_clean)
                except ValueError:
                    overrides["bw"] = bw_clean
        elif bw_raw not in (None, "") and not pd.isna(bw_raw):
            overrides["bw"] = float(bw_raw)

        prom_val = row.get("Prominence")
        if prom_val not in (None, "") and not pd.isna(prom_val):
            overrides["prom"] = float(prom_val)

        max_peaks_val = row.get("Max peaks")
        if max_peaks_val not in (None, "") and not pd.isna(max_peaks_val):
            overrides["max_peaks"] = int(max_peaks_val)

        min_width_val = row.get("Min width")
        if min_width_val not in (None, "") and not pd.isna(min_width_val):
            overrides["min_width"] = int(min_width_val)

        curv_val = row.get("Curvature")
        if curv_val not in (None, "") and not pd.isna(curv_val):
            overrides["curvature"] = float(curv_val)

        turn_val = row.get("Turning points")
        if isinstance(turn_val, str):
            clean = turn_val.strip().lower()
            if clean == "yes":
                overrides["turning_points"] = True
            elif clean == "no":
                overrides["turning_points"] = False
        elif isinstance(turn_val, bool):
            overrides["turning_points"] = turn_val

        sep_val = row.get("Min separation")
        if sep_val not in (None, "") and not pd.isna(sep_val):
            overrides["min_separation"] = max(0.0, float(sep_val))

        grid_val = row.get("Max grid")
        if grid_val not in (None, "") and not pd.isna(grid_val):
            overrides["max_grid"] = int(grid_val)

        drop_val = row.get("Valley drop")
        if drop_val not in (None, "") and not pd.isna(drop_val):
            overrides["valley_drop"] = float(drop_val)

        first_val = row.get("First valley")
        if isinstance(first_val, str):
            clean = first_val.strip()
            if clean in ("Slope change", "Valley drop"):
                overrides["first_valley"] = clean

        _update_pre_override(stem, overrides)

    total_for_sample = sum(1 for c in sample_combos if c["stem"] in st.session_state.pre_overrides)
    if total_for_sample:
        st.caption(
            f"{total_for_sample} combination(s) for **{selected_sample}** currently have overrides."
        )
    else:
        st.caption(
            "Using run setting for peaks, bandwidth "
            f"({bw_label}), prominence ({prom_label}), and advanced options."
        )

# ───────────────── helper: master results accordion ------------------------

def _cols_for_align_mode(mode: str) -> list[str]:
    """Return the proper column names for each align_mode."""
    return {
        "negPeak": ["neg_peak"],
        "valley":  ["valley"],
        "negPeak_valley": ["neg_peak", "valley"],
        "negPeak_valley_posPeak": ["neg_peak", "valley", "pos_peak"],
    }[mode]

def render_aligned(container):
    container.header("Aligned distributions")
    if not st.session_state.aligned_results:
        container.info("Run alignment first."); return

    for stem, info in st.session_state.aligned_results.items():
        with container.expander(stem, expanded=False):
            st.image(st.session_state.aligned_fig_pngs.get(f"{stem}_aligned.png", b""),
                     use_container_width=True)
            st.write(f"**Peaks (after warp):** {info['peaks']}")
            st.write(f"**Valleys (after warp):** {info['valleys']}")

def render_results(container):
    container.header("Processed datasets")
    if not st.session_state.results:
        container.info("No results yet."); return

    for stem, info in list(st.session_state.results.items()):
        p0 = st.session_state.params.get(stem, {})
        bw0, pr0 = str(p0.get("bw", "")), str(p0.get("prom", ""))
        k0 = int(p0.get("n_peaks", 1))

        rowL, rowR = container.columns([11, 1])
        with rowL.expander(stem, expanded=False):
            tab_plot, tab_params, tab_manual = st.tabs(["Plot", "Parameters", "Manual"])
            with tab_plot:
                img_data = st.session_state.fig_pngs.get(f"{stem}.png")
                if img_data:
                    st.image(img_data, use_container_width=True)
                else:
                    st.warning(f"No image available for {stem}.")
                st.write(f"**Peaks:** {info['peaks']}")
                st.write(f"**Valleys:** {info['valleys']}")
                q = info.get("quality", np.nan)
                if np.isfinite(q):
                    st.write(f"**Stain-quality score:** {q:.4f}")
            with tab_params:
                # ——— BANDWIDTH: PRESET VS SLIDER ———
                bw0 = st.session_state.params.get(stem, {}).get("bw", "scott")
                # decide initial mode based on type of bw0
                init_mode = "Numeric" if isinstance(bw0, (int, float)) or bw0.replace(".", "", 1).isdigit() else "Preset"
                bw_input_type = st.selectbox(
                    "Bandwidth input type",
                    ["Preset", "Numeric"],
                    index=0 if init_mode=="Preset" else 1,
                    key=f"{stem}_bw_type",
                    help="Use a preset rule or enter a numeric bandwidth value."
                )
                if bw_input_type == "Preset":
                    bw_opt = st.selectbox(
                        "Preset rule",
                        ["scott", "silverman", "0.5", "0.8", "1.0"],
                        index=(["scott","silverman","0.5","0.8","1.0"].index(str(bw0))
                            if str(bw0) in ["scott","silverman","0.5","0.8","1.0"] else 0),
                        key=f"{stem}_bw_preset",
                        help="Bandwidth rule or scaling factor for KDE smoothing."
                    )
                    bw_new = bw_opt
                else:
                    bw_new = st.slider(
                        "Custom bandwidth",
                        min_value=0.01,
                        max_value=5.0,
                        value=float(bw0) if isinstance(bw0, (int, float)) or bw0.replace(".", "", 1).isdigit() else 1.0,
                        step=0.01,
                        key=f"{stem}_bw_slider",
                        help="Set the KDE bandwidth manually; higher values smooth more."
                    )
                pr_new = st.text_input(
                    "Prominence", pr0, key=f"{stem}_pr",
                    help="Minimum relative drop from peak to surrounding valleys."
                )
                k_new = st.number_input(
                    "# peaks", 1, 6, k0, key=f"{stem}_k",
                    help="Expected number of peaks to detect."
                )
                if (bw_new != bw0) or (pr_new != pr0) or (k_new != k0):
                    st.session_state.params[stem] = {"bw": bw_new, "prom": pr_new, "n_peaks": k_new}
                    _mark_sample_dirty(stem, "manual")
            with tab_manual:
                if not info.get("xs"):
                    st.info("Run detector first to enable manual editing.")
                else:
                    _manual_editor(stem)
        # delete button ------------------------------------------------------
        if rowR.button("❌", key=f"del_{stem}", help="Delete sample"):
            for bucket in ("results", "fig_pngs", "params", "dirty"):
                st.session_state[bucket].pop(stem, None)
            st.session_state.pre_overrides.pop(stem, None)
            st.session_state.group_assignments.pop(stem, None)
            st.session_state.dirty_reason.pop(stem, None)
            st.session_state.results_raw.pop(stem, None)
            st.session_state.results_raw_meta.pop(stem, None)
            for k in (f"{stem}__pk_list", f"{stem}__vl_list"):
                st.session_state.pop(k, None)
            _mark_raw_ridge_stale()                # ← rebuild lazily
            st.rerun()


# ─────────────────────────────── SIDEBAR ─────────────────────────────────────
with st.sidebar:
    mode = st.radio(
        "Choose mode",
        ["Counts CSV files", "Whole dataset"],
        help="Work with individual counts files or an entire dataset.",
        key="mode_selector",
    )

    # 1 Counts-CSV workflow
    if mode == "Counts CSV files":
        uploaded_now = st.file_uploader(
            "Upload *_raw_counts.csv*", type=["csv"],
            accept_multiple_files=True, key="csv_up",
            help="Add one or more raw counts CSV files."
        )
        if uploaded_now:
            names = {f.name for f in st.session_state.cached_uploads}
            for uf in uploaded_now:
                if uf.name in names:
                    continue
                bio = io.BytesIO(uf.read()); bio.seek(0); bio.name = uf.name
                st.session_state.cached_uploads.append(bio)

        use_uploads: list[io.BytesIO] = []
        if st.session_state.cached_uploads:
            st.markdown("**Uploaded CSVs (cached)**")
            stems = [Path(f.name).stem for f in st.session_state.cached_uploads]
            pick  = st.multiselect(
                "Choose uploaded files", stems, stems,
                key="pick_up2", help="Select cached uploads to include in analysis."
            )
            for f in st.session_state.cached_uploads:
                if Path(f.name).stem in pick:
                    use_uploads.append(f)
            if st.button("Clear cached uploads"):
                st.session_state.cached_uploads.clear(); st.rerun()

        use_generated: list[io.BytesIO] = []
        if st.session_state.generated_csvs:
            st.markdown("**Generated CSVs (from dataset)**")
            stems_g = [s for s, _ in st.session_state.generated_csvs]
            pick_g  = st.multiselect(
                "Choose generated files", stems_g, stems_g,
                key="pick_gen2", help="Select generated files from the dataset to use."
            )
            for stem, bio in st.session_state.generated_csvs:
                if stem in pick_g:
                    bio.seek(0); bio.name = f"{stem}.csv"
                    use_generated.append(bio)

        header_row = st.number_input(
            "Header row (−1 = none)", 0, step=1,
            key="hdr", help="Row index containing column names; -1 if absent."
        )
        skip_rows  = st.number_input(
            "Rows to skip", 0, step=1, key="skip",
            help="Number of initial rows to ignore in each file."
        )

    # 2 Whole-dataset workflow
    else:
        expr_file = st.file_uploader(
            "expression_matrix_combined.csv", type=["csv"],
            help="Upload the expression matrix CSV."
        )
        meta_file = st.file_uploader(
            "cell_metadata_combined.csv", type=["csv"],
            help="Upload the cell metadata CSV."
        )

        if st.session_state.expr_df is not None:
            if st.button("Clear loaded dataset"):
                prev_dataset = set(st.session_state.generated_meta)
                for stem in list(st.session_state.pre_overrides):
                    if stem in prev_dataset and stem not in st.session_state.results:
                        st.session_state.pre_overrides.pop(stem, None)
                st.session_state.generated_csvs.clear()
                st.session_state.generated_meta = {}
                for k in (
                    "expr_df",
                    "meta_df",
                    "expr_name",
                    "meta_name",
                ):
                    st.session_state[k] = None
                st.session_state.cli_counts_native = False
                st.rerun()

        with st.expander("Optional: Combine dataset CSV parts"):
            st.markdown(
                "Upload expression and cell metadata CSVs (or ZIP archives of CSV "
                "parts) to merge them into single combined files for download or "
                "direct use in the workflow."
            )
            with st.form("combine_dataset_form", clear_on_submit=False):
                expr_combo = st.file_uploader(
                    "Expression CSV or ZIP",
                    type=["csv", "zip"],
                    accept_multiple_files=True,
                    help=(
                        "Upload one or more CSVs/ZIPs containing expression data; all parts "
                        "will be concatenated."
                    ),
                    key="combine_expr_file",
                )
                meta_combo = st.file_uploader(
                    "Cell metadata CSV or ZIP",
                    type=["csv", "zip"],
                    accept_multiple_files=True,
                    help=(
                        "Upload one or more CSVs/ZIPs containing cell metadata; all parts "
                        "will be concatenated."
                    ),
                    key="combine_meta_file",
                )
                if expr_combo is not None:
                    st.session_state.combine_expr_upload = [
                        (file.name, file.getbuffer().tobytes()) for file in expr_combo
                    ]
                if meta_combo is not None:
                    st.session_state.combine_meta_upload = [
                        (file.name, file.getbuffer().tobytes()) for file in meta_combo
                    ]
                submitted = st.form_submit_button("Combine files")

            expr_combo_file = st.session_state.get("combine_expr_upload")
            meta_combo_file = st.session_state.get("combine_meta_upload")

            if submitted:
                if not expr_combo_file or not meta_combo_file:
                    st.warning(
                        "Upload both an expression file and a cell metadata file before combining."
                    )
                else:
                    try:
                        def _combine_uploaded_files(
                            files: list[tuple[str, bytes]], *, allow_column_union: bool = False
                        ):
                            if not files:
                                raise ValueError("No files were uploaded.")

                            frames: list[pd.DataFrame] = []
                            sources: list[str] = []
                            columns: list[str] | None = None

                            for name, payload in files:
                                buffer = io.BytesIO(payload)
                                buffer.name = name

                                frame, srcs = load_combined_csv(buffer, low_memory=False)
                                if columns is None:
                                    columns = list(frame.columns)
                                elif allow_column_union:
                                    for col in frame.columns:
                                        if col not in columns:
                                            columns.append(col)
                                elif list(frame.columns) != columns:
                                    raise ValueError(
                                        "CSV files have inconsistent columns across uploads."
                                    )

                                frames.append(frame)
                                if srcs:
                                    sources.extend([f"{name}:{part}" for part in srcs])
                                else:
                                    sources.append(name)

                            if allow_column_union and columns is not None:
                                frames = [frame.reindex(columns=columns) for frame in frames]

                            combined = pd.concat(frames, ignore_index=True)
                            return combined, sources

                        expr_df_combo, expr_sources = _combine_uploaded_files(
                            expr_combo_file, allow_column_union=True
                        )
                        meta_df_combo, meta_sources = _combine_uploaded_files(meta_combo_file)
                    except ValueError as exc:
                        st.error(f"Unable to combine files: {exc}")
                    else:
                        st.session_state.combined_expr_df = expr_df_combo
                        st.session_state.combined_meta_df = meta_df_combo
                        st.session_state.combined_expr_sources = expr_sources
                        st.session_state.combined_meta_sources = meta_sources
                        st.session_state.combined_expr_bytes = expr_df_combo.to_csv(
                            index=False
                        ).encode("utf-8")
                        st.session_state.combined_meta_bytes = meta_df_combo.to_csv(
                            index=False
                        ).encode("utf-8")
                        st.session_state.combined_expr_name = "expression_matrix_combined.csv"
                        st.session_state.combined_meta_name = "cell_metadata_combined.csv"
                        st.success("Combined dataset ready below.")

            combined_expr_bytes = st.session_state.get("combined_expr_bytes")
            combined_meta_bytes = st.session_state.get("combined_meta_bytes")
            if combined_expr_bytes is not None and combined_meta_bytes is not None:
                expr_df_combo = st.session_state["combined_expr_df"]
                meta_df_combo = st.session_state["combined_meta_df"]
                expr_summary = _summarize_sources(
                    "Expression matrix", st.session_state.get("combined_expr_sources", [])
                )
                meta_summary = _summarize_sources(
                    "Cell metadata", st.session_state.get("combined_meta_sources", [])
                )
                if expr_summary:
                    st.caption(expr_summary)
                if meta_summary:
                    st.caption(meta_summary)
                st.markdown(
                    f"**Expression matrix:** {expr_df_combo.shape[0]:,} rows × "
                    f"{expr_df_combo.shape[1]:,} columns"
                )
                st.markdown(
                    f"**Cell metadata:** {meta_df_combo.shape[0]:,} rows × "
                    f"{meta_df_combo.shape[1]:,} columns"
                )
                st.download_button(
                    "Download expression_matrix_combined.csv",
                    data=combined_expr_bytes,
                    file_name=(
                        st.session_state.get("combined_expr_name")
                        or "expression_matrix_combined.csv"
                    ),
                    mime="text/csv",
                    key="download_combined_expr",
                )
                st.download_button(
                    "Download cell_metadata_combined.csv",
                    data=combined_meta_bytes,
                    file_name=(
                        st.session_state.get("combined_meta_name")
                        or "cell_metadata_combined.csv"
                    ),
                    mime="text/csv",
                    key="download_combined_meta",
                )
                col_use, col_clear = st.columns(2)
                if col_use.button(
                    "Load combined data into dataset workflow",
                    key="use_combined_dataset",
                ):
                    st.session_state.expr_df = expr_df_combo.copy()
                    st.session_state.meta_df = meta_df_combo.copy()
                    st.session_state.expr_name = (
                        st.session_state.get("combined_expr_name")
                        or "expression_matrix_combined.csv"
                    )
                    st.session_state.meta_name = (
                        st.session_state.get("combined_meta_name")
                        or "cell_metadata_combined.csv"
                    )
                    st.session_state.cli_counts_native = False
                    st.rerun()
                if col_clear.button(
                    "Clear combined dataset files",
                    key="clear_combined_dataset",
                ):
                    for key in (
                        "combined_expr_df",
                        "combined_meta_df",
                        "combined_expr_bytes",
                        "combined_meta_bytes",
                        "combined_expr_sources",
                        "combined_meta_sources",
                        "combined_expr_name",
                        "combined_meta_name",
                        "combine_expr_upload",
                        "combine_meta_upload",
                    ):
                        if isinstance(st.session_state.get(key), list):
                            st.session_state[key] = []
                        else:
                            st.session_state[key] = None
                    st.rerun()

        if expr_file and meta_file:
            need = (st.session_state.expr_df is None or
                    expr_file.name != st.session_state.expr_name or
                    meta_file.name != st.session_state.meta_name)
            if need:
                with st.spinner("Parsing expression / metadata …"):
                    st.session_state.expr_df = pd.read_csv(expr_file, low_memory=False)
                    st.session_state.meta_df = pd.read_csv(meta_file, low_memory=False)
                    st.session_state.expr_name, st.session_state.meta_name = (
                        expr_file.name, meta_file.name
                    )
                    st.session_state.cli_counts_native = False

        expr_df, meta_df = st.session_state.expr_df, st.session_state.meta_df
        if expr_df is not None and meta_df is not None:
            markers = [c for c in expr_df.columns if c != "cell_id"]
            if "batch" in meta_df.columns:
                batches = meta_df["batch"].unique().tolist()
                all_b = st.checkbox(
                    "All batches", True, key="chk_b",
                    help="Use every batch in the dataset."
                )
                sel_b_raw = batches if all_b else st.multiselect(
                    "Batch(es)", batches,
                    help="Select which batches to include."
                )
                sel_b_clean = [None if pd.isna(b) else b for b in sel_b_raw]
                st.session_state.sel_batches = sel_b_clean
                if all_b or not sel_b_clean:
                    meta_use = meta_df
                else:
                    batch_series = meta_df["batch"].where(~meta_df["batch"].isna(), None)
                    meta_use = meta_df[batch_series.isin(sel_b_clean)]
            else:
                st.session_state.sel_batches = []
                meta_use = meta_df
            samples = meta_use["sample"].unique().tolist()

            all_m = st.checkbox(
                "All markers", False, key="chk_m",
                help="Include every marker without selecting individually."
            )
            all_s = st.checkbox(
                "All samples", False, key="chk_s",
                help="Include every sample without selecting individually."
            )
            if all_m:
                excl_m = st.multiselect(
                    "Exclude marker(s)", markers,
                    key="excluded_markers",
                    help="Select markers to leave out while processing all markers.",
                )
            else:
                excl_m = st.session_state.get("excluded_markers", [])
            sel_m_base = markers if all_m else st.multiselect(
                "Marker(s)", markers,
                help="Choose specific markers to process."
            )
            excl_m_set = {str(m).casefold() for m in excl_m}
            sel_m = [m for m in sel_m_base if str(m).casefold() not in excl_m_set]

            if all_s:
                excl_s = st.multiselect(
                    "Exclude sample(s)", samples,
                    key="excluded_samples",
                    help="Select samples to leave out while processing all samples.",
                )
            else:
                excl_s = st.session_state.get("excluded_samples", [])
            sel_s_base = samples if all_s else st.multiselect(
                "Sample(s)", samples,
                help="Choose specific samples to process."
            )
            excl_s_set = {str(s).casefold() for s in excl_s}
            sel_s = [s for s in sel_s_base if str(s).casefold() not in excl_s_set]
            st.session_state.sel_markers = sel_m
            st.session_state.sel_samples = sel_s

            if sel_m and sel_s and st.button("Generate counts CSVs"):
                use_batches = "batch" in meta_df.columns
                batches_for_sample: dict[str, list[str | None]] = {}
                for s in sel_s:
                    batches = [None]
                    if use_batches:
                        batches = meta_df.loc[meta_df["sample"].eq(s), "batch"].unique().tolist()
                        sel_b = st.session_state.get("sel_batches", [])
                        if sel_b:
                            batches = [b for b in batches if b in sel_b]
                    batches_for_sample[s] = batches

                tot = len(sel_m) * sum(len(batches_for_sample[s]) for s in sel_s)
                bar = st.progress(0.0, "Generating …")
                exist = {s for s, _ in st.session_state.generated_csvs}
                idx = 0
                for m in sel_m:
                    for s in sel_s:
                        for b in batches_for_sample[s]:
                            idx += 1
                            stem = f"{s}_{b}_{m}_raw_counts" if b is not None else f"{s}_{m}_raw_counts"
                            if stem in exist:
                                bar.progress(idx / tot,
                                             f"Skip {stem} (exists)")
                                continue
                            mask = meta_df["sample"].eq(s)
                            if use_batches and b is not None:
                                mask &= meta_df["batch"].eq(b)
                            vals = expr_df.loc[mask, m]
                            if st.session_state.get("apply_arcsinh", True):
                                counts = arcsinh_transform(
                                    vals,
                                    a=st.session_state.get("arcsinh_a", 1.0),
                                    b=st.session_state.get("arcsinh_b", 1 / 5),
                                    c=st.session_state.get("arcsinh_c", 0.0),
                                )
                                arcsinh_applied = True
                            else:
                                counts = vals.astype(float)
                                arcsinh_applied = False
                            bio = io.BytesIO()
                            counts.to_csv(bio, index=False, header=False)
                            bio.seek(0)
                            bio.name = f"{stem}.csv"
                            setattr(bio, "marker", m)
                            setattr(bio, "arcsinh", arcsinh_applied)
                            if b is not None:
                                setattr(bio, "batch", b)
                            st.session_state.generated_csvs.append((stem, bio))
                            bar.progress(idx / tot,
                                         f"Added {stem} ({idx}/{tot})")
                bar.empty()
                st.success("CSVs cached – switch to **Counts CSV files**")

        header_row, skip_rows = -1, 0
        use_uploads, use_generated = [], []

    # ───────────── Preprocessing ──────────────
    st.markdown("---\n### Preprocessing")
    apply_arc = st.checkbox(
        "Apply arcsinh transform",
        True,
        key="apply_arcsinh",
        help=(
            "Transform counts using (1/b)·arcsinh(a·x + c). "
            "Uncheck if data is already arcsinh-transformed."
        ),
    )
    if apply_arc:
        st.number_input("a", value=1.0, key="arcsinh_a")
        st.number_input("b", value=1 / 5, key="arcsinh_b")
        st.number_input("c", value=0.0, key="arcsinh_c")

    # ───────────── Detection options ──────────────
    st.markdown("---\n### Detection")
    auto = st.selectbox(
        "Number of peaks", ["GPT Automatic", 1, 2, 3, 4, 5, 6],
        help="Let GPT guess peak count or fix it manually."
    )
    n_fixed = None if auto == "GPT Automatic" else int(auto)
    cap_min = n_fixed if n_fixed else 1
    max_peaks = st.number_input(
        "Maximum peaks (Automatic cap)",
        cap_min, 6, max(2, cap_min), step=1,
        disabled=(n_fixed is not None),
        help="Upper limit when GPT determines peak count."
    )

    # Bandwidth
    bw_mode = st.selectbox(
        "Bandwidth mode", ["Manual", "GPT automatic"],
        help="Choose manual bandwidth or let GPT estimate it."
    )
    bw_opt = None
    if bw_mode == "Manual":
        bw_opt = st.selectbox(
            "Rule / scale",
            ["scott", "silverman", "0.5", "0.8", "1.0"],
            key="bw_sel",
            help="Bandwidth rule or multiplier when set manually."
        )
        bw_val = (float(bw_opt)
                  if bw_opt.replace(".", "", 1).isdigit() else bw_opt)
    else:
        bw_val = None  # GPT later

    # Prominence
    prom_mode = st.selectbox(
        "Prominence", ["Manual", "GPT automatic"], key="prom_sel",
        help="Set prominence threshold yourself or let GPT decide."
    )
    prom_val = (
        st.slider(
            "Prominence value", 0.00, 0.30, 0.05, 0.01,
            help="Minimum relative drop from peak to valley."
        )
        if prom_mode == "Manual" else None
    )

    min_w    = st.slider(
        "Min peak width", 0, 6, 0, 1,
        help="Smallest allowable width for detected peaks."
    )
    curv = st.slider(
        "Curvature thresh (0 = off)", 0.0000, 0.005, 0.0001, 0.0001,
        format="%.4f",
        help="Filter out peaks with curvature below this value; 0 disables."
    )
    tp   = st.checkbox(
        "Treat concave-down turning points as peaks", False,
        help="Count concave-down turning points as valid peaks."
    )
    min_sep   = st.slider(
        "Min peak separation", 0.0, 10.0, DEFAULT_MIN_SEPARATION, 0.1,
        help="Minimum distance required between peaks."
    )
    grid_sz  = st.slider(
        "Max KDE grid", 4_000, 40_000, 20_000, 1_000,
        help="Number of grid points for KDE; higher is slower but finer."
    )
    val_drop = st.slider(
        "Valley drop (% of peak)", 1, 50, 10, 1,
        help="Required drop from peak height to qualify as a valley."
    )
    val_mode = st.radio(
        "First valley method", ["Slope change", "Valley drop"],
        horizontal=True,
        help="How to determine the first valley after a peak."
    )

    run_defaults = {
        "max_peaks": max_peaks,
        "min_width": min_w,
        "curvature": curv,
        "turning_points": tp,
        "min_separation": min_sep,
        "max_grid": grid_sz,
        "valley_drop": val_drop,
        "first_valley": val_mode,
    }

    st.checkbox(
        "Enforce marker consistency across samples",
        key="apply_consistency",
        help="Use the same marker selection for all samples."
    )

    if bw_mode == "Manual":
        if isinstance(bw_val, (int, float)):
            bw_label = f"Manual ({bw_val:.2f})"
        else:
            bw_label = f"Manual ({bw_opt})"
    else:
        bw_label = "GPT automatic"

    if mode == "Counts CSV files":
        selected_files = use_uploads + use_generated
        seen: set[str] = set()
        stems_for_override: list[str] = []
        for f in selected_files:
            stem = Path(f.name).stem
            if stem not in seen:
                stems_for_override.append(stem)
                seen.add(stem)

        for stem in list(st.session_state.pre_overrides):
            if stem not in seen and stem not in st.session_state.results:
                st.session_state.pre_overrides.pop(stem, None)

        _render_group_controls(
            stems_for_override,
            bw_label=bw_label,
            prom_mode=prom_mode,
            prom_default=prom_val,
            run_defaults=run_defaults,
        )

        if stems_for_override:
            _render_pre_run_overrides(
                stems_for_override,
                bw_label=bw_label,
                prom_mode=prom_mode,
                prom_default=prom_val,
                run_defaults=run_defaults,
            )
        else:
            for stem in list(st.session_state.pre_overrides):
                if stem not in st.session_state.results:
                    st.session_state.pre_overrides.pop(stem, None)

    elif mode == "Whole dataset":
        meta_df = st.session_state.meta_df
        sel_m = st.session_state.get("sel_markers", [])
        sel_s = st.session_state.get("sel_samples", [])
        sel_b = st.session_state.get("sel_batches", [])

        combos = (
            _enumerate_dataset_entries(sel_m, sel_s, meta_df, sel_b)
            if meta_df is not None
            else []
        )

        previous_dataset_stems = set(st.session_state.generated_meta)
        st.session_state.generated_meta = {
            combo["stem"]: {
                "sample": combo["sample"],
                "marker": combo["marker"],
                "batch": combo["batch"],
                "batch_label": combo["batch_label"],
            }
            for combo in combos
        }
        keep_stems = set(st.session_state.generated_meta)
        if keep_stems:
            st.session_state.generated_csvs = [
                (stem, bio)
                for stem, bio in st.session_state.generated_csvs
                if stem in keep_stems
            ]
        else:
            st.session_state.generated_csvs = []
        current_dataset_stems = set(st.session_state.generated_meta)
        for stem in list(st.session_state.pre_overrides):
            if (
                stem in previous_dataset_stems
                and stem not in current_dataset_stems
                and stem not in st.session_state.results
            ):
                st.session_state.pre_overrides.pop(stem, None)

        stems_for_override = [combo["stem"] for combo in combos]
        _render_group_controls(
            stems_for_override,
            bw_label=bw_label,
            prom_mode=prom_mode,
            prom_default=prom_val,
            run_defaults=run_defaults,
        )

        if combos:
            _render_dataset_overrides(
                combos,
                bw_label=bw_label,
                prom_mode=prom_mode,
                prom_default=prom_val,
            )
        elif sel_m and sel_s:
            st.caption(
                "No dataset entries match the current selection. Adjust the filters above to configure overrides."
            )

    # ───────────── Alignment options ──────────────
    st.markdown("---\n### Alignment")

    align_mode = st.selectbox(
        "Landmark set",
        ["negPeak_valley_posPeak", "negPeak_valley", "negPeak", "valley"],
        index=0, key="align_mode",
        help="Choose which landmarks to align across samples."
    )

    col_names = _cols_for_align_mode(
        st.session_state.get("align_mode", "negPeak_valley_posPeak")
    )

    # ► choose where the *aligned* landmarks should end up
    target_mode = st.radio(
        "Target landmark positions",
        ["Automatic (median across samples)", "Custom (enter numbers)"],
        horizontal=True, key="target_mode",
        help="Use automatic median positions or provide custom targets."
    )

    def _ask_numbers(labels: list[str], defaults: list[float], prefix: str) -> list[float]:
        vals = []
        for lab, d, i in zip(labels, defaults, range(len(labels))):
            v = st.number_input(
                f"Target {lab}", value=float(d), key=f"{prefix}_{i}",
                help="Desired position for this landmark after alignment."
            )
            vals.append(float(v))
        return vals

    if target_mode.startswith("Automatic"):
        target_vec: list[float] | None = None    # -> compute later from data
    else:
        if align_mode == "negPeak":
            target_vec = _ask_numbers(["negative peak"], [2.0], "tgt1")
        elif align_mode == "valley":
            target_vec = _ask_numbers(["valley"], [3.0], "tgt2")
        elif align_mode == "negPeak_valley":
            target_vec = _ask_numbers(
                ["negative peak", "valley"], [2.0, 3.0], "tgt3"
            )
        else:   # 3-landmark
            target_vec = _ask_numbers(
                ["negative peak", "valley", "right-most peak"],
                [2.0, 3.0, 5.0], "tgt4"
            )


    st.markdown("---\n### GPT helper")
    pick = st.selectbox(
        "Model",
        ["o4-mini", "gpt-4o-mini", "gpt-4-turbo-preview", "Custom"],
        help="GPT model used for automatic parameter suggestions."
    )
    gpt_model = (
        st.text_input(
            "Custom model", help="Name of OpenAI model when using 'Custom'."
        )
        if pick == "Custom" else pick
    )
    api_key   = st.text_input(
        "OpenAI API key", type="password",
        help="Key for accessing the OpenAI API."
    )

    workers = st.slider(
        "Workers",
        1,
        8,
        int(st.session_state.get("workers", 1)),
        help="Number of parallel workers to use during batch processing.",
    )
    st.session_state.workers = workers
    gpt_features_selected = (n_fixed is None) or prom_mode == "GPT automatic" or bw_mode == "GPT automatic"
    if workers > 1 and gpt_features_selected:
        st.caption(
            "Multiple workers will send concurrent GPT requests. Reduce workers if you encounter API rate limits."
        )



_render_cli_import_section()

# ───────────────── main buttons & global progress bar ────────────────────────
run_col, clear_col, pause_col = st.columns(3)
if clear_col.button("Clear results"):
    # buckets that are always dict-like
    for bucket in ("results", "fig_pngs", "params", "dirty", "dirty_reason",
                   "aligned_results", "aligned_fig_pngs",
                   "results_raw",          # ← keep as dict, not None
                   "results_raw_meta",
                   ):
        st.session_state[bucket] = {}       # {} stays {}

    # scalar keys or arrays
    for key in ("aligned_counts", "aligned_landmarks",
                "aligned_ridge_png", "raw_ridge_png"):
        st.session_state[key] = None

    st.session_state.generated_csvs.clear()
    st.session_state.generated_meta = {}

    st.session_state.pending.clear()
    st.session_state.total_todo = 0
    st.session_state.run_active = False
    st.session_state.batch_thread = None
    st.session_state.batch_queue = None
    st.session_state.batch_progress = {"completed": 0, "total": 0}
    st.rerun()

run_clicked = run_col.button("Run detector")

pause_label = "Stop" if st.session_state.run_active else "Resume"
pause_disabled = not bool(
    st.session_state.run_active
    and st.session_state.batch_thread
    and st.session_state.batch_thread.is_alive()
)
pause_clicked = pause_col.button(pause_label, disabled=pause_disabled)

if pause_clicked and st.session_state.run_active:
    st.session_state.batch_run_id += 1
    st.session_state.run_active = False
    st.session_state.paused = False
    st.session_state.pending = []
    st.session_state.total_todo = 0
    st.session_state.batch_progress = {"completed": 0, "total": 0}
    st.session_state.batch_thread = None
    st.session_state.batch_queue = None
    st.rerun()

# ───────────────────────────── incremental processing ───────────────────────
prog_placeholder = st.empty()
_process_batch_events()
progress_state = st.session_state.get("batch_progress", {"completed": 0, "total": 0})
total_now = progress_state.get("total", 0)
completed_now = progress_state.get("completed", 0)
if total_now:
    prog_placeholder.progress(
        completed_now / total_now if total_now else 0,
        f"Processing… {completed_now}/{total_now}",
    )

def _start_batch_run(
    csv_files: list[io.BytesIO], *, require_gpt: bool = True
) -> None:
    if not csv_files:
        st.error("No CSV files selected.")
        st.stop()

    arcsinh_sig = _current_arcsinh_signature()
    samples = _prepare_sample_inputs(csv_files, arcsinh_sig, header_row, skip_rows)
    if not samples:
        st.error("No samples available to process.")
        st.stop()

    overrides = _prepare_batch_overrides([s.stem for s in samples])

    options = BatchOptions(
        apply_arcsinh=bool(st.session_state.get("apply_arcsinh", True)),
        arcsinh_a=float(st.session_state.get("arcsinh_a", 1.0)),
        arcsinh_b=float(st.session_state.get("arcsinh_b", 1 / 5)),
        arcsinh_c=float(st.session_state.get("arcsinh_c", 0.0)),
        n_peaks=n_fixed,
        n_peaks_auto=n_fixed is None,
        max_peaks=int(max_peaks),
        bandwidth=bw_val if bw_mode == "Manual" else "scott",
        bandwidth_auto=bw_mode != "Manual",
        prominence=prom_val if prom_val is not None else 0.05,
        prominence_auto=prom_mode != "Manual",
        min_width=int(min_w),
        curvature=float(curv),
        turning_points=bool(tp),
        min_separation=float(min_sep),
        grid_size=int(grid_sz),
        valley_drop=float(val_drop),
        first_valley="drop" if val_mode == "Valley drop" else "slope",
        apply_consistency=bool(st.session_state.get("apply_consistency", False)),
        align=False,
        align_mode=align_mode,
        target_landmarks=target_vec,
        group_by_marker=bool(st.session_state.get("align_group_markers", False)),
        workers=int(st.session_state.get("workers", 1)),
        gpt_model=gpt_model,
    )

    gpt_needed = (
        require_gpt
        and (
            options.bandwidth_auto
            or options.prominence_auto
            or options.n_peaks_auto
        )
    )
    gpt_client = None
    if gpt_needed:
        if not api_key:
            st.error("GPT-based options need an OpenAI key.")
            st.stop()
        try:
            gpt_client = OpenAI(api_key=api_key)
        except AuthenticationError:
            if not st.session_state.invalid_api_key:
                st.warning(
                    "Invalid OpenAI API key; please update it to enable GPT features."
                )
                st.session_state.invalid_api_key = True
            st.stop()
        except Exception as exc:
            st.error(f"Failed to initialise OpenAI client: {exc}")
            st.stop()

    st.session_state.batch_run_id += 1
    st.session_state.batch_progress = {"completed": 0, "total": len(samples)}
    st.session_state.total_todo = len(samples)
    st.session_state.pending = csv_files
    st.session_state.run_active = True
    st.session_state.batch_error = None
    queue_obj = Queue()
    st.session_state.batch_queue = queue_obj
    thread = Thread(
        target=_run_batch_async,
        args=(
            st.session_state.batch_run_id,
            samples,
            options,
            overrides,
            gpt_client,
            queue_obj,
        ),
        daemon=True,
    )
    st.session_state.batch_thread = thread
    thread.start()
    st.rerun()

# 1 User clicked RUN: prepare pending queue
if run_clicked and not st.session_state.run_active:
    st.session_state.raw_ridge_png = None
    csv_files = use_uploads + use_generated
    if mode == "Whole dataset" and expr_df is not None and meta_df is not None:
        sel_m = st.session_state.get("sel_markers", [])
        sel_s = st.session_state.get("sel_samples", [])
        sel_b = st.session_state.get("sel_batches", [])
        if sel_m and sel_s:
            _sync_generated_counts(sel_m, sel_s, expr_df, meta_df, sel_b)
        use_uploads = []
        use_generated = [bio for _, bio in st.session_state.generated_csvs]
        csv_files = use_uploads + use_generated

    todo = [
        f
        for f in csv_files
        if (stem := Path(f.name).stem) not in st.session_state.results
        or st.session_state.dirty.get(stem, False)
    ]

    _start_batch_run(todo)

if (
  st.session_state.run_active
  and st.session_state.pending
  and st.session_state.batch_thread is None
):
  _start_batch_run(
      list(st.session_state.pending),
      require_gpt=not bool(st.session_state.get("cli_counts_native")),
  )

if st.session_state.batch_error:
    msg = str(st.session_state.batch_error)
    if "rate limit" in msg.lower():
        msg += " — try reducing the number of workers."
    st.error(msg)

if st.session_state.run_active:
    results_container = st.container()
    render_results(results_container)
    _schedule_batch_rerun()
    st.stop()

if st.session_state.results and st.session_state.get("raw_ridge_png") is None:
    _refresh_raw_ridge()


# ────────────────────────── static results & download ────────────────────────
results_container = st.container()
render_results(results_container)

# ──────────────  SUMMARY  +  RIDGE-PLOT TABS  ──────────────
if st.session_state.results:
  df = pd.DataFrame(
      [{
          "file":     k,
          "peaks":    v["peaks"],
          "valleys":  v["valleys"],
          "quality":  round(v.get("quality", np.nan), 4),
      } for k, v in st.session_state.results.items()]
  )
else:
  df = pd.DataFrame()

download_section = st.container()

tab_sum, tab_quality, tab_cmp = st.tabs(
    ["Summary",
    "Quality",
    "Comparison"]
)

# TAB 1  – summary table
with tab_sum:
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run the detector first to see summary information.")

# TAB 2 – quality‐score bar plot
with tab_quality:
    if not df.empty:
        # build DataFrame of scores
        qual_df = df[["file", "quality"]]
        # plot
        fig, ax = plt.subplots()
        ax.bar(qual_df["file"], qual_df["quality"])
        ax.set_xticks(range(len(qual_df)))
        ax.set_xticklabels(qual_df["file"], rotation=45, ha="right")
        ax.set_ylabel("Stain-quality score")
        ax.set_title("Quality scores across samples")
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Run the detector first to see quality scores.")

with download_section:
    if st.session_state.results:
        st.markdown("### Downloads")
        col_before, col_after, col_both = st.columns(3)

        col_before.download_button(
            "Download before-alignment data",
            _make_before_alignment_zip(),
            _before_alignment_zip_name(),
            mime="application/zip",
            key="before_alignment_download",
        )

        if st.session_state.get("aligned_counts"):
            col_after.download_button(
                "Download aligned data",
                _make_final_alignment_zip(),
                _final_alignment_zip_name(),
                mime="application/zip",
                key="aligned_download",
            )
            col_both.download_button(
                "Download before & after data",
                _make_combined_alignment_zip(),
                _combined_alignment_zip_name(),
                mime="application/zip",
                key="before_after_download",
            )
        else:
            col_after.button("Download aligned data", disabled=True)
            col_both.button("Download before & after data", disabled=True)

# # TAB 2  – raw ridge
# with tab_raw:
#     if st.session_state.get("raw_ridge_png") is None:
#         _refresh_raw_ridge()
#     if st.session_state.get("raw_ridge_png"):
#         st.image(st.session_state.raw_ridge_png,
#                  use_container_width=True,
#                  caption="Stacked densities – *before* alignment")

# # TAB 3  – aligned ridge
# with tab_aln:
#     if st.session_state.get("aligned_ridge_png"):
#         st.image(st.session_state.aligned_ridge_png,
#                  use_container_width=True,
#                  caption="Stacked densities – *after* alignment")
#         st.download_button("Download Aligned Data",
#                    _make_aligned_zip(),
#                    "alignedData.zip",
#                    mime="application/zip",
#                    key="alignedDownload")
#     else:
#         st.markdown("You need to select 'Align landmarks * normalize counts' in order for graphs to generate.")

# TAB 4  – side-by-side comparison
with tab_cmp:
    col1, col2 = st.columns(2, gap="small")
    if st.session_state.get("raw_ridge_png"):
        col1.image(st.session_state.raw_ridge_png,
                   use_container_width=True,
                   caption="Raw")
    if st.session_state.get("aligned_ridge_png"):
        col2.image(st.session_state.aligned_ridge_png,
                   use_container_width=True,
                   caption="Aligned")

    if st.session_state.results:
        group_map = _group_stems_with_results()
        groups_with_data = {k: v for k, v in group_map.items() if v}
        has_extra_groups = any(name != "Default" for name in groups_with_data)

        if has_extra_groups:
            st.markdown("---")
            st.markdown("#### Group comparison")
            st.caption(
                "Review the stacked densities for each sample group before "
                "and after alignment."
            )

            align_available = bool(st.session_state.aligned_results)

            def _group_sort(item: tuple[str, list[str]]) -> tuple[int, str]:
                name, _ = item
                return (0, "") if name == "Default" else (1, name.lower())

            for group_name, stems in sorted(groups_with_data.items(), key=_group_sort):
                sample_count = len(stems)
                display_name = (
                    "Default (unassigned)" if group_name == "Default" else group_name
                )
                label = (
                    f"{display_name} ({sample_count} sample"
                    f"{'s' if sample_count != 1 else ''})"
                )
                grp_col_raw, grp_col_aligned = st.columns(2, gap="small")

                raw_png = _ridge_plot_for_stems(stems, st.session_state.results)
                if raw_png:
                    grp_col_raw.image(
                        raw_png,
                        use_container_width=True,
                        caption=f"Raw – {label}",
                    )
                else:
                    grp_col_raw.info("No processed counts available for this group yet.")

                if align_available:
                    aligned_png = _ridge_plot_for_stems(
                        stems, st.session_state.aligned_results
                    )
                    if aligned_png:
                        grp_col_aligned.image(
                            aligned_png,
                            use_container_width=True,
                            caption=f"Aligned – {label}",
                        )
                    else:
                        grp_col_aligned.info(
                            "No aligned curves are available for this group yet."
                        )
                else:
                    grp_col_aligned.info(
                        "Align landmarks to compare the normalized curves for this group."
                    )

if st.session_state.results:

    df = pd.DataFrame(
        [{"file": k,
        "peaks": v["peaks"],
        "valleys": v["valleys"],
        "quality": round(v.get("quality", np.nan), 4)}    #  ← NEW col
        for k, v in st.session_state.results.items()]
    )

    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("summary.csv", df.to_csv(index=False).encode())
            for fn, png in st.session_state.fig_pngs.items():
                z.writestr(fn, png)
        st.download_button("Download ZIP", buf.getvalue(),
                           "PeakValleyResults.zip", "application/zip")


if st.session_state.results:
    st.markdown("---")
    align_col, dl_col = st.columns([2, 1])
    with align_col:
        do_align = st.button("Align landmarks & normalize counts",
                             type="primary")
    if do_align:
        with st.spinner("Running landmark alignment …"):
            _align_results_by_group(align_mode=align_mode, target_vec=target_vec)

        st.success("Landmarks aligned – scroll down for the stacked view or download the ZIP!")
        st.rerun()
