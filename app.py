# app.py  – GPT-assisted bandwidth detector with
#           live incremental results + per-sample overrides
from __future__ import annotations
import io, zipfile, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI, AuthenticationError

from peak_valley.quality import stain_quality

from peak_valley import (
    arcsinh_transform, read_counts,
    kde_peaks_valleys, quick_peak_estimate,
    fig_to_png, enforce_marker_consistency,
)
from peak_valley.gpt_adapter import (
    ask_gpt_peak_count, ask_gpt_prominence, ask_gpt_bandwidth,
)
from peak_valley.alignment import align_distributions, fill_landmark_matrix

# ────────────────────────── Streamlit page & state ──────────────────────────
st.set_page_config("Peak & Valley Detector", None, layout="wide")
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
    "fig_pngs":    {},         # stem.png → png bytes (latest)
    "params":      {},         # stem → {"bw":…, "prom":…, "n_peaks":…}
    "dirty":       {},         # stem → True if user edited params *or* positions
    "pre_overrides": {},       # stem → pending overrides before first run
    "cached_uploads": [],
    "generated_csvs": [],
    "sel_markers": [], "sel_samples": [], "sel_batches": [],
    "expr_df": None, "meta_df": None,
    "expr_name": None, "meta_name": None,
    # incremental‑run machinery
    "pending":     [],         # list[io.BytesIO] still to process
    "total_todo":  0,
    "run_active":  False,
    "aligned_counts":    None,
    "aligned_landmarks": None,
    "aligned_results": {},   # stem → {"peaks":…, "valleys":…, "xs":…, "ys":…}
    "aligned_fig_pngs": {},  # stem_aligned.png → bytes
    "aligned_ridge_png":    None,
    "apply_consistency": True,  # enforce marker consistency across samples
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── helper #1 : aligned ZIP  (plots + aligned counts) ───────────────
def _refresh_raw_ridge() -> None:
    """Re-compute the stacked RAW ridge plot and store it in session_state."""
    if not st.session_state.results:
        st.session_state.raw_ridge_png = None
        return

    infos  = list(st.session_state.results.values())
    xs_all = [np.asarray(info.get("xs", []), float) for info in infos if info.get("xs")]

    x_min = float(min(x.min() for x in xs_all))
    x_max = float(max(x.max() for x in xs_all))
    pad   = 0.05 * (x_max - x_min)

    y_max = max(
        (np.asarray(info.get("ys", []), float).max() if info.get("ys") else 0.0)
        for info in infos
    )
    gap   = 1.2 * y_max

    fig, ax = plt.subplots(figsize=(6, 0.8 * len(xs_all)), dpi=150, sharex=True)

    for i, stem in enumerate(st.session_state.results):
        info   = st.session_state.results[stem]
        xs     = np.asarray(info.get("xs", []), float)
        ys     = np.asarray(info.get("ys", []), float)
        offset = i * gap

        ax.plot(xs, ys + offset, color="black", lw=1)
        ax.fill_between(xs, offset, ys + offset, color="#FFA50088", lw=0)

        for p in info["peaks"]:
            ax.vlines(p, offset, offset + ys.max(), color="black", lw=0.8)
        for v in info["valleys"]:
            ax.vlines(v, offset, offset + ys.max(), color="grey",
                      lw=0.8, linestyles=":")

        ax.text(x_min, offset + 0.5 * y_max, stem,
                ha="right", va="center", fontsize=7)

    ax.set_yticks([]); ax.set_xlim(x_min - pad, x_max + pad); fig.tight_layout()
    st.session_state.raw_ridge_png = fig_to_png(fig)
    plt.close(fig)

def _make_aligned_zip() -> bytes:
    """Build a ZIP with *everything* related to the aligned data."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:

        # 1  aligned counts (one CSV per sample)
        for stem, arr in st.session_state.aligned_counts.items():
            bio = io.BytesIO()
            np.savetxt(bio, arr, delimiter=",")
            z.writestr(f"{stem}_aligned.csv", bio.getvalue())

        # 2  aligned figures (per-sample + ridge)
        for fn, png in st.session_state.aligned_fig_pngs.items():
            z.writestr(fn, png)
        if st.session_state.aligned_ridge_png:
            z.writestr("aligned_ridge.png", st.session_state.aligned_ridge_png)

        # ───────────── 3  **NEW** overall summary ─────────────
        if st.session_state.aligned_results:                 # make sure it exists
            import pandas as pd                              # local import is fine
            rows = [
                {
                    "file":    stem,
                    "peaks":   info["peaks"],
                    "valleys": info["valleys"],
                }
                for stem, info in st.session_state.aligned_results.items()
            ]
            df_sum = pd.DataFrame(rows)
            z.writestr("aligned_summary.csv",
                       df_sum.to_csv(index=False).encode())

    return out.getvalue()


# ── helper #2 : curves ZIP  (xs/ys for every sample) ────────────────
def _make_curves_zip() -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for stem, info in st.session_state.results.items():
            bio_xs = io.BytesIO(); np.savetxt(bio_xs, info["xs"], delimiter=",")
            bio_ys = io.BytesIO(); np.savetxt(bio_ys, info["ys"], delimiter=",")
            z.writestr(f"{stem}_xs.csv", bio_xs.getvalue())
            z.writestr(f"{stem}_ys.csv", bio_ys.getvalue())
    return out.getvalue()


def _sync_generated_counts(sel_m: list[str], sel_s: list[str],
                           expr_df: pd.DataFrame, meta_df: pd.DataFrame,
                           sel_b: list[str] | None = None) -> None:
    """Refresh cached CSVs to match the current marker/sample selection.

    The ``stem`` for each generated CSV uniquely identifies the batch, sample,
    and marker combination (``<sample>_<batch>_<marker>_raw_counts``).  When no
    batch column is present the stem falls back to ``<sample>_<marker>``.
    """
    use_batches = "batch" in meta_df.columns

    desired: dict[str, tuple[str, str, str | None]] = {}
    for s in sel_s:
        batches = [None]
        if use_batches:
            batches = meta_df.loc[meta_df["sample"].eq(s), "batch"].unique()
            if sel_b:
                batches = [b for b in batches if b in sel_b]
        for b in batches:
            for m in sel_m:
                stem = f"{s}_{b}_{m}_raw_counts" if b is not None else \
                       f"{s}_{m}_raw_counts"
                desired[stem] = (s, m, b)

    # remove stale combinations
    st.session_state.generated_csvs = [
        (stem, bio) for stem, bio in st.session_state.generated_csvs
        if stem in desired
    ]

    # drop outdated results
    keep = set(desired)
    for bucket in ("results", "results_raw", "params", "dirty",
                   "aligned_results"):
        st.session_state[bucket] = {
            k: v for k, v in st.session_state[bucket].items() if k in keep
        }
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

    existing = {stem for stem, _ in st.session_state.generated_csvs}
    for stem, (s, m, b) in desired.items():
        if stem in existing:
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


# ───────────────────────── helper: (re)plot a dataset ───────────────────────

def _plot_png_fixed(stem, xs, ys, peaks, valleys,
                    xlim, ylim) -> bytes:
    """Same as _plot_png but with shared axes limits."""
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
    ax.plot(xs, ys, color="orange")
    ax.fill_between(xs, 0, ys, color="#FFA50088")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, ylim)
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
        st.session_state.results[stem]["peaks"]   = pk_list.copy()
        st.session_state.results[stem]["valleys"] = vl_list.copy()
        _refresh_raw_ridge()                       # ← REBUILD HERE
        st.rerun()


def _render_pre_run_overrides(
    stems: list[str], *,
    bw_label: str,
    prom_mode: str,
    prom_default: float | None,
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
        "Override the peak count, bandwidth, or prominence for specific files "
        "before running the detector. Leave everything on “Use run setting” to "
        "apply the global configuration."
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

            col_peak, col_bw, col_prom = st.columns(3)

            peak_options = [None, 1, 2, 3, 4, 5, 6]
            peak_default = prev.get("n_peaks")
            peak_index = (
                peak_options.index(peak_default)
                if peak_default in peak_options
                else 0
            )
            n_choice = col_peak.selectbox(
                "Peaks",
                options=peak_options,
                index=peak_index,
                format_func=lambda v: "Use run setting" if v is None else f"{v}",
                key=f"{stem}_pre_peak",
                help="Choose a fixed number of peaks for this file.",
            )

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
                key=f"{stem}_pre_bw_choice",
                help="Pick a bandwidth rule or numeric value for this file.",
            )

            if bw_choice == "custom":
                bw_value = col_bw.slider(
                    "Custom bandwidth",
                    min_value=0.01,
                    max_value=5.0,
                    value=float(bw_custom_default),
                    step=0.01,
                    key=f"{stem}_pre_bw_val",
                    help="Manual KDE bandwidth just for this file.",
                )
                bw_override: float | str | None = float(bw_value)
            elif bw_choice is None:
                bw_override = None
            else:
                bw_override = bw_choice

            prom_prev = prev.get("prom")
            prom_toggle = col_prom.checkbox(
                "Custom prominence",
                value=(prom_prev is not None),
                key=f"{stem}_pre_prom_toggle",
                help="Enable to override the prominence threshold for this file.",
            )
            if prom_toggle:
                prom_start = (
                    float(prom_prev)
                    if prom_prev is not None
                    else float(prom_default if prom_default is not None else 0.05)
                )
                prom_override = col_prom.slider(
                    "Prominence",
                    min_value=0.00,
                    max_value=0.30,
                    value=prom_start,
                    step=0.01,
                    key=f"{stem}_pre_prom_val",
                    help="Set the minimum drop from peak to valley for this file.",
                )
            else:
                prom_override = None

            overrides = {}
            if n_choice is not None:
                overrides["n_peaks"] = int(n_choice)
            if bw_override is not None:
                overrides["bw"] = bw_override
            if prom_override is not None:
                overrides["prom"] = float(prom_override)

            if overrides:
                if overrides != prev:
                    st.session_state.pre_overrides[stem] = overrides
            elif prev:
                st.session_state.pre_overrides.pop(stem, None)

            if overrides:
                summary = []
                if "n_peaks" in overrides:
                    summary.append(f"{overrides['n_peaks']} peaks")
                if "bw" in overrides:
                    bw_val = overrides["bw"]
                    bw_txt = (
                        f"{bw_val:.2f}"
                        if isinstance(bw_val, (int, float))
                        else str(bw_val)
                    )
                    summary.append(f"bandwidth {bw_txt}")
                if "prom" in overrides:
                    summary.append(f"prominence {overrides['prom']:.2f}")
                st.caption("Overrides: " + ", ".join(summary))
            else:
                st.caption(
                    f"Using run setting for peaks, bandwidth ({bw_label}), and "
                    f"prominence ({prom_label})."
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
                    st.session_state.dirty[stem] = True
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
            for k in (f"{stem}__pk_list", f"{stem}__vl_list"):
                st.session_state.pop(k, None)
            _refresh_raw_ridge()                   # ← keep ridge in sync
            st.rerun()


# ─────────────────────────────── SIDEBAR ─────────────────────────────────────
with st.sidebar:
    mode = st.radio(
        "Choose mode", ["Counts CSV files", "Whole dataset"],
        help="Work with individual counts files or an entire dataset."
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
                for k in ("expr_df", "meta_df", "expr_name", "meta_name"):
                    st.session_state[k] = None
                st.rerun()

        if expr_file and meta_file:
            need = (st.session_state.expr_df is None or
                    expr_file.name != st.session_state.expr_name or
                    meta_file.name != st.session_state.meta_name)
            if need:
                with st.spinner("Parsing expression / metadata …"):
                    st.session_state.expr_df = pd.read_csv(expr_file,
                                                           low_memory=False)
                    st.session_state.meta_df = pd.read_csv(meta_file,
                                                           low_memory=False)
                    st.session_state.expr_name, st.session_state.meta_name = (
                        expr_file.name, meta_file.name
                    )

        expr_df, meta_df = st.session_state.expr_df, st.session_state.meta_df
        if expr_df is not None and meta_df is not None:
            markers = [c for c in expr_df.columns if c != "cell_id"]
            if "batch" in meta_df.columns:
                batches = meta_df["batch"].unique().tolist()
                all_b = st.checkbox(
                    "All batches", True, key="chk_b",
                    help="Use every batch in the dataset."
                )
                sel_b = batches if all_b else st.multiselect(
                    "Batch(es)", batches,
                    help="Select which batches to include."
                )
                st.session_state.sel_batches = sel_b
                meta_use = (meta_df if (all_b or not sel_b)
                            else meta_df[meta_df["batch"].isin(sel_b)])
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
            sel_m = markers if all_m else st.multiselect(
                "Marker(s)", markers,
                help="Choose specific markers to process."
            )
            sel_s = samples if all_s else st.multiselect(
                "Sample(s)", samples,
                help="Choose specific samples to process."
            )
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
        "Min peak separation", 0.0, 10.0, 0.7, 0.1,
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

    st.checkbox(
        "Enforce marker consistency across samples",
        key="apply_consistency",
        help="Use the same marker selection for all samples."
    )

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

        if stems_for_override:
            if bw_mode == "Manual":
                if isinstance(bw_val, (int, float)):
                    bw_label = f"Manual ({bw_val:.2f})"
                else:
                    bw_label = f"Manual ({bw_opt})"
            else:
                bw_label = "GPT automatic"

            _render_pre_run_overrides(
                stems_for_override,
                bw_label=bw_label,
                prom_mode=prom_mode,
                prom_default=prom_val,
            )
        else:
            st.session_state.pre_overrides.clear()

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


# ───────────────── main buttons & global progress bar ────────────────────────
run_col, clear_col, pause_col = st.columns(3)
if clear_col.button("Clear results"):
    # buckets that are always dict-like
    for bucket in ("results", "fig_pngs", "params", "dirty",
                   "aligned_results", "aligned_fig_pngs",
                   "results_raw",          # ← keep as dict, not None
                   ):
        st.session_state[bucket] = {}       # {} stays {}

    # scalar keys or arrays
    for key in ("aligned_counts", "aligned_landmarks",
                "aligned_ridge_png"):
        st.session_state[key] = None

    st.session_state.generated_csvs.clear()

    st.session_state.pending.clear()
    st.session_state.total_todo = 0
    st.session_state.run_active = False
    st.rerun()

run_clicked = run_col.button("Run detector")

# pause button (if run is active)
pause_label = "Pause" if st.session_state.run_active else "Resume"
pause_disabled = not bool(st.session_state.pending)
pause_clicked = pause_col.button(pause_label, disabled=pause_disabled)

if pause_clicked:
    if st.session_state.run_active:
        # Pause mid-batch
        st.session_state.run_active = False
        st.session_state.paused     = True
        st.rerun()
    elif st.session_state.paused:
        # Resume where you left off
        st.session_state.run_active = True
        st.session_state.paused     = False
        st.rerun()

# progress bar placeholder (top-level, reused)
prog_placeholder = st.empty()
if st.session_state.total_todo:
    done = st.session_state.total_todo - len(st.session_state.pending)
    prog_placeholder.progress(done / st.session_state.total_todo,
                              f"Processing… {done}/{st.session_state.total_todo}")


# ───────────────────────────── incremental processing ───────────────────────
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

    if not csv_files:
        st.error("No CSV files selected."); st.stop()

    need_gpt = ((n_fixed is None) or prom_mode == "GPT automatic"
                or bw_mode == "GPT automatic")
    if need_gpt and not api_key:
        st.error("GPT-based options need an OpenAI key."); st.stop()

    todo = [f for f in csv_files
            if (stem := Path(f.name).stem) not in st.session_state.results
               or st.session_state.dirty.get(stem, False)]

    st.session_state.pending    = todo
    st.session_state.total_todo = len(todo)
    st.session_state.run_active = bool(todo)
    st.rerun()

# 2 Queue active → process ONE file then rerun
if st.session_state.run_active and st.session_state.pending:
    f      = st.session_state.pending.pop(0)
    stem   = Path(f.name).stem
    marker = getattr(f, "marker", None)
    cnts   = read_counts(f, header_row, skip_rows)
    if st.session_state.get("apply_arcsinh", True) and not getattr(f, "arcsinh", False):
        cnts = arcsinh_transform(
            cnts,
            a=st.session_state.get("arcsinh_a", 1.0),
            b=st.session_state.get("arcsinh_b", 1 / 5),
            c=st.session_state.get("arcsinh_c", 0.0),
        )
    st.session_state.results_raw.setdefault(stem, cnts)

    if st.session_state.dirty.get(stem, False):
        over = st.session_state.params.get(stem, {})
    elif stem in st.session_state.results:
        over = {}
    else:
        over = st.session_state.pre_overrides.get(stem, {})

    bw_raw = over.get("bw")
    bw_over: float | None
    bw_rule: str | None
    if isinstance(bw_raw, (int, float)) and not np.isnan(bw_raw):
        bw_over = float(bw_raw)
        bw_rule = None
    elif isinstance(bw_raw, str) and bw_raw:
        try:
            bw_over = float(bw_raw)
            bw_rule = None
        except ValueError:
            bw_over = None
            bw_rule = bw_raw
    else:
        bw_over = None
        bw_rule = None

    prom_raw = over.get("prom", "")
    try:
        pr_over = float(prom_raw) if prom_raw not in ("", None) else None
    except Exception:
        pr_over = None

    k_over = over.get("n_peaks")
    if k_over in ("", None):
        k_over = None
    elif isinstance(k_over, str):
        k_over = int(k_over) if k_over.isdigit() else None
    elif isinstance(k_over, (int, float)):
        k_over = int(k_over)

    client = OpenAI(api_key=api_key) if api_key else None

    # bandwidth
    if bw_over is not None:
        bw_use = bw_over
    elif bw_rule is not None:
        bw_use = bw_rule
    elif bw_val is not None:
        bw_use = bw_val
    else:
        expected = (k_over if k_over is not None else
                    n_fixed if n_fixed is not None else max_peaks)
        if client is None:
            bw_use = 'scott'
        else:
            try:
                bw_use = ask_gpt_bandwidth(
                    client, gpt_model, cnts, peak_amount=expected, default='scott'
                )
            except AuthenticationError:
                if not st.session_state.invalid_api_key:
                    st.warning("Invalid OpenAI API key; please update it to enable GPT features.")
                    st.session_state.invalid_api_key = True
                st.stop()

    # prominence
    if pr_over is not None:
        prom_use = pr_over
    elif prom_val is not None:
        prom_use = prom_val
    else:
        if client is None:
            prom_use = 0.05
        else:
            try:
                prom_use = ask_gpt_prominence(
                    client, gpt_model, cnts, default=0.05
                )
            except AuthenticationError:
                if not st.session_state.invalid_api_key:
                    st.warning("Invalid OpenAI API key; please update it to enable GPT features.")
                    st.session_state.invalid_api_key = True
                st.stop()

    # peak count
    if k_over is not None:                  # manual override from per-file form
        n_use = int(k_over)
    elif n_fixed is not None:               # fixed via sidebar selector
        n_use = n_fixed
    else:                                   # GPT automatic
        if client is None:
            n_use = None
        else:
            try:
                n_use = ask_gpt_peak_count(
                    client, gpt_model, max_peaks, counts_full=cnts,
                    marker_name=marker
                )
            except AuthenticationError:
                if not st.session_state.invalid_api_key:
                    st.warning("Invalid OpenAI API key; please update it to enable GPT features.")
                    st.session_state.invalid_api_key = True
                st.stop()
        if n_use is None:                   # fallback to heuristic
            n_est, confident = quick_peak_estimate(
                cnts, prom_use, bw_use, min_w or None, grid_sz
            )
            n_use = n_est if confident else None

    if n_use is None:
        n_use = max_peaks
    else:
        n_use = min(n_use, max_peaks)

    peaks, valleys, xs, ys = kde_peaks_valleys(
        cnts, n_use, prom_use, bw_use, min_w or None, grid_sz,
        drop_frac=val_drop / 100.0,
        min_x_sep=min_sep,
        curvature_thresh = curv if curv > 0 else None,
        turning_peak     = tp,
        first_valley     = "drop" if val_mode == "Valley drop" else "slope",
    )

    if len(peaks) == 1 and not valleys:
        p_idx = np.searchsorted(xs, peaks[0])
        y_pk  = ys[p_idx]
        drop  = np.where(ys[p_idx:] < (val_drop / 100) * y_pk)[0]
        if drop.size:
            valleys = [float(xs[p_idx + drop[0]])]

    # quality
    qual = stain_quality(cnts, peaks, valleys)

    st.session_state.results[stem] = {
        "peaks": peaks,
        "valleys": valleys,
        "quality": qual,                             #  store it
        "xs": xs.tolist(),
        "ys": ys.tolist(),
        "marker": marker,
    }
    st.session_state.dirty[stem] = False

    # draw only this sample for now; full consistency applied later
    info1 = st.session_state.results[stem]
    xs1 = np.asarray(info1.get("xs", []), float)
    ys1 = np.asarray(info1.get("ys", []), float)
    pk1 = info1.get("peaks", [])
    vl1 = info1.get("valleys", [])
    st.session_state.fig_pngs[f"{stem}.png"] = _plot_png(
        stem, xs1, ys1, pk1, vl1
    )
    st.session_state[f"{stem}__pk_list"] = pk1.copy()
    st.session_state[f"{stem}__vl_list"] = vl1.copy()

    _refresh_raw_ridge()

    st.session_state.params[stem] = {
        "bw": bw_use,
        "prom": prom_use,
        "n_peaks": n_use,
    }
    st.session_state.pre_overrides.pop(stem, None)
    st.session_state.dirty[stem] = False

    # progress update
    done = st.session_state.total_todo - len(st.session_state.pending)
    prog_placeholder.progress(done / st.session_state.total_todo,
                              f"Processing… {done}/{st.session_state.total_todo}")

    # finished queue?
    if not st.session_state.pending:
        if st.session_state.get("apply_consistency", True):
            enforce_marker_consistency(st.session_state.results)
            for stem2, info2 in st.session_state.results.items():
                xs2 = np.asarray(info2.get("xs", []), float)
                ys2 = np.asarray(info2.get("ys", []), float)
                pk2 = info2.get("peaks", [])
                vl2 = info2.get("valleys", [])
                st.session_state.fig_pngs[f"{stem2}.png"] = _plot_png(
                    stem2, xs2, ys2, pk2, vl2
                )
                st.session_state[f"{stem2}__pk_list"] = pk2.copy()
                st.session_state[f"{stem2}__vl_list"] = vl2.copy()
            _refresh_raw_ridge()
        st.session_state.run_active = False
        st.success("All files processed!")

    # show current results & run again if queue not empty
    if st.session_state.run_active:
        results_container = st.container()
        render_results(results_container)
        st.rerun()
        st.stop()

    # ── build a 'RAW' stacked ridge plot once we have ≥1 samples ─────────
    if st.session_state.results and st.session_state.get("raw_ridge_png") is None:
        _refresh_raw_ridge()


# ────────────────────────── static results & download ────────────────────────
results_container = st.container()
render_results(results_container)

# ──────────────  SUMMARY  +  RIDGE-PLOT TABS  ──────────────
if st.session_state.results:
    # summary table
    df = pd.DataFrame(
        [{
            "file":     k,
            "peaks":    v["peaks"],
            "valleys":  v["valleys"],
            "quality":  round(v.get("quality", np.nan), 4),
        } for k, v in st.session_state.results.items()]
    )

    # ZIP with per-sample PNGs + summary CSV  (same as before)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("summary.csv", df.to_csv(index=False).encode())
        for fn, png in st.session_state.fig_pngs.items():
            z.writestr(fn, png)
    buf.seek(0)                             #  ← be sure to rewind!

    # quality table for the extra download button
    qual_df = df[["file", "quality"]]

else:
    df = pd.DataFrame()                     # empty placeholders
    buf = io.BytesIO()
    qual_df = pd.DataFrame()

tab_sum, tab_quality, tab_cmp = st.tabs(
    ["Summary ∣ downloads",
    "Quality",
    "Comparison"]
)

# TAB 1  – summary table & the three download buttons
with tab_sum:
    if not df.empty:
        st.dataframe(df, use_container_width=True)

        # the two extra download buttons you already kept:
        st.download_button("Download per-sample xs / ys",
                   _make_curves_zip(),
                   "SampleCurves.zip",
                   mime="application/zip",
                   key="curves_dl_tab")
        
        if st.session_state.aligned_results:
            st.download_button(
                "Download Aligned Data",
                _make_aligned_zip(),
                "alignedData.zip",
                mime="application/zip",
                key="aligned_dl_tab",
            )

        st.download_button("Download quality table",
                   qual_df.to_csv(index=False).encode(),
                   "StainQuality.csv",
                   "text/csv",
                   key="qual_dl_tab")
    else:
        st.info("Run the detector first to see summary & downloads.")
        # the two extra download buttons stay where they are
        # (nothing else to change here)

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
            peaks_all   = [v["peaks"]   for v in st.session_state.results.values()]
            valleys_all = [v["valleys"] for v in st.session_state.results.values()]
            counts_all  = [st.session_state.results_raw[k]
                        for k in st.session_state.results]

            # —— 1. warp every distribution ---------------------------------------
            # -------- deterministically build + fill landmark matrix ----------------
            landmark_mat = fill_landmark_matrix(
                peaks   = peaks_all,
                valleys = valleys_all,
                align_type  = align_mode,
                midpoint_type        = "valley",          # valley-based fallback
            )

            # choose the *target* positions
            if target_vec is None:                      # ► AUTOMATIC
                target_landmark = np.nanmedian(landmark_mat, axis=0).tolist()
            else:                                       # ► USER-SUPPLIED
                need = _cols_for_align_mode(align_mode)
                if len(target_vec) != len(need):
                    st.error(
                        f"Need {len(need)} values for '{align_mode}', "
                        f"but got {len(target_vec)}."
                    )
                    st.stop()
                target_landmark = target_vec            # <- exactly as typed

            # -------- run the warp, forcing ALL THREE landmarks ---------------------
            warped, warped_lm, warp_funs = align_distributions(
                counts_all,
                peaks_all,            # unused internally when landmark_mat supplied,
                valleys_all,          # but kept for API compatibility
                align_type      = align_mode,
                landmark_matrix = landmark_mat,
                target_landmark = target_landmark,
            )

            warped      = list(warped)                    # list of 1-D arrays
            warp_funs   = list(warp_funs)                 # list of callables
            warped_lm   = [list(row) for row in warped_lm]  # list of [neg, val, pos]

            n_samples   = len(st.session_state.results)

            # cohort mean location of the negative peak among samples that did warp
            neg_peaks   = [lm[0] for lm in warped_lm]          # first column = neg-peak
            target_neg  = float(np.nanmean(neg_peaks)) if neg_peaks else 0.0

            for idx in range(n_samples):
                if idx >= len(warp_funs):
                    # sample idx was dropped  → fabricate a simple shift warp
                    pks      = peaks_all[idx]
                    offset   = (target_neg - pks[0]) if pks else 0.0
                    warp_funs.append(lambda x, o=offset: x + o)
                    warped.append(counts_all[idx] + offset)
                    warped_lm.append([
                        (pks[0] + offset) if pks else np.nan,  # neg-peak
                        np.nan,                                # missing valley
                        np.nan                                 # missing pos-peak
                    ])

            # build aligned curves directly from the already-computed KDEs
            # to avoid bandwidth discrepancies between raw and aligned views
            xs_map: dict[str, np.ndarray] = {}
            ys_map: dict[str, np.ndarray] = {}
            all_xmin = np.inf
            all_xmax = -np.inf
            all_ymax = 0.0
            for stem, f in zip(st.session_state.results, warp_funs):
                xs_raw = np.asarray(st.session_state.results[stem].get("xs", []), float)
                ys_raw = np.asarray(st.session_state.results[stem].get("ys", []), float)
                xs_al  = f(xs_raw)
                xs_map[stem] = xs_al
                ys_map[stem] = ys_raw
                if xs_al.size:
                    all_xmin = min(all_xmin, float(xs_al.min()))
                    all_xmax = max(all_xmax, float(xs_al.max()))
                if ys_raw.size:
                    all_ymax = max(all_ymax, float(ys_raw.max()))

            if not np.isfinite(all_xmin) or not np.isfinite(all_xmax):
                all_xmin, all_xmax = 0.0, 1.0
            elif all_xmax == all_xmin:
                d = abs(all_xmin) if all_xmin != 0 else 1.0
                all_xmin -= 0.05 * d
                all_xmax += 0.05 * d

            span = all_xmax - all_xmin
            pad  = 0.05 * span
            if not np.isfinite(all_ymax):
                all_ymax = 1.0

            st.session_state.aligned_counts     = dict(zip(st.session_state.results,
                                                        warped))
            st.session_state.aligned_landmarks  = warped_lm
            st.session_state.aligned_results    = {}
            st.session_state.aligned_fig_pngs   = {}
            st.session_state.aligned_ridge_png  = None

            # —— 2. per-sample PNGs & metadata ------------------------------------
            for stem, f in zip(st.session_state.results, warp_funs):
                xs      = xs_map[stem]
                ys      = ys_map[stem]
                pk_align = f(np.asarray(st.session_state.results[stem]["peaks"]))
                vl_align = f(np.asarray(st.session_state.results[stem]["valleys"]))

                png = _plot_png_fixed(
                    f"{stem} (aligned)", xs, ys,
                    pk_align[~np.isnan(pk_align)],
                    vl_align[~np.isnan(vl_align)],
                    (all_xmin - pad, all_xmax + pad),
                    all_ymax
                )

                st.session_state.aligned_fig_pngs[f"{stem}_aligned.png"] = png
                st.session_state.aligned_results[stem] = {
                    "peaks":   pk_align.round(4).tolist(),
                    "valleys": vl_align.round(4).tolist(),
                    "xs": xs.tolist(),
                    "ys": ys.tolist(),
                }

            # --- 3. stacked ridge-plot (plain) -----------------------------------
            gap = 1.2 * all_ymax
            fig, ax = plt.subplots(
                figsize=(6, 0.8 * len(st.session_state.results)),
                dpi=150,
                sharex=True,
            )

            for i, stem in enumerate(st.session_state.results):
                xs     = xs_map[stem]
                ys     = ys_map[stem]
                offset = i * gap

                ax.plot(xs, ys + offset, color="black", lw=1)
                ax.fill_between(xs, offset, ys + offset,
                                color="#FFA50088", lw=0)

                info = st.session_state.aligned_results[stem]
                for p in info["peaks"]:
                    ax.vlines(p, offset, offset + ys.max(), color="black", lw=0.8)
                for v in info["valleys"]:
                    ax.vlines(v, offset, offset + ys.max(), color="grey",
                              lw=0.8, linestyles=":")

                ax.text(all_xmin, offset + 0.5 * all_ymax, stem,
                        ha="right", va="center", fontsize=7)

            ax.set_yticks([]); ax.set_xlim(all_xmin, all_xmax)
            fig.tight_layout()

            st.session_state.aligned_ridge_png = fig_to_png(fig)
            plt.close(fig)

            st.success("Landmarks aligned – scroll down for the stacked view or download the ZIP!")
            st.rerun()
