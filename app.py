# app.py  – GPT-assisted bandwidth detector with
#           live incremental results + per-sample overrides
from __future__ import annotations
import io, zipfile, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

from peak_valley.quality import stain_quality

from peak_valley import (
    arcsinh_transform, read_counts,
    kde_peaks_valleys, quick_peak_estimate,
    fig_to_png,
)
from peak_valley.gpt_adapter import (
    ask_gpt_peak_count, ask_gpt_prominence, ask_gpt_bandwidth,
)
from peak_valley.alignment import align_distributions, fill_landmark_matrix

# ────────────────────────── Streamlit page & state ──────────────────────────
st.set_page_config("Peak & Valley Detector", "🔬", layout="wide")
st.title("🔬 Peak & Valley Detector — CSV *or* full dataset")

for key, default in {
    "results":     {},         # stem → {"peaks":…, "valleys":…, "xs":…, "ys":…}
    "results_raw": {},         # stem → raw counts (np.ndarray)
    "fig_pngs":    {},         # stem.png → png bytes (latest)
    "params":      {},         # stem → {"bw":…, "prom":…, "n_peaks":…}
    "dirty":       {},         # stem → True if user edited params *or* positions
    "cached_uploads": [],
    "generated_csvs": [],
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
    "aligned_ridge_png":    None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── helper #1 : aligned ZIP  (plots + aligned counts) ───────────────
def _make_aligned_zip() -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        # aligned counts
        for stem, arr in st.session_state.aligned_counts.items():
            bio = io.BytesIO(); np.savetxt(bio, arr, delimiter=",")
            z.writestr(f"{stem}_aligned.csv", bio.getvalue())
        # ridge & per-sample aligned plots
        for fn, png in st.session_state.aligned_fig_pngs.items():
            z.writestr(fn, png)
        if st.session_state.aligned_ridge_png:
            z.writestr("aligned_ridge.png",
                       st.session_state.aligned_ridge_png)
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
    pad = 0.05 * (xs.max() - xs.min())
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
    ax.plot(xs, ys, color="skyblue"); ax.fill_between(xs, 0, ys, color="#87CEEB88")
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
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
    xmin, xmax = float(xs.min()), float(xs.max())

    pk_key, vl_key = f"{stem}__pk_list", f"{stem}__vl_list"
    if pk_key not in st.session_state:
        st.session_state[pk_key] = info["peaks"].copy()
    if vl_key not in st.session_state:
        st.session_state[vl_key] = info["valleys"].copy()

    pk_list = st.session_state[pk_key]
    vl_list = st.session_state[vl_key]

    ### one single column for sliders + buttons
    colL, colR = st.columns([3, 5])

    # ---------- sliders & delete buttons (NO nested columns!) ----------
    with colL:
        st.subheader("Peaks")
        i = 0
        while i < len(pk_list):
            pk_list[i] = st.slider(
                f"Peak #{i+1}", xmin, xmax, float(pk_list[i]), 0.01,
                key=f"{stem}_pk_slider_{i}"
            )
            if st.button(f"❌ Delete peak #{i+1}", key=f"{stem}_pk_del_{i}"):
                pk_list.pop(i)
                st.rerun()
            else:
                i += 1

        if st.button("➕ Add peak", key=f"{stem}_add_pk"):
            pk_list.append((xmin + xmax) / 2)
            st.rerun()

        st.divider()
        st.subheader("Valleys")
        i = 0
        while i < len(vl_list):
            vl_list[i] = st.slider(
                f"Valley #{i+1}", xmin, xmax, float(vl_list[i]), 0.01,
                key=f"{stem}_vl_slider_{i}"
            )
            if st.button(f"❌ Delete valley #{i+1}", key=f"{stem}_vl_del_{i}"):
                vl_list.pop(i)
                st.rerun()
            else:
                i += 1

        if st.button("➕ Add valley", key=f"{stem}_add_vl"):
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
    if st.button("✅ Apply changes", key=apply_key):
        st.session_state.results[stem]["peaks"]   = pk_list.copy()
        st.session_state.results[stem]["valleys"] = vl_list.copy()
        st.session_state.dirty[stem] = True       # mark for slow re-run
        st.rerun()

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
    container.header("🔧 Aligned distributions")
    if not st.session_state.aligned_results:
        container.info("Run alignment first."); return

    for stem, info in st.session_state.aligned_results.items():
        with container.expander(stem, expanded=False):
            st.image(st.session_state.aligned_fig_pngs.get(f"{stem}_aligned.png", b""),
                     use_container_width=True)
            st.write(f"**Peaks (after warp):** {info['peaks']}")
            st.write(f"**Valleys (after warp):** {info['valleys']}")

def render_results(container):
    container.header("📊 Processed datasets")
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
                st.image(st.session_state.fig_pngs.get(f"{stem}.png", b""),
                        use_container_width=True)
                st.write(f"**Peaks:** {info['peaks']}")
                st.write(f"**Valleys:** {info['valleys']}")
                q = info.get("quality", np.nan)
                if np.isfinite(q):
                    st.write(f"**Stain-quality score:** {q:.4f}")
            with tab_params:
                bw_new = st.text_input("Bandwidth", bw0, key=f"{stem}_bw")
                pr_new = st.text_input("Prominence", pr0, key=f"{stem}_pr")
                k_new = st.number_input("# peaks", 1, 6, k0, key=f"{stem}_k")
                if (bw_new != bw0) or (pr_new != pr0) or (k_new != k0):
                    st.session_state.params[stem] = {"bw": bw_new, "prom": pr_new, "n_peaks": k_new}
                    st.session_state.dirty[stem] = True
            with tab_manual:
                if not info.get("xs"):
                    st.info("Run detector first to enable manual editing.")
                else:
                    _manual_editor(stem)
        # delete button ------------------------------------------------------
        if rowR.button("❌", key=f"del_{stem}"):
            for bucket in ("results", "fig_pngs", "params", "dirty"):
                st.session_state[bucket].pop(stem, None)
            for k in (f"{stem}__pk_list", f"{stem}__vl_list"):
                st.session_state.pop(k, None)          # clear per‑dataset edits
            st.rerun()


# ─────────────────────────────── SIDEBAR ─────────────────────────────────────
with st.sidebar:
    mode = st.radio("Choose mode", ["Counts CSV files", "Whole dataset"])

    # 1️⃣  Counts-CSV workflow
    if mode == "Counts CSV files":
        uploaded_now = st.file_uploader(
            "Upload *_raw_counts.csv*", type=["csv"],
            accept_multiple_files=True, key="csv_up"
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
            pick  = st.multiselect("Choose uploaded files", stems, stems,
                                   key="pick_up2")
            for f in st.session_state.cached_uploads:
                if Path(f.name).stem in pick:
                    use_uploads.append(f)
            if st.button("🗑 Clear cached uploads"):
                st.session_state.cached_uploads.clear(); st.rerun()

        use_generated: list[io.BytesIO] = []
        if st.session_state.generated_csvs:
            st.markdown("**Generated CSVs (from dataset)**")
            stems_g = [s for s, _ in st.session_state.generated_csvs]
            pick_g  = st.multiselect("Choose generated files", stems_g, stems_g,
                                     key="pick_gen2")
            for stem, bio in st.session_state.generated_csvs:
                if stem in pick_g:
                    bio.seek(0); bio.name = f"{stem}_raw_counts.csv"
                    use_generated.append(bio)

        header_row = st.number_input("Header row (−1 = none)", 0, step=1,
                                     key="hdr")
        skip_rows  = st.number_input("Rows to skip", 0, step=1, key="skip")

    # 2️⃣  Whole-dataset workflow
    else:
        expr_file = st.file_uploader("expression_matrix_combined.csv",
                                     type=["csv"])
        meta_file = st.file_uploader("cell_metadata_combined.csv",
                                     type=["csv"])

        if st.session_state.expr_df is not None:
            if st.button("🗑 Clear loaded dataset"):
                for k in ("expr_df", "meta_df", "expr_name", "meta_name"):
                    st.session_state[k] = None
                st.rerun()

        if expr_file and meta_file:
            need = (st.session_state.expr_df is None or
                    expr_file.name != st.session_state.expr_name or
                    meta_file.name != st.session_state.meta_name)
            if need:
                with st.spinner("⌛ Parsing expression / metadata …"):
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
            samples = meta_df["sample"].unique().tolist()

            all_m = st.checkbox("All markers", False, key="chk_m")
            all_s = st.checkbox("All samples", False, key="chk_s")
            sel_m = markers if all_m else st.multiselect("Marker(s)", markers)
            sel_s = samples if all_s else st.multiselect("Sample(s)", samples)

            if sel_m and sel_s and st.button("Generate counts CSVs"):
                tot = len(sel_m) * len(sel_s)
                bar = st.progress(0.0, "Generating …")
                exist = {s for s, _ in st.session_state.generated_csvs}
                for i, m in enumerate(sel_m, 1):
                    for j, s in enumerate(sel_s, 1):
                        idx  = (i - 1) * len(sel_s) + j
                        stem = f"{s}_{m}"
                        if stem in exist:
                            bar.progress(idx / tot,
                                         f"Skip {stem} (exists)")
                            continue
                        counts = arcsinh_transform(
                            expr_df.loc[meta_df["sample"].eq(s), m]
                        )
                        bio = io.BytesIO()
                        counts.to_csv(bio, index=False, header=False)
                        bio.seek(0); bio.name = f"{stem}_raw_counts.csv"
                        st.session_state.generated_csvs.append((stem, bio))
                        bar.progress(idx / tot,
                                     f"Added {stem} ({idx}/{tot})")
                bar.empty()
                st.success("✓ CSVs cached – switch to **Counts CSV files**")

        header_row, skip_rows = -1, 0
        use_uploads, use_generated = [], []

    # ───────────── Detection options ──────────────
    st.markdown("---\n### Detection")
    auto = st.selectbox("Number of peaks",
                        ["Automatic", 1, 2, 3, 4, 5, 6])
    n_fixed = None if auto == "Automatic" else int(auto)
    cap_min = n_fixed if n_fixed else 1
    max_peaks = st.number_input("Maximum peaks (Automatic cap)",
                                cap_min, 6, max(2, cap_min), step=1,
                                disabled=(n_fixed is not None))

    # Bandwidth
    bw_mode = st.selectbox("Bandwidth mode",
                           ["Manual", "GPT automatic"])
    if bw_mode == "Manual":
        bw_opt = st.selectbox("Rule / scale",
                              ["scott", "silverman",
                               "0.5", "0.8", "1.0"],
                              key="bw_sel")
        bw_val = (float(bw_opt)
                  if bw_opt.replace(".", "", 1).isdigit() else bw_opt)
    else:
        bw_val = None  # GPT later

    # Prominence
    prom_mode = st.selectbox("Prominence",
                             ["Manual", "GPT automatic"], key="prom_sel")
    prom_val = (st.slider("Prominence value", 0.00, 0.30, 0.05, 0.01)
                if prom_mode == "Manual" else None)

    min_w    = st.slider("Min peak width", 0, 6, 0, 1)
    curv = st.slider("Curvature thresh (0 = off)", 0.0000, 0.005, 0.0001, 0.0001)
    tp   = st.checkbox("Treat concave-down turning points as peaks", False)
    min_sep   = st.slider("Min peak separation", 0.0, 10.0, 0.7, 0.1)
    grid_sz  = st.slider("Max KDE grid", 4_000, 40_000, 20_000, 1_000)
    val_drop = st.slider("Valley drop (% of peak)", 1, 50, 10, 1)

    # ───────────── Alignment options ──────────────
    st.markdown("---\n### Alignment")

    align_mode = st.selectbox(
        "Landmark set",
        ["negPeak_valley_posPeak", "negPeak_valley", "negPeak", "valley"],
        index=0, key="align_mode",
    )

    col_names = _cols_for_align_mode(
        st.session_state.get("align_mode", "negPeak_valley_posPeak")
    )

    # ► choose where the *aligned* landmarks should end up
    target_mode = st.radio(
        "Target landmark positions",
        ["Automatic (median across samples)", "Custom (enter numbers)"],
        horizontal=True, key="target_mode",
    )

    def _ask_numbers(labels: list[str], defaults: list[float], prefix: str) -> list[float]:
        vals = []
        for lab, d, i in zip(labels, defaults, range(len(labels))):
            v = st.number_input(
                f"Target {lab}", value=float(d), key=f"{prefix}_{i}"
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
    pick = st.selectbox("Model",
                        ["o4-mini", "gpt-4o-mini",
                         "gpt-4-turbo-preview", "Custom"])
    gpt_model = st.text_input("Custom model") if pick == "Custom" else pick
    api_key   = st.text_input("OpenAI API key", type="password")


# ───────────────── main buttons & global progress bar ────────────────────────
run, clear_all = st.columns(2)
if clear_all.button("🗑 Clear results"):
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

    st.session_state.pending.clear()
    st.session_state.total_todo = 0
    st.session_state.run_active = False
    st.rerun()

run_clicked = run.button("🚀 Run detector")

# progress bar placeholder (top-level, reused)
prog_placeholder = st.empty()
if st.session_state.total_todo:
    done = st.session_state.total_todo - len(st.session_state.pending)
    prog_placeholder.progress(done / st.session_state.total_todo,
                              f"Processing… {done}/{st.session_state.total_todo}")


# ───────────────────────────── incremental processing ───────────────────────
# 1️⃣ User clicked RUN: prepare pending queue
if run_clicked and not st.session_state.run_active:
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

# 2️⃣ Queue active → process ONE file then rerun
if st.session_state.run_active and st.session_state.pending:
    f     = st.session_state.pending.pop(0)
    stem  = Path(f.name).stem
    cnts  = read_counts(f, header_row, skip_rows)
    st.session_state.results_raw.setdefault(stem, cnts)

    over  = st.session_state.params.get(stem, {}) \
            if st.session_state.dirty.get(stem, False) else {}
    try:    bw_over = float(over.get("bw", "")) or None
    except Exception: bw_over = None
    try:    pr_over = float(over.get("prom", "")) or None
    except Exception: pr_over = None
    k_over  = over.get("n_peaks", None)
    if k_over in ("", None): k_over = None

    # bandwidth
    if bw_over is not None:
        bw_use = bw_over
    elif bw_val is not None:
        bw_use = bw_val
    else:
        expected = (k_over if k_over is not None else
                    n_fixed if n_fixed is not None else max_peaks)
        bw_use = ask_gpt_bandwidth(
            OpenAI(api_key=api_key) if api_key else None,
            gpt_model, cnts, peak_amount=expected, default='scott'
        )

    # prominence
    if pr_over is not None:
        prom_use = pr_over
    elif prom_val is not None:
        prom_use = prom_val
    else:
        prom_use = ask_gpt_prominence(
            OpenAI(api_key=api_key) if api_key else None,
            gpt_model, cnts, default=0.05
        )

    # peak count
    if k_over is not None:
        n_use = int(k_over)
    elif n_fixed is None:
        n_est, confident = quick_peak_estimate(
            cnts, prom_use, bw_use, min_w or None, grid_sz
        )
        n_use = n_est if confident else None
        if n_use is not None:
            n_use = min(n_use, max_peaks)
    else:
        n_use = n_fixed

    if n_use is None and n_fixed is None:
        def _infer_marker(stem_: str) -> str | None:
            if stem_.endswith("_raw_counts"):
                stem_ = stem_[:-11]
            parts = stem_.split("_")
            return parts[-1] if parts else None
        n_use = ask_gpt_peak_count(
            OpenAI(api_key=api_key) if api_key else None,
            gpt_model, max_peaks, counts_full=cnts,
            marker_name=_infer_marker(stem)
        )
    if n_use is None:
        n_use = max_peaks

    peaks, valleys, xs, ys = kde_peaks_valleys(
        cnts, n_use, prom_use, bw_use, min_w or None, grid_sz,
        drop_frac=val_drop / 100.0,
        min_x_sep=min_sep,
        curvature_thresh = curv if curv > 0 else None,
        turning_peak     = tp
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
        "quality": qual,                             # ★ store it
        "xs": xs.tolist(),
        "ys": ys.tolist(),
    }

    # plot
    pad = 0.05 * (xs.max() - xs.min())
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
    cL, cF = ("skyblue", "#87CEEB88")
    ax.plot(xs, ys, color=cL); ax.fill_between(xs, 0, ys, color=cF)
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    for p in peaks:   ax.axvline(p, color="red",   ls="--", lw=1)
    for v in valleys: ax.axvline(v, color="green", ls=":",  lw=1)
    ax.set_xlabel("Arcsinh counts"); ax.set_ylabel("Density")
    ax.set_title(stem, fontsize=9); fig.tight_layout()

    st.session_state.fig_pngs[f"{stem}.png"] = fig_to_png(fig)
    plt.close(fig)

    st.session_state[f"{stem}__pk_list"] = peaks.copy()
    st.session_state[f"{stem}__vl_list"] = valleys.copy()
    st.session_state.params [stem] = {
        "bw": bw_use, "prom": prom_use, "n_peaks": n_use,
    }
    st.session_state.dirty  [stem] = False

    # progress update
    done = st.session_state.total_todo - len(st.session_state.pending)
    prog_placeholder.progress(done / st.session_state.total_todo,
                              f"Processing… {done}/{st.session_state.total_todo}")

    # finished queue?
    if not st.session_state.pending:
        st.session_state.run_active = False
        st.success("All files processed!")

    # show current results & run again if queue not empty
    if st.session_state.run_active:
        results_container = st.container()
        render_results(results_container)
        st.rerun()
        st.stop()

# ────────────────────────── static results & download ────────────────────────
results_container = st.container()
render_results(results_container)

# aligned_container = st.container()
# render_aligned(aligned_container)

if st.session_state.aligned_ridge_png:
    st.subheader("📈 Stacked ridge-plot (aligned)")
    st.image(st.session_state.aligned_ridge_png,
             use_container_width=True)

if st.session_state.results:

    df = pd.DataFrame(
        [{"file": k,
        "peaks": v["peaks"],
        "valleys": v["valleys"],
        "quality": round(v.get("quality", np.nan), 4)}    #  ← NEW col
        for k, v in st.session_state.results.items()]
    )
    st.subheader("📋 Summary")
    st.dataframe(df, use_container_width=True)

    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("summary.csv", df.to_csv(index=False).encode())
            for fn, png in st.session_state.fig_pngs.items():
                z.writestr(fn, png)
        st.download_button("⬇️ Download ZIP", buf.getvalue(),
                           "PeakValleyResults.zip", "application/zip")


if st.session_state.results:
    st.markdown("---")
    align_col, dl_col = st.columns([2, 1])
    with align_col:
        do_align = st.button("🔧 Align landmarks & normalise counts",
                             type="primary")
    if do_align:
        with st.spinner("⌛ Running landmark alignment …"):
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

            # global finite x-range ------------------------------------------------
            all_xmin = float(np.nanmin([np.nanmin(w) for w in warped]))
            all_xmax = float(np.nanmax([np.nanmax(w) for w in warped]))
            if not np.isfinite(all_xmin) or not np.isfinite(all_xmax):
                all_xmin, all_xmax = 0.0, 1.0                 # extreme fallback
            x_grid        = np.linspace(all_xmin, all_xmax, 4000)
            std_fallback  = 0.05 * (all_xmax - all_xmin or 1.0)

            st.session_state.aligned_counts     = dict(zip(st.session_state.results,
                                                        warped))
            st.session_state.aligned_landmarks  = warped_lm
            st.session_state.aligned_results    = {}
            st.session_state.aligned_fig_pngs   = {}
            st.session_state.aligned_ridge_png  = None

            # —— 2. build KDE / fallback curve for *every* sample -----------------
            kdes = {}
            from scipy.stats import gaussian_kde
            for idx, stem in enumerate(st.session_state.results):
                wc   = warped[idx]
                good = wc[~np.isnan(wc)]
                if np.unique(good).size >= 2:
                    try:
                        kdes[stem] = gaussian_kde(good, bw_method="scott")(x_grid)
                    except Exception:  # numerical failure
                        kdes[stem] = np.exp(-(x_grid - good.mean())**2 / (2*std_fallback**2))
                else:                  # single value or empty
                    center = good[0] if good.size else 0.5*(all_xmin + all_xmax)
                    kdes[stem] = np.exp(-(x_grid - center)**2 / (2*std_fallback**2))

            all_ymax = max(ys.max() for ys in kdes.values())

            # —— 3. per-sample PNGs & metadata ------------------------------------
            for idx, stem in enumerate(st.session_state.results):
                f        = warp_funs[idx]
                ys       = kdes[stem]
                pk_align = f(np.asarray(st.session_state.results[stem]["peaks"]))
                vl_align = f(np.asarray(st.session_state.results[stem]["valleys"]))

                png = _plot_png_fixed(
                    f"{stem} (aligned)", x_grid, ys,
                    pk_align[~np.isnan(pk_align)],
                    vl_align[~np.isnan(vl_align)],
                    (all_xmin, all_xmax), all_ymax
                )

                st.session_state.aligned_fig_pngs[f"{stem}_aligned.png"] = png
                st.session_state.aligned_results[stem] = {
                    "peaks":   pk_align.round(4).tolist(),
                    "valleys": vl_align.round(4).tolist(),
                    "xs": x_grid.tolist(),
                    "ys": ys.tolist(),
                }

            # --- 4. stacked ridge-plot (plain) -----------------------------------
            gap = 1.2 * all_ymax
            fig, ax = plt.subplots(
                figsize=(6, 0.8 * len(st.session_state.results)),
                dpi=150,
                sharex=True,
            )

            for i, stem in enumerate(st.session_state.results):
                ys      = kdes[stem]
                offset  = i * gap

                # KDE curve + fill
                ax.plot(x_grid, ys + offset, color="black", lw=1)
                ax.fill_between(x_grid, offset, ys + offset,
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

            # aesthetics
            ax.set_yticks([]); ax.set_xlim(all_xmin, all_xmax)

            fig.tight_layout()
            st.session_state.aligned_ridge_png = fig_to_png(fig)
            plt.close(fig)

            ax.set_yticks([]); ax.set_xlim(all_xmin, all_xmax)
            fig.tight_layout()
            st.session_state.aligned_ridge_png = fig_to_png(fig)
            plt.close(fig)

            st.success("✓ Landmarks aligned – scroll down for the stacked view or download the ZIP!")
            st.rerun()

    # -------- download aligned results ----------------------------------
    ac = st.session_state.get("aligned_counts")
    if ac:
        with dl_col:
            st.download_button(
                "⬇️ Download aligned ZIP",
                _make_aligned_zip(),
                "AlignedDistributions.zip",
                mime="application/zip"
            )
            st.download_button(
                "⬇️ Download per-sample xs / ys",
                _make_curves_zip(),
                "SampleCurves.zip",
                mime="application/zip"
            )
            qual_df = df[["file", "quality"]]
            st.download_button("⬇️ Download quality table",
                            qual_df.to_csv(index=False).encode(),
                            "StainQuality.csv", "text/csv")