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
    fig_to_png, enforce_marker_consistency,
)
from peak_valley.gpt_adapter import (
    ask_gpt_peak_count, ask_gpt_prominence, ask_gpt_bandwidth,
)
from peak_valley.alignment import align_distributions, fill_landmark_matrix

# ────────────────────────── Streamlit page & state ──────────────────────────
st.set_page_config("Peak & Valley Detector", "🔬", layout="wide")
st.title("🔬 Peak & Valley Detector — CSV *or* full dataset")
st.warning(
    "⚠️ **Heads-up:** if you refresh or close this page, all of your uploaded data and results will be lost."
)

st.session_state.setdefault("active_sample", None)
st.session_state.setdefault("active_subtab", {})   # stem → "plot" / "params" / "manual"

st.session_state.setdefault("paused", False)

for key, default in {
    "results":     {},         # stem → {"peaks":…, "valleys":…, "xs":…, "ys":…}
    "results_raw": {},         # stem → raw counts (np.ndarray)
    "fig_pngs":    {},         # stem.png → png bytes (latest)
    "params":      {},         # stem → {"bw":…, "prom":…, "n_peaks":…}
    "dirty":       {},         # stem → True if user edited params *or* positions
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

    from scipy.stats import gaussian_kde
    counts_all = list(st.session_state.results_raw.values())

    x_min = float(min(np.nanmin(c) for c in counts_all))
    x_max = float(max(np.nanmax(c) for c in counts_all))
    pad   = 0.05 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 4000)

    # densities for every sample
    kdes = {}
    for stem, arr in st.session_state.results_raw.items():
        good = arr[~np.isnan(arr)]
        if np.unique(good).size >= 2:
            kdes[stem] = gaussian_kde(good, bw_method="scott")(x_grid)
        else:                               # fallback single value / empty
            μ = good.mean() if good.size else 0.5 * (x_min + x_max)
            σ = 0.05 * (x_max - x_min or 1.0)
            kdes[stem] = np.exp(-(x_grid - μ) ** 2 / (2 * σ ** 2))

    y_max = max(ys.max() for ys in kdes.values())
    gap   = 1.2 * y_max

    fig, ax = plt.subplots(figsize=(6, 0.8 * len(kdes)), dpi=150, sharex=True)

    for i, stem in enumerate(st.session_state.results):
        ys     = kdes[stem]
        offset = i * gap

        ax.plot(x_grid, ys + offset, color="black", lw=1)
        ax.fill_between(x_grid, offset, ys + offset, color="#FFA50088", lw=0)

        # current PEAK & VALLEY markers
        info = st.session_state.results[stem]
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

        # 1️⃣  aligned counts (one CSV per sample)
        for stem, arr in st.session_state.aligned_counts.items():
            bio = io.BytesIO()
            np.savetxt(bio, arr, delimiter=",")
            z.writestr(f"{stem}_aligned.csv", bio.getvalue())

        # 2️⃣  aligned figures (per-sample + ridge)
        for fn, png in st.session_state.aligned_fig_pngs.items():
            z.writestr(fn, png)
        if st.session_state.aligned_ridge_png:
            z.writestr("aligned_ridge.png", st.session_state.aligned_ridge_png)

        # ───────────── 3️⃣  **NEW** overall summary ─────────────
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

    If ``sel_b`` is provided, only cells belonging to those batches will
    contribute to the generated counts.
    """
    desired = {f"{s}_{m}_raw_counts": (s, m) for m in sel_m for s in sel_s}

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
    batch_mask = (meta_df["batch"].isin(sel_b)
                  if sel_b else pd.Series(True, index=meta_df.index))
    for stem, (s, m) in desired.items():
        if stem in existing:
            continue
        mask = batch_mask & meta_df["sample"].eq(s)
        counts = arcsinh_transform(
            expr_df.loc[mask, m]
        )
        bio = io.BytesIO()
        counts.to_csv(bio, index=False, header=False)
        bio.seek(0)
        bio.name = f"{stem}.csv"
        setattr(bio, "marker", m)
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
                key=f"{stem}_pk_slider_{i}"
            )
            if st.button(f"❌ Delete peak #{i+1}", key=f"{stem}_pk_del_{i}"):
                pk_list.pop(i)
                st.session_state.raw_ridge_png = None
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
        _refresh_raw_ridge()                       # ← REBUILD HERE
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
                    key=f"{stem}_bw_type"
                )
                if bw_input_type == "Preset":
                    bw_opt = st.selectbox(
                        "Preset rule",
                        ["scott", "silverman", "0.5", "0.8", "1.0"],
                        index=(["scott","silverman","0.5","0.8","1.0"].index(str(bw0))
                            if str(bw0) in ["scott","silverman","0.5","0.8","1.0"] else 0),
                        key=f"{stem}_bw_preset"
                    )
                    bw_new = bw_opt
                else:
                    bw_new = st.slider(
                        "Custom bandwidth",
                        min_value=0.01,
                        max_value=5.0,
                        value=float(bw0) if isinstance(bw0, (int, float)) or bw0.replace(".", "", 1).isdigit() else 1.0,
                        step=0.01,
                        key=f"{stem}_bw_slider"
                    )
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
                st.session_state.pop(k, None)
            _refresh_raw_ridge()                   # ← keep ridge in sync
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
                    bio.seek(0); bio.name = f"{stem}.csv"
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
            if "batch" in meta_df.columns:
                batches = meta_df["batch"].unique().tolist()
                all_b = st.checkbox("All batches", True, key="chk_b")
                sel_b = batches if all_b else st.multiselect("Batch(es)", batches)
                st.session_state.sel_batches = sel_b
                meta_use = (meta_df if (all_b or not sel_b)
                            else meta_df[meta_df["batch"].isin(sel_b)])
            else:
                st.session_state.sel_batches = []
                meta_use = meta_df
            samples = meta_use["sample"].unique().tolist()

            all_m = st.checkbox("All markers", False, key="chk_m")
            all_s = st.checkbox("All samples", False, key="chk_s")
            sel_m = markers if all_m else st.multiselect("Marker(s)", markers)
            sel_s = samples if all_s else st.multiselect("Sample(s)", samples)
            st.session_state.sel_markers = sel_m
            st.session_state.sel_samples = sel_s

            if sel_m and sel_s and st.button("Generate counts CSVs"):
                tot = len(sel_m) * len(sel_s)
                bar = st.progress(0.0, "Generating …")
                exist = {s for s, _ in st.session_state.generated_csvs}
                sel_b = st.session_state.get("sel_batches", [])
                batch_mask = (meta_df["batch"].isin(sel_b)
                              if sel_b else pd.Series(True, index=meta_df.index))
                for i, m in enumerate(sel_m, 1):
                    for j, s in enumerate(sel_s, 1):
                        idx  = (i - 1) * len(sel_s) + j
                        stem = f"{s}_{m}_raw_counts"
                        if stem in exist:
                            bar.progress(idx / tot,
                                         f"Skip {stem} (exists)")
                            continue
                        mask = batch_mask & meta_df["sample"].eq(s)
                        counts = arcsinh_transform(
                            expr_df.loc[mask, m]
                        )
                        bio = io.BytesIO()
                        counts.to_csv(bio, index=False, header=False)
                        bio.seek(0)
                        bio.name = f"{stem}.csv"
                        setattr(bio, "marker", m)
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
                        ["GPT Automatic", 1, 2, 3, 4, 5, 6])
    n_fixed = None if auto == "GPT Automatic" else int(auto)
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

    st.checkbox(
        "Enforce marker consistency across samples",
        key="apply_consistency",
    )

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
run_col, clear_col, pause_col = st.columns(3)
if clear_col.button("🗑 Clear results"):
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

run_clicked = run_col.button("🚀 Run detector")

# pause button (if run is active)
pause_label = "⏸ Pause" if st.session_state.run_active else "▶ Resume"
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
# 1️⃣ User clicked RUN: prepare pending queue
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

# 2️⃣ Queue active → process ONE file then rerun
if st.session_state.run_active and st.session_state.pending:
    f      = st.session_state.pending.pop(0)
    stem   = Path(f.name).stem
    marker = getattr(f, "marker", None)
    cnts   = read_counts(f, header_row, skip_rows)
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
    if k_over is not None:                  # manual override from per-file form
        n_use = int(k_over)
    elif n_fixed is not None:               # fixed via sidebar selector
        n_use = n_fixed
    else:                                   # GPT automatic
        n_use = ask_gpt_peak_count(
            OpenAI(api_key=api_key) if api_key else None,
            gpt_model, max_peaks, counts_full=cnts,
            marker_name=marker
        )
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
        st.download_button("⬇️ Download per-sample xs / ys",
                   _make_curves_zip(),
                   "SampleCurves.zip",
                   mime="application/zip",
                   key="curves_dl_tab")
        
        if st.session_state.aligned_results:
            st.download_button(
                "⬇️ Download Aligned Data",
                _make_aligned_zip(),
                "alignedData.zip",
                mime="application/zip",
                key="aligned_dl_tab",
            )

        st.download_button("⬇️ Download quality table",
                   qual_df.to_csv(index=False).encode(),
                   "StainQuality.csv",
                   "text/csv",
                   key="qual_dl_tab")
    else:
        st.info("Run the detector first to see summary & downloads.")
        # the two extra download buttons stay where they are ↓
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
#         st.download_button("⬇️ Download Aligned Data",
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
        st.download_button("⬇️ Download ZIP", buf.getvalue(),
                           "PeakValleyResults.zip", "application/zip")


if st.session_state.results:
    st.markdown("---")
    align_col, dl_col = st.columns([2, 1])
    with align_col:
        do_align = st.button("🔧 Align landmarks & normalize counts",
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

            # —— 3. Compute finite x‐range, pad, and grid *before* any fallback to [0,1]
            all_xmin = float(np.nanmin([np.nanmin(w) for w in warped]))
            all_xmax = float(np.nanmax([np.nanmax(w) for w in warped]))

            # if non‐finite or degenerate, give a tiny wiggle instead of full [0,1]
            if not np.isfinite(all_xmin) or not np.isfinite(all_xmax):
                all_xmin, all_xmax = 0.0, 1.0
            elif all_xmax == all_xmin:
                d = abs(all_xmin) if all_xmin != 0 else 1.0
                all_xmin -= 0.05 * d
                all_xmax += 0.05 * d

            # build the grid *after* fixing finite span
            span = all_xmax - all_xmin
            pad  = 0.05 * span
            x_grid = np.linspace(all_xmin - pad, all_xmax + pad, 4000)
            std_fallback = 0.05 * (span or 1.0)

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
                bw   = st.session_state.params.get(stem, {}).get("bw", "scott")
                try:
                    bw_val = float(bw)
                except Exception:
                    bw_val = bw

                if np.unique(good).size >= 2:
                    try:
                        kdes[stem] = gaussian_kde(good, bw_method="scott")(x_grid)
                    except Exception:  # numerical failure
                        kdes[stem] = np.exp(-(x_grid - good.mean())**2 / (2*std_fallback**2))
                else:                  # single value or empty
                    center = good[0] if good.size else 0.5*(all_xmin + all_xmax)
                    kdes[stem] = np.exp(-(x_grid - center)**2 / (2*std_fallback**2))

            all_ymax = max(ys.max() for ys in kdes.values())
            if not np.isfinite(all_ymax):
                # choose a sensible default—e.g. 1.0 or the next-largest finite value
                all_ymax = 1.0

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
                    (all_xmin - pad, all_xmax + pad),
                    all_ymax      # now guaranteed finite
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

            st.success("✓ Landmarks aligned – scroll down for the stacked view or download the ZIP!")
            st.rerun()
