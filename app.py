# app.py  – Streamlit entry-point
from __future__ import annotations
import io, zipfile, tempfile
from pathlib import Path
import re                                         # only for log messages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

from peak_valley import (
    arcsinh_transform, read_counts,
    kde_peaks_valleys, quick_peak_estimate,
    fig_to_png,
    ask_gpt_peak_count, ask_gpt_prominence,
)

# ───────────────────────────────────────── UI set-up ─────────────────────────────────────────
st.set_page_config("Peak & Valley Detector", "🔬", layout="wide")
st.title("🔬 Peak & Valley Detector — CSV *or* full dataset")

# persistent slots ───────────────────────────────────────────────────────────────────────────
for key, default in {
    "results": {}, "fig_pngs": {}, "generated_csvs": [],
    "expr_df": None, "meta_df": None,
    "expr_name": None, "meta_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ════════════════════════  SIDEBAR  ════════════════════════════════════════════════════════
with st.sidebar:
    mode = st.radio("Choose mode", ["Counts CSV files", "Whole dataset"])

    # ───────────────────── 1) Counts-CSV workflow ─────────────────────
    if mode == "Counts CSV files":
        counts_files = st.file_uploader(
            "Upload *_raw_counts.csv*", type=["csv"], accept_multiple_files=True
        )

        # already-generated CSVs from previous dataset uploads
        gen_files: list[io.BytesIO] = []
        if st.session_state.generated_csvs:
            st.markdown("**Generated CSVs (cached)**")
            stems  = [s for s, _ in st.session_state.generated_csvs]
            picked = st.multiselect("Choose cached files", stems, stems)
            for stem, bio in st.session_state.generated_csvs:
                if stem in picked:
                    bio.seek(0)
                    bio.name = f"{stem}_raw_counts.csv"
                    gen_files.append(bio)

        st.markdown("---  \n**CSV layout**")
        header_row = st.number_input("Header row  (-1 = none)", 0, step=1)
        skip_rows  = st.number_input("Rows to skip",              0, step=1)

    # ───────────────────── 2) Whole-dataset workflow ──────────────────
    else:
        # uploaders always visible – users may replace the dataset any time
        expr_file = st.file_uploader("expression_matrix_combined.csv", type=["csv"])
        meta_file = st.file_uploader("cell_metadata_combined.csv",    type=["csv"])

        # clear-cache button
        if st.session_state.expr_df is not None:
            if st.button("🗑 Clear loaded dataset"):
                for k in ("expr_df", "meta_df", "expr_name", "meta_name"):
                    st.session_state[k] = None
                st.experimental_rerun()

        # (re)load if BOTH uploaders have files and nothing cached OR names changed
        if expr_file and meta_file:
            reload_needed = (
                st.session_state.expr_df is None or
                expr_file.name != st.session_state.expr_name or
                meta_file.name != st.session_state.meta_name
            )
            if reload_needed:
                with st.spinner("⌛ Parsing expression & metadata …"):
                    st.session_state.expr_df  = pd.read_csv(expr_file, low_memory=False)
                    st.session_state.meta_df  = pd.read_csv(meta_file,  low_memory=False)
                    st.session_state.expr_name = expr_file.name
                    st.session_state.meta_name = meta_file.name

        expr_df, meta_df = st.session_state.expr_df, st.session_state.meta_df
        if expr_df is not None and meta_df is not None:
            markers = [c for c in expr_df.columns if c != "cell_id"]
            samples = meta_df["sample"].unique().tolist()

            # simple select-all toggles avoid 400-item widgets
            all_markers = st.checkbox("All markers",  False, key="chk_m")
            all_samples = st.checkbox("All samples",  False, key="chk_s")

            sel_markers = markers if all_markers else st.multiselect("Marker(s)",  markers)
            sel_samples = samples if all_samples else st.multiselect("Sample(s)",  samples)

            if sel_markers and sel_samples:
                if st.button("Generate counts CSVs"):
                    total    = len(sel_markers) * len(sel_samples)
                    bar      = st.progress(0.0, "Generating …")
                    existing = {s for s, _ in st.session_state.generated_csvs}

                    for i, m in enumerate(sel_markers, 1):
                        for j, s in enumerate(sel_samples, 1):
                            idx  = (i-1)*len(sel_samples)+j
                            stem = f"{s}_{m}"
                            if stem in existing:
                                bar.progress(idx/total, f"Skip {stem} (exists)")
                                continue

                            counts = arcsinh_transform(
                                expr_df.loc[meta_df["sample"].eq(s), m]
                            )
                            bio = io.BytesIO()
                            counts.to_csv(bio, index=False, header=False)
                            bio.seek(0); bio.name = f"{stem}_raw_counts.csv"
                            st.session_state.generated_csvs.append((stem, bio))
                            bar.progress(idx/total, f"Added {stem} ({idx}/{total})")

                    bar.empty()
                    st.success("✓ CSVs cached – switch to **Counts CSV files** to run detector.")

        header_row, skip_rows = -1, 0      # generated CSVs are header-less

    # ───────────────────── Detection parameters ───────────────────────
    st.markdown("---  \n### Detection")
    auto    = st.selectbox("Number of peaks", ["Automatic", 1, 2, 3, 4, 5, 6])
    n_fixed = None if auto == "Automatic" else int(auto)
    cap_min = n_fixed if n_fixed else 1
    max_peaks = st.number_input("Maximum peaks (Automatic cap)",
                                cap_min, 6, max(2, cap_min), step=1,
                                disabled=(n_fixed is not None))

    bw_opt = st.selectbox("Bandwidth", ["scott", "silverman", "0.5", "0.8", "1.0"])
    bw     = float(bw_opt) if bw_opt.replace(".", "", 1).isdigit() else bw_opt

    # ───── PROMINENCE: Manual vs GPT-automatic ────────────────────────
    prom_mode = st.radio("Prominence mode", ["Manual", "Automatic (GPT)"])
    if prom_mode == "Manual":
        promin = st.slider("Prominence", 0.01, 0.30, 0.05, 0.01)
    else:
        promin = None                                 # decide later

    min_w    = st.slider("Min peak width", 0, 6, 0, 1)
    grid_sz  = st.slider("Max KDE grid", 4_000, 40_000, 20_000, 1_000)
    val_drop = st.slider("Valley drop (% of peak)", 1, 50, 10, 1)

    st.markdown("---  \n### GPT helper")
    pick = st.selectbox("Model", ["o4-mini", "gpt-4o-mini",
                                  "gpt-4-turbo-preview", "Custom"])
    gpt_model = st.text_input("Custom model") if pick == "Custom" else pick
    api_key   = st.text_input("OpenAI API key", type="password")
    dark_bg   = st.checkbox("🌙 Dark plots")

# ═════════════════════ main buttons ═══════════════════════════════════
run       = st.button("🚀 Run detector")
clear_all = st.button("🗑️ Clear results")

if clear_all:
    st.session_state.results.clear()
    st.session_state.fig_pngs.clear()
    st.rerun()

# ═════════════════════ MAIN processing ════════════════════════════════
if run:
    if mode == "Counts CSV files":
        csv_files = list(counts_files or []) + gen_files
        if not csv_files:
            st.error("No CSV files selected."); st.stop()
    else:
        st.error("Generate CSVs first then switch to *Counts CSV files*."); st.stop()

    # GPT needed if either Automatic peaks **or** Automatic prominence is chosen
    gpt_needed = ((auto == "Automatic") or (prom_mode == "Automatic (GPT)"))
    if gpt_needed and not api_key:
        st.error("Automatic peak-count/prominence requires an OpenAI key.")
        st.stop()

    client   = OpenAI(api_key=api_key) if api_key else None
    progress = st.progress(0.0, "Processing …")

    new_files = [f for f in csv_files if Path(f.name).stem not in st.session_state.results]

    for idx, file in enumerate(new_files, 1):
        stem   = Path(file.name).stem
        counts = read_counts(file, header_row, skip_rows)

        # ───────── decide prominence ────────────────────────────────
        prom_used = promin
        if prom_used is None:                             # GPT-automatic
            prom_used = ask_gpt_prominence(
                client, gpt_model, counts_full=counts
            )

        # ───────── decide number of peaks ───────────────────────────
        if n_fixed is None:                               # Automatic
            n_est, confident = quick_peak_estimate(
                counts, prom_used, bw, min_w or None, grid_sz
            )
            n_use = n_est if confident else None
            if n_use is not None:
                n_use = min(n_use, max_peaks)
        else:
            n_use = n_fixed

        if n_use is None and n_fixed is None:
            n_use = ask_gpt_peak_count(
                client, gpt_model, max_peaks, counts_full=counts
            )
        if n_use is None:                                 # fallback → cap
            n_use = max_peaks

        # ───────── detector ────────────────────────────────────────
        peaks, valleys, xs, ys = kde_peaks_valleys(
            counts,
            n_use,
            prom_used,
            bw,
            min_w or None,
            grid_sz,
            drop_frac=val_drop / 100.0,
        )

        # single-peak tail valley heuristic
        if len(peaks) == 1 and not valleys:
            idx_peak = np.searchsorted(xs, peaks[0])
            y_peak   = ys[idx_peak]
            below    = np.where(ys[idx_peak:] < (val_drop/100)*y_peak)[0]
            if below.size:
                valleys = [float(xs[idx_peak + below[0]])]

        # ───────── plot ────────────────────────────────────────────
        pad = 0.05 * (xs.max() - xs.min())
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
        colL, colF = ("#4FC3F7", "#4FC3F766") if dark_bg else ("skyblue", "#87CEEB88")
        if dark_bg:
            fig.patch.set_facecolor("#222"); ax.set_facecolor("#222")
            ax.tick_params(colors="white"); ax.spines[:].set_color("white")
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")

        ax.plot(xs, ys, color=colL)
        ax.fill_between(xs, 0, ys, color=colF)
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        for p in peaks:
            ax.axvline(p, color="red", ls="--", lw=1)
        for v in valleys:
            ax.axvline(v, color="green", ls=":", lw=1)
        ax.set_xlabel("Arcsinh counts"); ax.set_ylabel("Density")
        ax.set_title(stem, fontsize=9); fig.tight_layout()

        st.session_state.fig_pngs[f"{stem}.png"] = fig_to_png(fig); plt.close(fig)
        st.session_state.results[stem] = {"peaks": peaks, "valleys": valleys}

        progress.progress(idx / len(new_files), f"Done {stem}")

    st.success("All files processed!")


# ═══════════════ History / controls (unchanged) ═══════════════════════
st.header("📊 Processed datasets")
if st.session_state.results:
    for stem, info in list(st.session_state.results.items()):
        colL, colR = st.columns([10, 1])
        with colL.expander(stem, False):
            st.image(st.session_state.fig_pngs[f"{stem}.png"], use_column_width=True)
            st.write(f"**Peaks:** {info['peaks']}")
            st.write(f"**Valleys:** {info['valleys']}")
        if colR.button("❌", key=f"del_{stem}", help="Remove result"):
            st.session_state.results.pop(stem)
            st.session_state.fig_pngs.pop(f"{stem}.png", None)
            st.rerun()
else:
    st.info("No results yet.")


# ═══════════════ Summary + ZIP (unchanged) ════════════════════════════
if st.session_state.results:
    df = pd.DataFrame([{"file": k, **v} for k, v in st.session_state.results.items()])
    st.subheader("📋 Summary")
    st.dataframe(df, use_container_width=True)

    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("summary.csv", df.to_csv(index=False).encode())
            for fn, png in st.session_state.fig_pngs.items():
                z.writestr(fn, png)
        st.download_button("⬇️ Download ZIP", buf.getvalue(),
                           "PeakValleyResults.zip", mime="application/zip")
