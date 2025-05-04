"""
Streamlit entry-point.
All heavy logic lives in the peak_valley package.
"""
from __future__ import annotations
import io, zipfile, tempfile, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from openai import OpenAI

from peak_valley import (
    arcsinh_transform, read_counts,
    kde_peaks_valleys, quick_peak_estimate,
    fig_to_png, thumb64, ask_gpt_peak_count,
)

# ──────────────────────────────────────────────────────────────
# UI boiler-plate (identical to your latest, shortened comments)
# ──────────────────────────────────────────────────────────────
st.set_page_config("Peak & Valley Detector", "🔬", layout="wide")
st.title("🔬 Peak & Valley Detector — CSV *or* full dataset")

# session state slots
for key, default in {
    "results": {}, "fig_pngs": {}, "generated_csvs": [], "tmpdir": tempfile.mkdtemp()
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------- sidebar ------------------------------------------------
with st.sidebar:
    mode = st.radio("Choose mode", ["Counts CSV files", "Whole dataset"])

    # —­­ counts CSV mode —
    if mode == "Counts CSV files":
        counts_files = st.file_uploader(
            "Upload *_raw_counts.csv*", type=["csv"], accept_multiple_files=True
        )
        # add generated CSVs
        gen_files = []
        if st.session_state.generated_csvs:
            st.markdown("**Generated CSVs (from dataset)**")
            stems = [s for s, _ in st.session_state.generated_csvs]
            picked = st.multiselect("Choose generated", stems, stems)
            for stem, bio in st.session_state.generated_csvs:
                if stem in picked:
                    bio.seek(0); bio.name = f"{stem}_raw_counts.csv"
                    gen_files.append(bio)

        st.markdown("---  \n**CSV layout**")
        header_row = st.number_input("Header row (-1 = none)", 0, step=1)
        skip_rows  = st.number_input("Rows to skip", 0, step=1)

    # —­­ full-dataset mode —
    else:
        expr_file = st.file_uploader("expression_matrix_combined.csv", type=["csv"])
        meta_file = st.file_uploader("cell_metadata_combined.csv",    type=["csv"])

        if st.session_state.get("UploadedFile", None):
            st.stop()

        if expr_file and meta_file:
            # show a spinner while pandas parses potentially huge files
            with st.spinner("⌛ Loading expression & metadata …"):
                expr_df = pd.read_csv(expr_file, low_memory=False)
                meta_df = pd.read_csv(meta_file,  low_memory=False)

            markers = [c for c in expr_df.columns if c != "cell_id"]
            samples = meta_df["sample"].unique().tolist()

            # quick “select all” toggles (avoids rendering 400-item boxes)
            all_markers = st.checkbox("All markers",  False)
            all_samples = st.checkbox("All samples",  False)

            sel_markers = markers if all_markers else \
                          st.multiselect("Marker(s)",  markers)
            sel_samples = samples if all_samples else \
                          st.multiselect("Sample(s)",  samples)

            # guard against empty pick
            if not sel_markers or not sel_samples:
                st.info("Select at least one marker & sample.")
            else:
                if st.button("Generate counts CSVs"):
                    total   = len(sel_markers) * len(sel_samples)
                    progbar = st.progress(0.0, "Generating …")
                    st.session_state.generated_csvs.clear()

                    for i, m in enumerate(sel_markers, 1):
                        for j, s in enumerate(sel_samples, 1):
                            idx     = (i-1)*len(sel_samples) + j
                            mask    = meta_df["sample"].eq(s)
                            counts  = arcsinh_transform(expr_df.loc[mask, m])
                            stem    = f"{s}_{m}"
                            bio     = io.BytesIO()
                            counts.to_csv(bio, index=False, header=False)
                            bio.seek(0); bio.name = f"{stem}_raw_counts.csv"
                            st.session_state.generated_csvs.append((stem, bio))
                            progbar.progress(idx/total,
                                             f"Generated {stem} ({idx}/{total})")
                    progbar.empty()
                    st.success("✓  In-memory CSVs created! "
                               "Switch to **Counts CSV files** mode.")


        header_row, skip_rows = -1, 0

    # shared detector widgets
    st.markdown("---  \n### Detection")
    auto      = st.selectbox("Number of peaks", ["Automatic", 1, 2, 3, 4, 5, 6])
    n_fixed   = None if auto == "Automatic" else int(auto)
    cap_min   = n_fixed if n_fixed is not None else 1
    max_peaks = st.number_input("Maximum peaks (Automatic cap)",
                                cap_min, 6, max(2, cap_min), step=1,
                                disabled=(n_fixed is not None))

    bw_opt   = st.selectbox("Bandwidth", ["scott", "silverman", "0.5", "0.8", "1.0"])
    bw       = float(bw_opt) if bw_opt.replace(".", "", 1).isdigit() else bw_opt

    promin   = st.slider("Prominence", 0.0, 1.0, 0.05, 0.01)
    min_w    = st.slider("Min peak width", 0, 6, 0, 1)
    grid_sz  = st.slider("Max KDE grid", 4_000, 40_000, 20_000, 1_000)

    st.markdown("---  \n### GPT helper (Automatic peak-count)")
    model_pick = st.selectbox("Model",
                              ["o4-mini", "gpt-4o-mini",
                               "gpt-4-turbo-preview", "Custom"])
    gpt_model  = st.text_input("Custom model") if model_pick == "Custom" else model_pick
    api_key    = st.text_input("OpenAI API key", type="password")

    dark_bg = st.checkbox("🌙 Dark plots")

# main buttons
run       = st.button("🚀 Run detector")
clear_all = st.button("🗑️ Clear results")

if clear_all:
    st.session_state.results.clear(); st.session_state.fig_pngs.clear(); st.rerun()

# ═════════════════════════════════════════════════════════════
# MAIN processing
# ═════════════════════════════════════════════════════════════
if run:
    if mode == "Counts CSV files":
        csv_files = list(counts_files or []) + gen_files
        if not csv_files:
            st.error("No CSV files selected."); st.stop()
    else:
        st.error("Generate CSVs first then switch to *Counts CSV files*."); st.stop()

    if auto == "Automatic" and not api_key:
        st.error("Automatic mode needs an OpenAI key."); st.stop()

    client   = OpenAI(api_key=api_key) if api_key else None
    progress = st.progress(0.0, "Processing…")

    new_files = [f for f in csv_files
                 if Path(f.name).stem not in st.session_state.results]

    for idx, file in enumerate(new_files, 1):
        stem   = Path(file.name).stem
        counts = read_counts(file, header_row, skip_rows)

        # 1. quick local peak estimate
        if n_fixed is None:
            guess, confident = quick_peak_estimate(
                counts, promin, bw, min_w or None, grid_sz
            )
            n_use = guess if confident and guess <= max_peaks else None
        else:
            n_use = n_fixed

        # 2. GPT fallback when not confident
        if n_use is None and n_fixed is None:
            n_use = ask_gpt_peak_count(
                client, gpt_model, counts[:200].tolist(),
                max_peaks, counts_full=counts      # ← add this kw-arg
            )

        # 3. run detector
        peaks, valleys, xs, ys = kde_peaks_valleys(
            counts, n_use, promin, bw, min_w or None, grid_sz
        )

        # ──────── NEW: ask GPT for the valley if exactly **one** peak ────────
        if len(peaks) == 1 and api_key:
            # tiny thumbnail to keep prompt short
            fig_t, ax_t = plt.subplots(figsize=(3, 1.5), dpi=80)
            ax_t.plot(xs, ys, color="black"); ax_t.axis("off")
            thumb_png = fig_to_png(fig_t); plt.close(fig_t)

            valley_prompt = (
                f"The attached thumbnail shows ONE density peak at x ≈ {peaks[0]:.3f}. "
                "Return the x-coordinate of the valley at the right tail where the density distribution is at the end of decreasing, valley should not be somewhere far from the main distribution."
                "Respond with one number only."
            )
            try:
                v_reply = client.chat.completions.create(
                    model=gpt_model, seed=2025, timeout=60,
                    messages=[
                        {"role": "system",
                        "content": "You detect the right-tail valley after a single peak."},
                        {"role": "user",
                        "content": [
                            {"type": "text", "text": valley_prompt},
                            {"type": "image_url",
                            "image_url":
                                {"url": f"data:image/png;base64,{thumb64(thumb_png)}"}}
                        ]},
                    ],
                )
                valley_val = float(
                    re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                            v_reply.choices[0].message.content)[0]
                )
                valleys = [valley_val]
            except Exception as e:
                st.warning(f"GPT valley failed for {stem}: {e}")
                # simple heuristic fall-back
                if not valleys:
                    try:
                        valleys = [float(xs[np.argmin(ys[xs > peaks[0]])])]
                    except Exception:
                        valleys = []
        # ---------------------------------------------------------------------

        # ----- plot (unchanged) --------------------------------------
        pad = 0.05 * (xs.max() - xs.min())
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
        if dark_bg:
            fig.patch.set_facecolor("#222"); ax.set_facecolor("#222")
            ax.tick_params(colors="white"); ax.spines[:].set_color("white")
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
            cL, cF = "#4FC3F7", "#4FC3F766"
        else:
            cL, cF = "skyblue", "#87CEEB88"

        ax.plot(xs, ys, color=cL); ax.fill_between(xs, 0, ys, color=cF)
        ax.set_xlim(xs.min()-pad, xs.max()+pad)
        for p in peaks:   ax.axvline(p, color="red",   ls="--", lw=1)
        for v in valleys: ax.axvline(v, color="green", ls=":",  lw=1)
        ax.set_xlabel("Arcsinh counts"); ax.set_ylabel("Density")
        ax.set_title(stem, fontsize=9); fig.tight_layout()

        st.session_state.fig_pngs[f"{stem}.png"] = fig_to_png(fig); plt.close(fig)
        st.session_state.results[stem] = {"peaks": peaks, "valleys": valleys}

        progress.progress(idx / len(new_files), f"Done {stem}")

    st.success("All files processed!")

# ══════════  History + controls  ═══════════════════════════════
st.header("📊 Processed datasets")

if st.session_state.results:
    for stem, info in list(st.session_state.results.items()):
        colL, colR = st.columns([10,1])
        with colL.expander(stem, False):
            st.image(st.session_state.fig_pngs[f"{stem}.png"],
                     use_column_width=True)
            st.write(f"**Peaks :** {info['peaks']}")
            st.write(f"**Valleys:** {info['valleys']}")
        if colR.button("❌", key=f"del_{stem}", help="Remove result"):
            st.session_state.results.pop(stem); st.session_state.fig_pngs.pop(
                f"{stem}.png", None); st.rerun()
else:
    st.info("No results yet.")


# ══════════  Summary + ZIP  ═══════════════════════════════════
if st.session_state.results:
    df = pd.DataFrame([{"file":k, **v} for k,v in st.session_state.results.items()])
    st.subheader("📋 Summary")
    st.dataframe(df, use_container_width=True)

    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("summary.csv", df.to_csv(index=False).encode())
            for fn, png in st.session_state.fig_pngs.items():
                z.writestr(fn, png)
        st.download_button("⬇️ Download ZIP", buf.getvalue(),
                           "PeakValleyResults.zip", mime="application/zip")
