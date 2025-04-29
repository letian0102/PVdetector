# peak_valley_web_app.py
# ======================================================================
#  pip install streamlit openai numpy pandas matplotlib pillow scipy
#  streamlit run peak_valley_web_app.py
# ======================================================================

import io, zipfile, base64, tempfile, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

import streamlit as st
from openai import OpenAI, OpenAIError


# ──────────────────────────────────────────────────────────────
#  simple helper: arcsinh-transform exactly like your notebook
# ──────────────────────────────────────────────────────────────
def arcsinh_transform(x, a=1.0, b=1 / 5, c=0.0):
    return (1 / b) * np.arcsinh(a * x + c)


# ──────────────────────────────────────────────────────────────
#  read a “*_raw_counts.csv” (just one numeric column)
#  honour header row + extra skip rows chosen by the user
# ──────────────────────────────────────────────────────────────
def read_counts(file, header_row: int, skip_rows: int) -> np.ndarray:
    file.seek(0)
    hdr = None if header_row < 0 else header_row
    try:
        return (
            pd.read_csv(
                file,
                header=hdr,
                skiprows=skip_rows or None,
                usecols=[0],
                dtype=np.float64,
                engine="c",
                memory_map=True,
            )
            .squeeze("columns")
            .values
        )
    except ValueError:  # very large file → stream
        file.seek(0)
        chunks = pd.read_csv(
            file,
            header=hdr,
            skiprows=skip_rows or None,
            usecols=[0],
            dtype=np.float64,
            engine="c",
            chunksize=200_000,
        )
        return np.concatenate([c.values.ravel() for c in chunks])


# ──────────────────────────────────────────────────────────────
#  KDE → peaks & valleys  (unchanged)
# ──────────────────────────────────────────────────────────────
def kde_peaks_valleys(
    data,
    n_peaks=None,
    prominence=0.05,
    bw="scott",
    min_width=None,
    grid_size=20_000,
):
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        return [], [], np.array([]), np.array([])

    if x.size > 10_000:
        x = np.random.choice(x, 10_000, replace=False)

    kde = gaussian_kde(x, bw_method=bw)
    h = kde.factor * x.std(ddof=1)
    xs = np.linspace(x.min() - h, x.max() + h, min(grid_size, max(4_000, 4 * x.size)))
    ys = kde(xs)

    kw = {"prominence": prominence}
    if min_width:
        kw["width"] = min_width
    locs, _ = find_peaks(ys, **kw)

    if n_peaks is not None and locs.size < n_peaks:           # relax prominence
        prom = prominence
        for _ in range(4):
            prom /= 2
            locs, _ = find_peaks(
                ys, prominence=prom, **({"width": min_width} if min_width else {})
            )
            if locs.size >= n_peaks:
                break

    if locs.size == 0:
        return [], [], xs, ys

    if n_peaks is None or n_peaks >= locs.size:
        peaks_idx = np.sort(locs)
    else:
        tallest = np.argsort(ys[locs])[-n_peaks:]
        peaks_idx = np.sort(locs[tallest])

    peaks_x = np.round(xs[peaks_idx], 10).tolist()
    valleys_x = []
    if len(peaks_idx) > 1:
        for L, R in zip(peaks_idx[:-1], peaks_idx[1:]):
            seg = np.arange(L + 1, R)
            mins, _ = find_peaks(-ys[seg])
            idx = seg[mins[np.argmin(ys[seg][mins])]] if mins.size else seg[np.argmin(ys[seg])]
            valleys_x.append(np.round(xs[idx], 10))
    return peaks_x, valleys_x, xs, ys


# ──────────────────────────────────────────────────────────────
#  tiny helpers
# ──────────────────────────────────────────────────────────────
def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0.15)
    buf.seek(0)
    return buf.getvalue()


def thumb64(png: bytes) -> str:
    im = Image.open(io.BytesIO(png)).resize((150, 100))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ╔═══════════════════════════════════════════════════════════╗
# ║                    STREAMLIT  UI                          ║
# ╚═══════════════════════════════════════════════════════════╝
st.set_page_config("Peak & Valley Detector", page_icon="🔬", layout="wide")
st.title("🔬 Peak & Valley Detector – CSV *or* full dataset")

# persistent objects
for key, default in {
    "results": {},
    "fig_pngs": {},
    "generated_csvs": [],
    "tmpdir": tempfile.mkdtemp(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ──────────────────────────────────────────────────────────────
# sidebar: mode switch + upload widgets
# ──────────────────────────────────────────────────────────────
with st.sidebar:

    mode = st.radio("Choose mode", ["Counts CSV files", "Whole dataset"])

    # ---------- COUNTS-CSV MODE ----------
    if mode == "Counts CSV files":
        counts_files = st.file_uploader(
            "Upload *_raw_counts.csv*", type=["csv"], accept_multiple_files=True
        )

        # show previously generated CSVs from dataset mode
        gen_files = []
        if st.session_state.generated_csvs:
            st.markdown("**Generated CSVs (from dataset)**")
            default = [stem for stem, _ in st.session_state.generated_csvs]
            picked  = st.multiselect(
                "Choose generated files",
                options=default,
                default=default,
            )
            for stem, bio in st.session_state.generated_csvs:
                if stem in picked:
                    bio.seek(0); bio.name = f"{stem}_raw_counts.csv"
                    gen_files.append(bio)

        # ↓ NEW: header / skip-rows settings for counts CSVs
        st.markdown("---  \n**CSV layout**")
        header_row = st.number_input("Header row  (-1 = none)", value=0, step=1)
        skip_rows  = st.number_input("Extra rows to skip", value=0, min_value=0, step=1)

    # ---------- DATASET MODE -------------
    else:
        expr_file = st.file_uploader("expression_matrix_combined.csv", type=["csv"])
        meta_file = st.file_uploader("cell_metadata_combined.csv",    type=["csv"])

        if expr_file and meta_file:
            expr_df = pd.read_csv(expr_file)
            meta_df = pd.read_csv(meta_file)
            markers = [c for c in expr_df.columns if c != "cell_id"]
            samples = meta_df["sample"].unique().tolist()

            sel_markers = st.multiselect("Marker(s)",  markers, markers[:1])
            sel_samples = st.multiselect("Sample(s)",  samples, samples[:1])

            if st.button("Generate counts CSVs"):
                st.session_state.generated_csvs.clear()
                for m in sel_markers:
                    for s in sel_samples:
                        mask   = meta_df["sample"] == s
                        counts = arcsinh_transform(expr_df.loc[mask, m])
                        stem   = f"{s}_{m}"
                        bio    = io.BytesIO()
                        counts.to_csv(bio, index=False, header=False)
                        bio.seek(0); bio.name = f"{stem}_raw_counts.csv"
                        st.session_state.generated_csvs.append((stem, bio))
                st.success("Generated!  Switch to **Counts CSV files** mode.")

        # dataset-generated CSVs have no header & no extra skip rows
        header_row = -1
        skip_rows  = 0

    # -------- common detection parameters ----------
    st.markdown("---  \n### Detection")
    auto = st.selectbox("Number of peaks", ["Automatic", 1, 2, 3, 4, 5, 6])
    n_fixed = None if auto == "Automatic" else int(auto)

    cap_min = n_fixed if n_fixed is not None else 1
    max_peaks = st.number_input(
        "Maximum peaks (cap for Automatic)",
        min_value=cap_min,
        max_value=6,
        value=max(2, cap_min),
        step=1,
        disabled=(n_fixed is not None),
    )

    bw_opt = st.selectbox("Bandwidth", ["scott", "silverman", "0.5", "0.8", "1.0"], 0)
    bw = float(bw_opt) if bw_opt.replace(".", "", 1).isdigit() else bw_opt

    promin    = st.slider("Prominence", 0.0, 1.0, 0.05, 0.01)
    min_width = st.slider("Min peak width", 0, 6, 0, 1)
    grid_sz   = st.slider("Max KDE grid", 4_000, 40_000, 20_000, 1_000)

    # -------- optional GPT helper ----------
    st.markdown("---  \n### GPT helper (Automatic peak-count)")
    model_pick = st.selectbox(
        "Model",
        ["o4-mini", "gpt-4o-mini", "gpt-4-turbo-preview", "Custom"],
        0,
    )
    gpt_model = st.text_input("Custom model name") if model_pick == "Custom" else model_pick
    api_key   = st.text_input("OpenAI API key", type="password")

    dark_bg = st.checkbox("🌙 Dark plots", False)

# ──────────────────────────────────────────────────────────────
# buttons
# ──────────────────────────────────────────────────────────────
run       = st.button("🚀 Run detector")
clear_all = st.button("🗑️ Clear results")

if clear_all:
    st.session_state.results.clear()
    st.session_state.fig_pngs.clear()
    st.rerun()

# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
if run:
    if mode == "Counts CSV files":
        csv_files = list(counts_files or []) + gen_files
        if not csv_files:
            st.error("No CSV selected."); st.stop()
    else:
        st.error("Generate counts first, then switch to *Counts CSV files*."); st.stop()

    if auto == "Automatic" and not api_key:
        st.error("Automatic mode needs an OpenAI key."); st.stop()

    client   = OpenAI(api_key=api_key) if api_key else None
    progress = st.progress(0.0, "Processing…")

    new_files = [f for f in csv_files if Path(f.name).stem not in st.session_state.results]

    for idx, file in enumerate(new_files, 1):
        stem   = Path(file.name).stem
        counts = read_counts(file, header_row, skip_rows)

        # automatic peak-count via GPT
        if n_fixed is None:
            prompt = ("How many distinct density peaks (modes) does this list show? "
                      "Answer with a single integer.\n\n"
                      f"{counts[:200].tolist()}")
            try:
                reply = client.chat.completions.create(
                    model=gpt_model, seed=2025, timeout=60,
                    messages=[{"role":"user","content":prompt}],
                )
                guess = int(re.findall(r"\d+", reply.choices[0].message.content)[0])
                n_use = min(max_peaks, guess) if guess > 0 else None
            except Exception as e:
                st.error(f"GPT peak-count failed for {stem}: {e}")
                n_use = None
        else:
            n_use = n_fixed

        peaks, valleys, xs, ys = kde_peaks_valleys(
            counts, n_use, promin, bw, min_width or None, grid_sz
        )

        # fallback right-tail valley if single peak
        if n_fixed is None and len(peaks) == 1 and not valleys:
            try:
                valleys = [float(xs[np.argmin(ys[xs > peaks[0]])])]
            except Exception:
                valleys = []

        # plot
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
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        for p in peaks: ax.axvline(p, color="red",   ls="--", lw=1)
        for v in valleys: ax.axvline(v, color="green", ls=":", lw=1)
        ax.set_title(stem, fontsize=9); ax.set_xlabel("Arcsinh counts"); ax.set_ylabel("Density")
        fig.tight_layout()

        st.session_state.fig_pngs[f"{stem}.png"] = fig_to_png(fig); plt.close(fig)
        st.session_state.results[stem] = {"peaks": peaks, "valleys": valleys}

        progress.progress(idx / len(new_files), f"Done {stem}")

    st.success("All files processed!")

# ──────────────────────────────────────────────────────────────
# history / summary / download  (unchanged)
# ──────────────────────────────────────────────────────────────
st.header("📊 Processed datasets")
if st.session_state.results:
    for stem, info in list(st.session_state.results.items()):
        c1, c2 = st.columns([10, 1])
        with c1.expander(stem, False):
            st.image(st.session_state.fig_pngs[f"{stem}.png"], use_column_width=True)
            st.write(f"**Peaks:** {info['peaks']}")
            st.write(f"**Valleys:** {info['valleys']}")
        if c2.button("❌", key=f"del_{stem}", help="Remove result"):
            st.session_state.results.pop(stem); st.session_state.fig_pngs.pop(f"{stem}.png", None)
            st.rerun()
else:
    st.info("No results yet.")

if st.session_state.results:
    df = pd.DataFrame([{"file": k, **v} for k, v in st.session_state.results.items()])
    st.subheader("📋 Summary"); st.dataframe(df, use_container_width=True)
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("summary.csv", df.to_csv(index=False).encode())
            for fn, png in st.session_state.fig_pngs.items():
                z.writestr(fn, png)
        st.download_button("⬇️ Download ZIP", buf.getvalue(),
                           "PeakValleyResults.zip", mime="application/zip")
