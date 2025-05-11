# app.py  – GPT-assisted bandwidth option + persistent uploads
from __future__ import annotations
import io, zipfile, tempfile, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
from openai import OpenAI

from peak_valley import (                        # ───── internal helpers
    arcsinh_transform, read_counts,
    kde_peaks_valleys, quick_peak_estimate,
    fig_to_png
)

from peak_valley.gpt_adapter import (
    ask_gpt_peak_count,
    ask_gpt_prominence,
    ask_gpt_bandwidth,
)

# ──────────────────────────────────────────  UI boiler-plate
st.set_page_config("Peak & Valley Detector", "🔬", layout="wide")
st.title("🔬 Peak & Valley Detector — CSV *or* full dataset")

# ──────────────────────────────────────────  session-state slots
for key, default in {
    "results"        : {},
    "fig_pngs"       : {},
    "generated_csvs" : [],
    "cached_uploads" : [],          # persistent user-uploaded *_raw_counts.csv*
    "expr_df"        : None,
    "meta_df"        : None,
    "expr_name"      : None,
    "meta_name"      : None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═════════════════════════════  SIDEBAR  ═════════════════════════════
with st.sidebar:
    mode = st.radio("Choose mode", ["Counts CSV files", "Whole dataset"])

    # ──────────────────── 1)  Counts-CSV workflow ────────────────────
    if mode == "Counts CSV files":
        uploaded_now = st.file_uploader(
            "Upload *_raw_counts.csv*",
            type=["csv"], accept_multiple_files=True, key="csv_up"
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
            pick  = st.multiselect("Choose uploaded files", stems, stems, key="pick_up2")
            for f in st.session_state.cached_uploads:
                if Path(f.name).stem in pick:
                    use_uploads.append(f)
            if st.button("🗑 Clear cached uploads"):
                st.session_state.cached_uploads.clear(); st.rerun()

        use_generated: list[io.BytesIO] = []
        if st.session_state.generated_csvs:
            st.markdown("**Generated CSVs (from dataset)**")
            stems_g = [s for s,_ in st.session_state.generated_csvs]
            pick_g  = st.multiselect("Choose generated files", stems_g, stems_g, key="pick_gen2")
            for stem,bio in st.session_state.generated_csvs:
                if stem in pick_g:
                    bio.seek(0); bio.name = f"{stem}_raw_counts.csv"
                    use_generated.append(bio)

        header_row = st.number_input("Header row (−1 = none)", 0, step=1, key="hdr")
        skip_rows  = st.number_input("Rows to skip", 0, step=1, key="skip")

    # ──────────────────── 2)  Whole-dataset workflow ─────────────────
    else:
        expr_file = st.file_uploader("expression_matrix_combined.csv", type=["csv"])
        meta_file = st.file_uploader("cell_metadata_combined.csv",    type=["csv"])

        if st.session_state.expr_df is not None:
            if st.button("🗑 Clear loaded dataset"):
                for k in ("expr_df","meta_df","expr_name","meta_name"):
                    st.session_state[k] = None
                st.rerun()

        if expr_file and meta_file:
            need = (st.session_state.expr_df is None or
                    expr_file.name != st.session_state.expr_name or
                    meta_file.name != st.session_state.meta_name)
            if need:
                with st.spinner("⌛ Parsing expression / metadata …"):
                    st.session_state.expr_df  = pd.read_csv(expr_file, low_memory=False)
                    st.session_state.meta_df  = pd.read_csv(meta_file, low_memory=False)
                    st.session_state.expr_name, st.session_state.meta_name = expr_file.name, meta_file.name

        expr_df, meta_df = st.session_state.expr_df, st.session_state.meta_df
        if expr_df is not None and meta_df is not None:
            markers = [c for c in expr_df.columns if c != "cell_id"]
            samples = meta_df["sample"].unique().tolist()

            all_m = st.checkbox("All markers",  False, key="chk_m")
            all_s = st.checkbox("All samples",  False, key="chk_s")
            sel_m = markers if all_m else st.multiselect("Marker(s)",  markers)
            sel_s = samples if all_s else st.multiselect("Sample(s)",  samples)

            if sel_m and sel_s and st.button("Generate counts CSVs"):
                tot = len(sel_m)*len(sel_s); bar = st.progress(0.0, "Generating …")
                exist = {s for s,_ in st.session_state.generated_csvs}
                for i,m in enumerate(sel_m,1):
                    for j,s in enumerate(sel_s,1):
                        idx=(i-1)*len(sel_s)+j; stem=f"{s}_{m}"
                        if stem in exist:
                            bar.progress(idx/tot,f"Skip {stem} (exists)"); continue
                        counts = arcsinh_transform(expr_df.loc[meta_df["sample"].eq(s), m])
                        bio=io.BytesIO(); counts.to_csv(bio,index=False,header=False)
                        bio.seek(0); bio.name=f"{stem}_raw_counts.csv"
                        st.session_state.generated_csvs.append((stem,bio))
                        bar.progress(idx/tot,f"Added {stem} ({idx}/{tot})")
                bar.empty(); st.success("✓ CSVs cached – switch to **Counts CSV files**")

        header_row, skip_rows = -1, 0
        use_uploads, use_generated = [], []

    # ─────────────────── detector parameters ───────────────────
    st.markdown("---\n### Detection")
    auto    = st.selectbox("Number of peaks", ["Automatic",1,2,3,4,5,6])
    n_fixed = None if auto=="Automatic" else int(auto)
    cap_min = n_fixed if n_fixed else 1
    max_peaks = st.number_input("Maximum peaks (Automatic cap)",
                                cap_min,6,max(2,cap_min),step=1,disabled=(n_fixed is not None))

    # ▶ Bandwidth mode: Manual or GPT
    bw_mode = st.selectbox("Bandwidth mode", ["Manual","GPT automatic"])
    if bw_mode=="Manual":
        bw_opt = st.selectbox("Rule / scale",
                              ["scott","silverman","0.5","0.8","1.0"], key="bw_sel")
        bw_val = float(bw_opt) if bw_opt.replace(".","",1).isdigit() else bw_opt
    else:
        bw_val = None                                # will call GPT later

    # Prominence
    prom_mode = st.selectbox("Prominence", ["Manual","GPT automatic"], key="prom_sel")
    if prom_mode=="Manual":
        prom_val = st.slider("Prominence value", 0.00, 0.30, 0.05, 0.01)
    else:
        prom_val = None

    min_w    = st.slider("Min peak width", 0, 6, 0, 1)
    grid_sz  = st.slider("Max KDE grid", 4_000, 40_000, 20_000, 1_000)
    val_drop = st.slider("Valley drop (% of peak)", 1, 50, 10, 1)

    st.markdown("---\n### GPT helper")
    pick = st.selectbox("Model", ["o4-mini","gpt-4o-mini","gpt-4-turbo-preview","Custom"])
    gpt_model = st.text_input("Custom model") if pick=="Custom" else pick
    api_key   = st.text_input("OpenAI API key", type="password")
    dark_bg   = st.checkbox("🌙 Dark plots")

# ═══════════ main buttons ═══════════
run, clear_all = st.columns(2)
if clear_all.button("🗑 Clear results"):
    st.session_state.results.clear(); st.session_state.fig_pngs.clear(); st.rerun()

run_clicked = run.button("🚀 Run detector")

# ═══════════ MAIN processing ═════════
if run_clicked:
    csv_files = use_uploads + use_generated
    if not csv_files:
        st.error("No CSV files selected."); st.stop()

    need_gpt = ((n_fixed is None) or prom_mode=="GPT automatic" or bw_mode=="GPT automatic")
    if need_gpt and not api_key:
        st.error("GPT-based options need an OpenAI key."); st.stop()

    client = OpenAI(api_key=api_key) if api_key else None
    progress = st.progress(0.0,"Processing…")
    todo = [f for f in csv_files if Path(f.name).stem not in st.session_state.results]

    for idx,f in enumerate(todo,1):
        stem  = Path(f.name).stem
        cnts  = read_counts(f, header_row, skip_rows)

        # --- bandwidth ------------------------------------
        bw_use: str|float
        if bw_val is not None:                      # Manual
            bw_use = bw_val
        else:                                       # GPT automatic
            # expected_peaks = user-fixed value *or* the cap they set
            expected_peaks = n_fixed if n_fixed is not None else max_peaks
            bw_use = ask_gpt_bandwidth(
                client,
                gpt_model,
                cnts,
                peak_amount=expected_peaks,         # 🔸 NEW
                default=0.8,
            )

        # --- prominence -----------------------------------
        if prom_val is not None:
            prom_use = prom_val
        else:
            prom_use = ask_gpt_prominence(client, gpt_model, cnts, default=0.05)

        # --- peak count -----------------------------------
        if n_fixed is None:
            n_est, confident = quick_peak_estimate(cnts, prom_use, bw_use, min_w or None, grid_sz)
            n_use = n_est if confident else None
            if n_use is not None:
                n_use = min(n_use, max_peaks)
        else:
            n_use = n_fixed

        if n_use is None and n_fixed is None:
            def _infer_marker(stem: str) -> str | None:
                if stem.endswith("_raw_counts"):
                    stem = stem[:-11]                           # strip suffix
                parts = stem.split("_")
                print("Got it."+str(parts[-1]))
                return parts[-1] if parts else None
            n_use = ask_gpt_peak_count(client, gpt_model, max_peaks, counts_full=cnts, marker_name=_infer_marker(stem))
        if n_use is None:
            n_use = max_peaks

        # --- run detector ---------------------------------
        peaks,valleys,xs,ys = kde_peaks_valleys(
            cnts,n_use,prom_use,bw_use,min_w or None,grid_sz,
            drop_frac=val_drop/100.0
        )

        if len(peaks)==1 and not valleys:
            p_idx = np.searchsorted(xs,peaks[0]); y_pk=ys[p_idx]
            drop  = np.where(ys[p_idx:]<(val_drop/100)*y_pk)[0]
            if drop.size: valleys=[float(xs[p_idx+drop[0]])]

        # --- plot -----------------------------------------
        pad = 0.05*(xs.max()-xs.min())
        fig,ax = plt.subplots(figsize=(5,2.5),dpi=150)
        cL,cF = ("#4FC3F7","#4FC3F766") if dark_bg else ("skyblue","#87CEEB88")
        if dark_bg:
            fig.patch.set_facecolor("#222"); ax.set_facecolor("#222")
            for spine in ax.spines.values(): spine.set_color("white")
            ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.plot(xs,ys,color=cL); ax.fill_between(xs,0,ys,color=cF)
        ax.set_xlim(xs.min()-pad,xs.max()+pad)
        for p in peaks:   ax.axvline(p,color="red",ls="--",lw=1)
        for v in valleys: ax.axvline(v,color="green",ls=":",lw=1)
        ax.set_xlabel("Arcsinh counts"); ax.set_ylabel("Density")
        ax.set_title(stem,fontsize=9); fig.tight_layout()

        st.session_state.fig_pngs[f"{stem}.png"]=fig_to_png(fig); plt.close(fig)
        st.session_state.results[stem]={"peaks":peaks,"valleys":valleys}
        progress.progress(idx/len(todo),f"Done {stem}")

    st.success("All files processed!")

# ═══════════ history / summary / download (unchanged) ══════════
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

if st.session_state.results:
    df=pd.DataFrame([{"file":k,**v} for k,v in st.session_state.results.items()])
    st.subheader("📋 Summary"); st.dataframe(df,use_container_width=True)
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf,"w") as z:
            z.writestr("summary.csv",df.to_csv(index=False).encode())
            for fn,png in st.session_state.fig_pngs.items(): z.writestr(fn,png)
        st.download_button("⬇️ Download ZIP",buf.getvalue(),"PeakValleyResults.zip","application/zip")
