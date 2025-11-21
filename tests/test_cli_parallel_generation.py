import numpy as np
import pandas as pd
import streamlit as st

from app import _sync_generated_counts


def test_sync_generated_counts_uses_workers(monkeypatch):
    st.session_state.clear()
    st.session_state.generated_csvs = []
    st.session_state.generated_meta = {}
    st.session_state.results = {}
    st.session_state.results_raw = {}
    st.session_state.results_raw_meta = {}
    st.session_state.params = {}
    st.session_state.dirty = {}
    st.session_state.aligned_results = {}
    st.session_state.fig_pngs = {}
    st.session_state.aligned_fig_pngs = {}
    st.session_state.aligned_counts = None
    st.session_state.aligned_landmarks = None
    st.session_state.aligned_ridge_png = None
    st.session_state.raw_ridge_png = None
    st.session_state.apply_arcsinh = False
    st.session_state.cli_counts_native = False
    st.session_state.workers = 3

    expr = pd.DataFrame(
        {
            "marker_a": np.arange(4),
            "marker_b": np.arange(10, 14),
        }
    )
    meta = pd.DataFrame({"sample": ["S1"] * 4})

    _sync_generated_counts(["marker_a", "marker_b"], ["S1"], expr, meta)

    stems = {stem for stem, _ in st.session_state.generated_csvs}
    assert stems == {"S1_marker_a_raw_counts", "S1_marker_b_raw_counts"}

    meta_cells = st.session_state.generated_meta["S1_marker_a_raw_counts"]["cell_indices"]
    assert meta_cells == [0, 1, 2, 3]
