import pandas as pd
import streamlit as st

from app import _queue_cli_samples


def test_cli_queue_preserves_marker_labels(tmp_path):
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
    st.session_state.group_assignments = {}
    st.session_state.group_overrides = {"Default": {}}
    st.session_state.cli_positions_pending = []
    st.session_state.cli_positions_fixed = set()
    st.session_state.workers = 1

    summary_df = pd.DataFrame(
        {
            "stem": ["S1_CD3_raw_counts"],
            "sample": ["S1"],
            "marker": [" CD3 "],
        }
    )
    expr_df = pd.DataFrame({"CD3": [0.1, 0.2, 0.3]})
    meta_df = pd.DataFrame({"sample": ["S1", "S1", "S1"]})

    st.session_state.cli_summary_df = summary_df
    st.session_state.expr_df = expr_df
    st.session_state.meta_df = meta_df

    _queue_cli_samples(["S1_CD3_raw_counts"], group_mode="none", new_group=None)

    assert st.session_state.sel_markers == ["CD3"]
    meta_entry = st.session_state.generated_meta["S1_CD3_raw_counts"]
    assert meta_entry["marker"] == "CD3"

    stem, counts_file = st.session_state.generated_csvs[0]
    counts_df = pd.read_csv(counts_file)
    assert stem == "S1_CD3_raw_counts"
    assert list(counts_df.columns) == ["CD3"]
