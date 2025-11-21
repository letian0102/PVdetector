import pandas as pd
import streamlit as st

from app import _queue_cli_samples


def test_queue_cli_samples_explains_missing_counts(monkeypatch):
    st.session_state.clear()
    st.session_state.cli_summary_df = pd.DataFrame(
        {
            "stem": ["sampleA_markerA"],
            "sample": ["SampleA"],
            "marker": ["MarkerA"],
        }
    )
    st.session_state.expr_df = pd.DataFrame({"MarkerA": [1.0, 2.0, 3.0]})
    # Metadata intentionally omits SampleA to trigger the missing-cells path
    st.session_state.meta_df = pd.DataFrame({"sample": ["Other"], "batch": [pd.NA]})
    st.session_state.generated_csvs = []
    st.session_state.generated_meta = {}
    st.session_state.pre_overrides = {}
    st.session_state.group_assignments = {}
    st.session_state.group_overrides = {"Default": {}}
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

    warnings: list[str] = []
    monkeypatch.setattr(st, "warning", lambda msg: warnings.append(str(msg)))

    _queue_cli_samples(["sampleA_markerA"], group_mode="none", new_group=None)

    assert any("No cells for sample 'SampleA'" in msg for msg in warnings)
