import numpy as np
import pandas as pd
import pytest
import streamlit as st

from app import _align_results_by_group, _ordered_stems_for_results, _queue_cli_samples


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


def test_cli_marker_group_mode_sets_groups_and_alignment():
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
            "stem": ["S1_CD3_raw_counts", "S2_CD4_raw_counts"],
            "sample": ["S1", "S2"],
            "marker": [" CD3 ", "CD4"],
        }
    )
    expr_df = pd.DataFrame({"CD3": [0.1, 0.0], "CD4": [0.0, 0.2]})
    meta_df = pd.DataFrame({"sample": ["S1", "S2"]})

    st.session_state.cli_summary_df = summary_df
    st.session_state.expr_df = expr_df
    st.session_state.meta_df = meta_df

    _queue_cli_samples(
        ["S1_CD3_raw_counts", "S2_CD4_raw_counts"],
        group_mode="marker_groups",
        new_group=None,
    )

    assert st.session_state.group_assignments == {
        "S1_CD3_raw_counts": "cd3",
        "S2_CD4_raw_counts": "cd4",
    }
    assert "cd3" in st.session_state.group_overrides
    assert "cd4" in st.session_state.group_overrides
    assert st.session_state.align_group_markers is True


def test_marker_grouping_orders_ridges_and_aligns_per_group():
    st.session_state.clear()
    st.session_state.generated_csvs = []
    st.session_state.generated_meta = {}
    st.session_state.results = {
        "S1_CD3_raw_counts": {
            "peaks": [-1.0],
            "valleys": [],
            "xs": np.array([-1.5, -1.0, -0.5]),
            "ys": np.array([0.1, 0.2, 0.1]),
        },
        "S2_CD4_raw_counts": {
            "peaks": [10.0],
            "valleys": [],
            "xs": np.array([9.5, 10.0, 10.5]),
            "ys": np.array([0.2, 0.4, 0.2]),
        },
    }
    st.session_state.results_raw = {
        "S1_CD3_raw_counts": np.array([-1.6, -1.2, -1.0]),
        "S2_CD4_raw_counts": np.array([9.8, 10.0, 10.4]),
    }
    st.session_state.results_raw_meta = {}
    st.session_state.params = {}
    st.session_state.dirty = {}
    st.session_state.aligned_results = {}
    st.session_state.fig_pngs = {}
    st.session_state.aligned_fig_pngs = {}
    st.session_state.aligned_counts = None
    st.session_state.aligned_landmarks = None
    st.session_state.raw_ridge_png = None
    st.session_state.align_group_markers = True
    st.session_state.group_assignments = {
        "S1_CD3_raw_counts": "cd3",
        "S2_CD4_raw_counts": "cd4",
    }
    st.session_state.group_overrides = {"Default": {}, "cd3": {}, "cd4": {}}

    ordered = _ordered_stems_for_results(use_groups=True)
    assert ordered == ["S1_CD3_raw_counts", "S2_CD4_raw_counts"]

    _align_results_by_group(align_mode="negPeak", target_vec=None)

    peaks_cd3 = st.session_state.aligned_results["S1_CD3_raw_counts"]["peaks"]
    peaks_cd4 = st.session_state.aligned_results["S2_CD4_raw_counts"]["peaks"]

    assert pytest.approx(peaks_cd3[0], rel=0.01) == -1.0
    assert pytest.approx(peaks_cd4[0], rel=0.01) == 10.0


def test_marker_grouping_invalidates_cached_ridges():
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
    st.session_state.aligned_ridge_png = b"cached"
    st.session_state.raw_ridge_png = b"cached"
    st.session_state.apply_arcsinh = False
    st.session_state.cli_counts_native = False
    st.session_state.group_assignments = {}
    st.session_state.group_overrides = {"Default": {}}
    st.session_state.cli_positions_pending = []
    st.session_state.cli_positions_fixed = set()
    st.session_state.workers = 1

    summary_df = pd.DataFrame(
        {
            "stem": ["S1_CD3_raw_counts", "S2_CD4_raw_counts"],
            "sample": ["S1", "S2"],
            "marker": ["CD3", "CD4"],
        }
    )
    expr_df = pd.DataFrame({"CD3": [0.1, 0.2], "CD4": [0.3, 0.4]})
    meta_df = pd.DataFrame({"sample": ["S1", "S2"]})

    st.session_state.cli_summary_df = summary_df
    st.session_state.expr_df = expr_df
    st.session_state.meta_df = meta_df

    _queue_cli_samples(
        ["S1_CD3_raw_counts", "S2_CD4_raw_counts"],
        group_mode="marker_groups",
        new_group=None,
    )

    assert st.session_state.raw_ridge_png is None
    assert st.session_state.aligned_ridge_png is None
