import numpy as np
import pandas as pd
import streamlit as st

from app import (
    _apply_cli_positions,
    _apply_sample_result,
    _cli_assign_groups,
    _load_cli_import,
    _restore_cli_positions,
    _release_cli_positions,
)
from peak_valley.batch import SampleResult


def setup_state():
    st.session_state.clear()
    st.session_state.cli_counts_native = True
    st.session_state.cli_summary_lookup = {}
    st.session_state.cli_positions_cache = {}
    st.session_state.cli_positions_pending = []
    st.session_state.cli_positions_fixed = set()
    st.session_state.results = {}
    st.session_state.fig_pngs = {}
    st.session_state.params = {}
    st.session_state.dirty = {}
    st.session_state.dirty_reason = {}
    st.session_state.results_raw = {}
    st.session_state.results_raw_meta = {}
    st.session_state.aligned_results = {}
    st.session_state.aligned_fig_pngs = {}
    st.session_state.aligned_counts = None
    st.session_state.aligned_landmarks = None
    st.session_state.aligned_ridge_png = None
    st.session_state.generated_meta = {}
    st.session_state.raw_ridge_png = None


def test_apply_cli_positions_overrides_once():
    setup_state()
    st.session_state.cli_summary_lookup = {
        "sample_marker": {"peaks": (1.0, 2.0), "valleys": (1.5,)},
    }
    st.session_state.cli_positions_pending = ["sample_marker"]

    peaks, valleys, applied = _apply_cli_positions("sample_marker", [0.0], [])
    assert applied
    assert peaks == [1.0, 2.0]
    assert valleys == [1.5]
    assert st.session_state.cli_positions_pending == []

    # Subsequent calls should still reuse fixed CLI positions until released
    peaks2, valleys2, applied2 = _apply_cli_positions("sample_marker", [0.0], [])
    assert applied2
    assert peaks2 == [1.0, 2.0]
    assert valleys2 == [1.5]

    _release_cli_positions("sample_marker")
    peaks3, valleys3, applied3 = _apply_cli_positions("sample_marker", [0.0], [])
    assert not applied3
    assert peaks3 == [0.0]
    assert valleys3 == []


def test_apply_cli_positions_requires_cli_flag():
    setup_state()
    st.session_state.cli_counts_native = False
    st.session_state.cli_summary_lookup = {
        "sample_marker": {"peaks": (1.0,), "valleys": ()},
    }
    st.session_state.cli_positions_pending = ["sample_marker"]

    peaks, valleys, applied = _apply_cli_positions("sample_marker", [0.0], [])
    assert not applied
    assert peaks == [0.0]
    # Pending list should remain untouched when CLI mode is inactive
    assert st.session_state.cli_positions_pending == ["sample_marker"]


def test_apply_cli_positions_sanitizes_values():
    setup_state()
    st.session_state.cli_summary_lookup = {
        "sample_marker": {
            "peaks": (" 1.0 ", "bad", 3.0),
            "valleys": (float("nan"), "2.5"),
        },
    }
    st.session_state.cli_positions_pending = ["sample_marker"]

    peaks, valleys, applied = _apply_cli_positions("sample_marker", [0.0], [])
    assert applied
    assert peaks == [1.0, 3.0]
    assert valleys == [2.5]


def test_restore_cli_positions_reverts_consistency_changes():
    setup_state()
    st.session_state.cli_summary_lookup = {
        "stemA_marker": {"peaks": (1.0, 2.0), "valleys": (1.4,)},
    }
    st.session_state.cli_positions_cache = {
        "stemA_marker": {"peaks": (1.0, 2.0), "valleys": (1.4,)},
    }
    st.session_state.cli_positions_pending = ["stemA_marker"]

    peaks, valleys, applied = _apply_cli_positions("stemA_marker", [0.5], [0.8])
    assert applied
    assert peaks == [1.0, 2.0]
    assert valleys == [1.4]

    # Simulate enforcement overwriting imported valley
    st.session_state.results = {
        "stemA_marker": {
            "peaks": [1.0, 2.0],
            "valleys": [1.2],
            "xs": [0.0, 1.0, 2.0],
            "ys": [0.0, 1.0, 0.0],
        }
    }
    st.session_state.results_raw = {"stemA_marker": [0.0, 1.0, 2.0]}
    st.session_state.params = {"stemA_marker": {"bw": 1.0, "prom": 0.1, "n_peaks": 2}}
    st.session_state.cli_positions_fixed = {"stemA_marker"}

    _restore_cli_positions()

    restored = st.session_state.results["stemA_marker"]
    assert restored["peaks"] == [1.0, 2.0]
    assert restored["valleys"] == [1.4]
    assert st.session_state.params["stemA_marker"]["n_peaks"] == 2


def test_cli_assign_groups_aligns_by_sample():
    st.session_state.clear()
    st.session_state.group_assignments = {}
    st.session_state.group_overrides = {"Default": {}}
    st.session_state.results = {}

    subset = pd.DataFrame({
        "stem": ["sampleA_marker", "sampleB_marker"],
        "sample": ["SampleA", "SampleB"],
    })

    applied = _cli_assign_groups(subset, mode="align_sample", new_group=None)

    assert applied
    assert st.session_state.group_assignments["sampleA_marker"] == "SampleA"
    assert st.session_state.group_assignments["sampleB_marker"] == "SampleB"
    assert "SampleA" in st.session_state.group_overrides
    assert "SampleB" in st.session_state.group_overrides


def test_cli_assign_groups_creates_new_group():
    st.session_state.clear()
    st.session_state.group_assignments = {}
    st.session_state.group_overrides = {"Default": {}}
    st.session_state.results = {}

    subset = pd.DataFrame({
        "stem": ["sampleC_marker"],
    })

    blocked = _cli_assign_groups(subset, mode="new_group", new_group="")
    assert not blocked
    assert "sampleC_marker" not in st.session_state.group_assignments

    applied = _cli_assign_groups(subset, mode="new_group", new_group="MyGroup")
    assert applied
    assert st.session_state.group_assignments["sampleC_marker"] == "MyGroup"
    assert "MyGroup" in st.session_state.group_overrides


def test_apply_sample_result_honours_imported_peaks():
    setup_state()
    st.session_state.cli_summary_lookup = {
        "stem_marker": {"peaks": (1.0, 2.0), "valleys": (1.4,)},
    }
    st.session_state.cli_positions_pending = ["stem_marker"]

    res = SampleResult(
        stem="stem_marker",
        peaks=[0.5],
        valleys=[0.9],
        xs=np.array([0.0, 1.0, 2.0]),
        ys=np.array([0.0, 1.0, 0.0]),
        counts=np.array([0.0, 1.0, 2.0]),
        params={"bw": 1.0, "prom": 0.1, "n_peaks": 1},
        quality=0.0,
        metadata={"marker": "CD3"},
    )

    _apply_sample_result(res)

    stored = st.session_state.results["stem_marker"]
    assert stored["peaks"] == [1.0, 2.0]
    assert stored["valleys"] == [1.4]
    assert st.session_state.params["stem_marker"]["n_peaks"] == 2


def test_apply_sample_result_uses_fixed_positions_when_pending_empty():
    setup_state()
    st.session_state.cli_summary_lookup = {
        "stem_marker": {"peaks": (1.0, 2.0), "valleys": (1.4,)},
    }
    st.session_state.cli_positions_pending = []
    st.session_state.cli_positions_fixed = {"stem_marker"}

    res = SampleResult(
        stem="stem_marker",
        peaks=[0.5],
        valleys=[0.9],
        xs=np.array([0.0, 1.0, 2.0]),
        ys=np.array([0.0, 1.0, 0.0]),
        counts=np.array([0.0, 1.0, 2.0]),
        params={"bw": 1.0, "prom": 0.1, "n_peaks": 1},
        quality=0.0,
        metadata={"marker": "CD3"},
    )

    _apply_sample_result(res)

    stored = st.session_state.results["stem_marker"]
    assert stored["peaks"] == [1.0, 2.0]
    assert stored["valleys"] == [1.4]
    assert st.session_state.params["stem_marker"]["n_peaks"] == 2


def test_cli_import_trims_stem_whitespace_for_positions():
    setup_state()
    st.session_state.generated_csvs = []
    st.session_state.generated_meta = {}
    st.session_state.pre_overrides = {}
    st.session_state.group_overrides = {"Default": {}}
    st.session_state.group_assignments = {}
    st.session_state.cli_summary_df = None
    st.session_state.cli_summary_selection = []
    st.session_state.cli_filter_text = ""
    st.session_state.sel_markers = []
    st.session_state.sel_samples = []
    st.session_state.sel_batches = []

    summary_df = pd.DataFrame(
        {
            "stem": ["sample_marker   "],
            "sample": ["S1"],
            "marker": ["CD3"],
            "peaks": ["1.0; 2.0"],
            "valleys": ["1.5"],
        }
    )
    expr_df = pd.DataFrame({"CD3": [0.0, 1.0]})
    meta_df = pd.DataFrame({"sample": ["S1", "S1"]})

    _load_cli_import(
        summary_df,
        expr_df,
        meta_df,
        summary_name="summary.csv",
        expr_name="expr.csv",
        meta_name="meta.csv",
    )

    assert set(st.session_state.cli_summary_lookup) == {"sample_marker"}
    assert st.session_state.cli_positions_pending == ["sample_marker"]

    res = SampleResult(
        stem="sample_marker",
        peaks=[0.5],
        valleys=[0.9],
        xs=np.array([0.0, 1.0]),
        ys=np.array([0.0, 1.0]),
        counts=np.array([0.0, 1.0]),
        params={"bw": 1.0, "prom": 0.1, "n_peaks": 1},
        quality=0.0,
        metadata={"marker": "CD3"},
    )

    _apply_sample_result(res)

    stored = st.session_state.results["sample_marker"]
    assert stored["peaks"] == [1.0, 2.0]
    assert stored["valleys"] == [1.5]
    assert st.session_state.cli_positions_pending == []
