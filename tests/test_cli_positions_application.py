import pandas as pd
import streamlit as st

from app import _apply_cli_positions, _restore_cli_positions, _cli_assign_groups


def setup_state():
    st.session_state.clear()
    st.session_state.cli_counts_native = True
    st.session_state.cli_summary_lookup = {}
    st.session_state.cli_positions_cache = {}
    st.session_state.cli_positions_pending = []
    st.session_state.cli_positions_fixed = set()


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

    # Subsequent calls should not reapply once pending list is cleared
    peaks2, valleys2, applied2 = _apply_cli_positions("sample_marker", [0.0], [])
    assert not applied2
    assert peaks2 == [0.0]
    assert valleys2 == []


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
