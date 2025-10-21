import streamlit as st

from app import _apply_cli_positions


def setup_state():
    st.session_state.clear()
    st.session_state.cli_counts_native = True
    st.session_state.cli_summary_lookup = {}
    st.session_state.cli_positions_pending = []


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
