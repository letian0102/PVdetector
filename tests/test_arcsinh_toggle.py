import io

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app import _sync_generated_counts


def setup_state():
    st.session_state.clear()
    st.session_state.generated_csvs = []
    st.session_state.generated_meta = {}
    st.session_state.results = {}
    st.session_state.results_raw = {}
    st.session_state.params = {}
    st.session_state.dirty = {}
    st.session_state.aligned_results = {}
    st.session_state.fig_pngs = {}
    st.session_state.aligned_fig_pngs = {}
    st.session_state.aligned_counts = None
    st.session_state.aligned_landmarks = None
    st.session_state.aligned_ridge_png = None


def read_bio(bio: io.BytesIO) -> np.ndarray:
    bio.seek(0)
    df = pd.read_csv(bio)
    return df.iloc[:, 0].to_numpy()


def test_arcsinh_applied():
    setup_state()
    st.session_state.apply_arcsinh = True
    st.session_state.arcsinh_a = 1.0
    st.session_state.arcsinh_b = 0.2
    st.session_state.arcsinh_c = 0.0

    expr_df = pd.DataFrame({"CD3": [10.0, 20.0]})
    meta_df = pd.DataFrame({"sample": ["s1", "s1"]})

    _sync_generated_counts(["CD3"], ["s1"], expr_df, meta_df)
    stem, bio = st.session_state.generated_csvs[0]
    arr = read_bio(bio)
    expected = (1 / 0.2) * np.arcsinh(np.array([10.0, 20.0]))
    assert np.allclose(arr, expected)
    assert getattr(bio, "arcsinh")


def test_arcsinh_skipped():
    setup_state()
    st.session_state.apply_arcsinh = False

    expr_df = pd.DataFrame({"CD3": [5.0, 15.0]})
    meta_df = pd.DataFrame({"sample": ["s1", "s1"]})

    _sync_generated_counts(["CD3"], ["s1"], expr_df, meta_df)
    stem, bio = st.session_state.generated_csvs[0]
    arr = read_bio(bio)
    assert np.allclose(arr, np.array([5.0, 15.0]))
    assert not getattr(bio, "arcsinh")


def test_batches_produce_unique_stems():
    """Generating counts for the same sample across batches should keep both."""
    setup_state()
    st.session_state.apply_arcsinh = False

    expr_df = pd.DataFrame({"CD3": [1.0, 2.0, 3.0, 4.0]})
    meta_df = pd.DataFrame({
        "sample": ["s1", "s1", "s1", "s1"],
        "batch": ["b1", "b1", "b2", "b2"],
    })

    _sync_generated_counts(["CD3"], ["s1"], expr_df, meta_df, ["b1", "b2"])
    assert len(st.session_state.generated_csvs) == 2
    stems = {stem for stem, _ in st.session_state.generated_csvs}
    assert any("b1" in s for s in stems)
    assert any("b2" in s for s in stems)
    # verify each batch's data
    for stem, bio in st.session_state.generated_csvs:
        arr = read_bio(bio)
        if "b1" in stem:
            assert np.allclose(arr, np.array([1.0, 2.0]))
        else:
            assert np.allclose(arr, np.array([3.0, 4.0]))

