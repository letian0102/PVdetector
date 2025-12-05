import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.alignment import fill_landmark_matrix


def test_fill_landmark_matrix_preserves_observed_positive_peaks():
    peaks = [
        [0.1, 1.2],   # has explicit positive peak
        [0.05],       # only negative peak detected
        [0.2, 0.9],   # also has a positive peak
    ]
    valleys = [
        [0.45],
        [0.32],
        [0.4],
    ]

    mat = fill_landmark_matrix(peaks, valleys)

    assert mat.shape == (3, 3)

    # Samples with real positive peaks keep their original location
    assert mat[0, 2] == pytest.approx(1.2)
    assert mat[2, 2] == pytest.approx(0.9)

    # Samples missing a positive peak receive an imputed landmark
    assert mat[1, 2] > mat[1, 1]


def test_fill_landmark_matrix_respects_alignment_mode():
    peaks = [[0.1], [0.2]]
    valleys = [[0.4], [0.45]]

    mat_neg_only = fill_landmark_matrix(peaks, valleys, align_type="negPeak")
    assert mat_neg_only.shape == (2, 1)
    assert np.allclose(mat_neg_only[:, 0], [0.1, 0.2])

    mat_neg_val = fill_landmark_matrix(
        peaks,
        valleys,
        align_type="negPeak_valley",
    )
    assert mat_neg_val.shape == (2, 2)
    assert np.allclose(mat_neg_val[:, 0], [0.1, 0.2])
    assert np.allclose(mat_neg_val[:, 1], [0.4, 0.45])


def test_single_peak_samples_align_to_positive_mode_when_right_shifted():
    peaks = [
        [0.1, 1.0],   # canonical negative/positive pair
        [0.11, 0.98],
        [0.93],       # single peak near the positive mode
    ]
    valleys = [
        [0.5],
        [0.52],
        [0.51],
    ]

    mat = fill_landmark_matrix(peaks, valleys)

    # The single-peak sample should contribute its observed peak to the
    # positive column so it can align with other positive peaks.
    assert mat[2, 2] == pytest.approx(0.93)
    # A negative anchor is imputed from the cohort, not copied from the
    # right-shifted single peak.
    assert mat[2, 0] != pytest.approx(0.93)


def test_single_peak_samples_stay_on_negative_mode_when_left_shifted():
    peaks = [
        [0.12, 0.95],
        [0.14, 1.02],
        [0.15],
    ]
    valleys = [
        [0.48],
        [0.5],
        [0.47],
    ]

    mat = fill_landmark_matrix(peaks, valleys)

    # The lone peak is left of the cohort valley so it should remain a
    # negative-mode anchor.
    assert mat[2, 0] == pytest.approx(0.15)
    assert mat[2, 2] != pytest.approx(0.15)
