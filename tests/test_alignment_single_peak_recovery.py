import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.alignment import align_distributions, _augment_single_peak_samples


def _synthetic_counts(rng: np.random.Generator, neg_center: float, pos_center: float) -> np.ndarray:
    neg = rng.normal(neg_center, 0.03, size=1500)
    pos = rng.normal(pos_center, 0.05, size=2000)
    return np.concatenate([neg, pos])


def test_detects_positive_only_peak_and_recovers_negative():
    rng = np.random.default_rng(42)

    counts = [
        _synthetic_counts(rng, 0.1, 1.0),
        _synthetic_counts(rng, 0.12, 0.95),
        _synthetic_counts(rng, 0.14, 0.98),
    ]

    peaks = [
        [0.1, 1.0],
        [0.12, 0.95],
        [0.98],  # detector only found the positive peak
    ]
    valleys = [[0.5], [0.48], []]

    adjusted_peaks, adjusted_valleys = _augment_single_peak_samples(peaks, valleys, counts)

    assert len(adjusted_peaks[2]) == 2
    assert adjusted_peaks[2][0] < adjusted_peaks[2][1]
    assert adjusted_peaks[2][0] < 0.5  # recovered negative peak lies to the left

    warped_counts, warped_landmarks, *_ = align_distributions(counts, peaks, valleys)

    target_negative = warped_landmarks[0, 0]
    assert target_negative < 0.2

