import numpy as np

from peak_valley.kde_detector import kde_peaks_valleys


def test_relative_prominence_prunes_shoulder_peak():
    rng = np.random.default_rng(42)
    dominant = rng.normal(0.0, 0.35, size=800)
    shoulder = rng.normal(1.8, 0.05, size=12)
    data = np.concatenate([dominant, shoulder])

    peaks_loose, *_ = kde_peaks_valleys(
        data,
        prominence=0.01,
        relative_prominence=0.0,
        min_x_sep=0.2,
    )
    peaks_filtered, *_ = kde_peaks_valleys(
        data,
        prominence=0.01,
        relative_prominence=0.25,
        min_x_sep=0.2,
    )

    assert len(peaks_loose) >= 2
    assert len(peaks_filtered) == 1
