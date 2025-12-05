import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.kde_detector import kde_peaks_valleys


def test_sharpened_retry_recovers_hidden_negative_peak():
    rng = np.random.default_rng(0)
    data = np.concatenate(
        [rng.normal(-1.0, 0.4, 800), rng.normal(1.0, 0.4, 800)]
    )

    # A deliberately wide bandwidth collapses the doublet into a single mound
    # unless the KDE detector performs a sharper retry.
    peaks, valleys, *_ = kde_peaks_valleys(
        data,
        n_peaks=2,
        grid_size=4_000,
        bw=0.8,
        prominence=0.05,
        min_x_sep=0.6,
    )

    assert len(peaks) == 2
    assert peaks[0] < 0 < peaks[1]
    assert len(valleys) == 1
    assert peaks[0] < valleys[0] < peaks[1]
