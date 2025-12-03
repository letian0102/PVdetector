import numpy as np

from peak_valley.kde_detector import kde_peaks_valleys


def test_multiscale_votes_filter_narrow_noise():
    np.random.seed(2)
    data = np.concatenate(
        [
            np.random.normal(-1, 0.08, 400),
            np.random.normal(0, 0.005, 5),
            np.random.normal(1, 0.08, 400),
        ]
    )

    base_peaks, *_ = kde_peaks_valleys(
        data,
        prominence=0.02,
        bw=0.05,
        min_x_sep=0.3,
        grid_size=4000,
    )
    ms_peaks, *_ = kde_peaks_valleys(
        data,
        prominence=0.02,
        bw=0.05,
        min_x_sep=0.3,
        grid_size=4000,
        multiscale=[0.5, 1.0, 4.0],
        multiscale_fraction=0.75,
    )

    assert len(base_peaks) == 3
    assert len(ms_peaks) == 2
    assert np.allclose(sorted(ms_peaks), [-1.0, 0.985], atol=0.02)


def test_auto_shoulders_adds_hidden_mode():
    np.random.seed(11)
    data = np.concatenate(
        [
            np.random.normal(-0.3, 0.05, 400),
            np.random.normal(0.3, 0.05, 400),
        ]
    )

    base_peaks, *_ = kde_peaks_valleys(
        data,
        prominence=0.05,
        bw=1.0,
        min_x_sep=0.25,
        grid_size=4000,
    )
    shoulder_peaks, *_ = kde_peaks_valleys(
        data,
        prominence=0.05,
        bw=1.0,
        min_x_sep=0.25,
        grid_size=4000,
        auto_shoulders=True,
    )

    assert len(base_peaks) == 1
    assert len(shoulder_peaks) == 2
    assert np.allclose(sorted(shoulder_peaks), [-0.01, 0.3], atol=0.05)


def test_lowland_merge_collapses_shallow_valley():
    np.random.seed(16)
    data = np.concatenate(
        [
            np.random.normal(-0.5, 0.05, 300),
            np.random.normal(0.0, 0.05, 80),
            np.random.normal(0.5, 0.05, 300),
        ]
    )

    base_peaks, *_ = kde_peaks_valleys(
        data,
        prominence=0.02,
        min_x_sep=0.15,
        grid_size=4000,
    )
    merged_peaks, *_ = kde_peaks_valleys(
        data,
        prominence=0.02,
        min_x_sep=0.15,
        grid_size=4000,
        valley_merge=0.2,
    )

    assert len(base_peaks) == 3
    assert len(merged_peaks) == 2
    assert np.allclose(sorted(merged_peaks), [-0.5, 0.5], atol=0.02)
