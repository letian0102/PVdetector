import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peak_valley.consistency import enforce_marker_consistency


def test_enforce_marker_consistency_handles_multiple_landmarks():
    xs = np.linspace(0.0, 4.0, 401)
    ys = np.exp(-(xs - 1) ** 2) + np.exp(-(xs - 3) ** 2)

    results = {
        "s1": {"xs": xs.tolist(), "ys": ys.tolist(), "peaks": [1.4, 3.0], "valleys": [2.7]},
        "s2": {"xs": xs.tolist(), "ys": ys.tolist(), "peaks": [1.0, 3.0], "valleys": [2.0]},
        "s3": {"xs": xs.tolist(), "ys": ys.tolist(), "peaks": [1.0], "valleys": []},
    }

    enforce_marker_consistency(results, tol=0.3, window=1.0)

    assert abs(results["s1"]["peaks"][0] - 1.0) < 0.1
    assert abs(results["s1"]["valleys"][0] - 2.0) < 0.1
    assert abs(results["s2"]["peaks"][0] - 1.0) < 1e-6
    assert abs(results["s2"]["valleys"][0] - 2.0) < 1e-6
    assert abs(results["s2"]["peaks"][1] - 3.0) < 0.1
    assert len(results["s3"]["peaks"]) == 2
    assert abs(results["s3"]["peaks"][1] - 3.0) < 0.1
    assert len(results["s3"]["valleys"]) == 1
    assert abs(results["s3"]["valleys"][0] - 2.0) < 0.1


def test_enforce_marker_consistency_valley_rules():
    xs = np.linspace(0.0, 4.0, 401)
    ys = np.exp(-(xs - 1) ** 2) + np.exp(-(xs - 3) ** 2)

    results = {
        "bad": {"xs": xs.tolist(), "ys": ys.tolist(), "peaks": [1.0, 3.0], "valleys": [3.5]},
        "bad2": {"xs": xs.tolist(), "ys": ys.tolist(), "peaks": [1.0, 3.0], "valleys": [2.0, 3.5]},
        "good": {"xs": xs.tolist(), "ys": ys.tolist(), "peaks": [1.0, 3.0], "valleys": [2.0]},
    }

    enforce_marker_consistency(results, tol=0.3, window=1.0)

    for info in results.values():
        peaks = info["peaks"]
        valleys = info["valleys"]
        assert len(valleys) <= max(0, len(peaks) - 1)
        for val, left, right in zip(valleys, peaks[:-1], peaks[1:]):
            assert left < val < right


def test_enforce_marker_consistency_fills_missing_first_landmarks():
    xs = np.linspace(0.0, 4.0, 401)
    ys = np.exp(-(xs - 1) ** 2) + 0.8 * np.exp(-(xs - 3) ** 2)

    results = {
        "reference": {
            "xs": xs.tolist(),
            "ys": ys.tolist(),
            "peaks": [1.0, 3.0],
            "valleys": [2.0],
        },
        "missing": {
            "xs": xs.tolist(),
            "ys": ys.tolist(),
            "peaks": [3.0],
            "valleys": [],
        },
        "shifted": {
            "xs": xs.tolist(),
            "ys": ys.tolist(),
            "peaks": [1.8, 3.0],
            "valleys": [2.5],
        },
    }

    enforce_marker_consistency(results, tol=0.2, window=1.0)

    # Missing landmarks are imputed near the consensus and obey ordering
    miss_peaks = results["missing"]["peaks"]
    miss_valleys = results["missing"]["valleys"]
    assert len(miss_peaks) == 2
    assert len(miss_valleys) == 1
    assert abs(miss_peaks[0] - 1.0) < 0.15
    assert abs(miss_valleys[0] - 2.0) < 0.2
    assert miss_peaks[0] < miss_valleys[0] < miss_peaks[1]

    # Outlier landmarks are snapped back toward the local extrema
    shift_peaks = results["shifted"]["peaks"]
    shift_valleys = results["shifted"]["valleys"]
    assert abs(shift_peaks[0] - 1.0) < 0.15
    assert abs(shift_valleys[0] - 2.0) < 0.2
