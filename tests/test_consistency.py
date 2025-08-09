import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peak_valley.consistency import enforce_marker_consistency


def test_enforce_marker_consistency_handles_multiple_landmarks():
    xs = np.linspace(0.0, 4.0, 401)
    ys = np.exp(-(xs - 1) ** 2) + np.exp(-(xs - 3) ** 2)

    results = {
        "s1": {"xs": xs.tolist(), "ys": ys.tolist(),
                "peaks": [1.0, 3.0], "valleys": [2.0]},
        "s2": {"xs": xs.tolist(), "ys": ys.tolist(),
                "peaks": [1.4, 2.0], "valleys": [2.3]},
        "s3": {"xs": xs.tolist(), "ys": ys.tolist(),
                "peaks": [1.0], "valleys": []},
    }

    enforce_marker_consistency(results, tol=0.3, window=1.0)

    assert abs(results["s2"]["peaks"][0] - 1.4) < 1e-6
    assert abs(results["s2"]["valleys"][0] - 2.3) < 1e-6
    assert abs(results["s2"]["peaks"][1] - 3.0) < 0.1
    assert len(results["s3"]["peaks"]) == 2
    assert abs(results["s3"]["peaks"][1] - 3.0) < 0.1
    assert results["s3"]["valleys"] == []
