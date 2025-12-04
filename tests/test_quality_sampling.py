import numpy as np

from peak_valley import quality


def test_stain_quality_downsamples_large_inputs(monkeypatch):
    captured = {}

    class DummyKDE:
        def __init__(self, data, bw_method=None):
            captured["size"] = np.asarray(data).size

        def __call__(self, xs):
            return np.ones_like(xs, float) * 0.1

    monkeypatch.setattr(quality, "gaussian_kde", DummyKDE)

    counts = np.linspace(0, 10, 200_000)
    score = quality.stain_quality(counts, peaks=[2.0, 8.0], valleys=[5.0])

    assert np.isfinite(score)
    assert captured["size"] == quality._QUALITY_SAMPLE_LIMIT
