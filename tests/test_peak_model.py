import numpy as np
import pytest
import peak_valley.kde_detector as kde_detector
from peak_valley.kde_detector import kde_peaks_valleys
from peak_valley.peak_model import GradientBoostingPeakScorer


def test_peak_model_scores_central_peak():
    trace = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    labels = np.array([0, 1, 0, 0, 0])
    scorer = GradientBoostingPeakScorer(window_radius=1).fit([trace], [labels])

    xs = np.arange(trace.size)
    probs = scorer.score_grid(xs, trace)

    assert probs[1] > probs.mean()
    assert probs[1] == pytest.approx(np.max(probs))


def test_kde_peaks_valleys_prefers_model_map():
    rng = np.random.default_rng(0)
    data = np.concatenate([
        rng.normal(loc=-2.0, scale=0.05, size=150),
        rng.normal(loc=2.0, scale=0.05, size=150),
    ])

    class DummyModel:
        def score_grid(self, xs, ys):
            probs = np.zeros_like(xs, dtype=float)
            probs[len(xs) // 2] = 0.9
            return probs

    peaks, valleys, xs, ys = kde_peaks_valleys(
        data,
        grid_size=101,
        min_x_sep=0.1,
        peak_model=DummyModel(),
        peak_model_threshold=0.8,
        peak_model_min_confidence=0.8,
    )

    assert peaks == [pytest.approx(xs[len(xs) // 2])]
    assert valleys


def test_kde_peaks_valleys_falls_back_on_low_confidence(monkeypatch):
    calls = {"used": False}

    def fake_find_peaks(*args, **kwargs):
        calls["used"] = True
        return np.array([1, 3]), {}

    monkeypatch.setattr(kde_detector, "find_peaks", fake_find_peaks)

    class LowModel:
        def score_grid(self, xs, ys):
            return np.zeros_like(xs, dtype=float) + 0.1

    data = np.random.default_rng(0).normal(0, 0.1, 100)
    peaks, valleys, xs, ys = kde_peaks_valleys(
        data,
        grid_size=101,
        peak_model=LowModel(),
        peak_model_threshold=0.0,
        peak_model_min_confidence=0.5,
    )

    assert calls["used"] is True
    assert peaks  # still returns heuristic peaks
    assert xs.size == ys.size
