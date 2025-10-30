from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.gpt_adapter import (
    SUMMARY_BINS,
    BASELINE_BINS,
    _build_feature_payload,
    ask_gpt_peak_count,
)


def _dummy_counts() -> np.ndarray:
    left = np.linspace(-2.0, 0.0, 60)
    right = np.linspace(2.5, 4.0, 60)
    return np.concatenate([left, right])


def test_feature_payload_includes_kde_trace():
    counts = _dummy_counts()
    payload = _build_feature_payload(counts)

    assert "histogram" in payload
    hist = payload["histogram"]
    assert hist["bin_count"] >= BASELINE_BINS
    assert hist["bin_edges"]
    assert hist["counts"]
    assert hist["run_length_counts"]
    assert hist["smoothed_counts"]
    assert hist["second_derivative_summary"]["max_abs"] >= 0
    summary_bins = hist["summary_bins"]
    assert summary_bins["bin_count"] == SUMMARY_BINS
    assert len(summary_bins["counts"]) == SUMMARY_BINS
    assert hist.get("sparkline")
    assert len(hist["sparkline"]) == SUMMARY_BINS
    assert hist.get("profile_points")
    assert len(hist["profile_points"]) == SUMMARY_BINS
    runs = hist.get("slope_runs")
    assert isinstance(runs, list)
    assert runs, "slope_runs should capture monotonic segments"
    projection = hist.get("bandwidth_projection")
    assert projection is None or projection["sigma_bins"] > 0

    assert "kde" in payload
    kde = payload["kde"]

    assert kde["bandwidth"] is None or kde["bandwidth"] > 0
    assert kde["scale"] is None or kde["scale"] > 0
    assert len(kde["x"]) == len(kde["density"])
    assert 128 <= len(kde["x"]) <= 256

    # JSON serialization should succeed for downstream GPT usage
    json.dumps(payload)


def test_ask_gpt_peak_count_accepts_missing_kde():
    counts = _dummy_counts()
    features = _build_feature_payload(counts)
    features.pop("kde", None)
    hist = features.get("histogram", {})
    hist.pop("kde_bandwidth", None)
    hist.pop("kde_scale", None)

    captured: dict | None = None

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            nonlocal captured
            user = next(msg for msg in kwargs["messages"] if msg["role"] == "user")
            captured = json.loads(user["content"])
            response = {"peak_count": 1, "confidence": 0.7, "reason": "ok", "peak_indices": [0]}
            message = SimpleNamespace(content=json.dumps(response))
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    client = DummyClient()
    result = ask_gpt_peak_count(
        client,
        model_name="dummy",
        max_peaks=3,
        counts_full=counts,
        features=features,
    )

    assert result == 1
    assert captured is not None
    assert "kde" not in captured
    assert "kde_bandwidth" not in captured.get("histogram", {})


def test_peak_caps_allow_three_with_clear_triplet():
    rng = np.random.default_rng(123)
    counts = np.concatenate(
        [
            rng.normal(-3.0, 0.25, size=250),
            rng.normal(0.0, 0.3, size=260),
            rng.normal(3.1, 0.28, size=240),
        ]
    )

    features = _build_feature_payload(counts)

    captured: dict | None = None

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            nonlocal captured
            user = next(msg for msg in kwargs["messages"] if msg["role"] == "user")
            captured = json.loads(user["content"])
            response = {
                "peak_count": 3,
                "confidence": 0.8,
                "reason": "clear triplet",
                "peak_indices": [0, 1, 2],
            }
            message = SimpleNamespace(content=json.dumps(response))
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    client = DummyClient()
    result = ask_gpt_peak_count(
        client,
        model_name="dummy",
        max_peaks=3,
        counts_full=counts,
        features=features,
    )

    assert result == 3
    assert captured is not None
    heuristics = captured.get("heuristics", {}) if captured else {}
    assert heuristics.get("final_allowed_max") == 3
    assert heuristics.get("evidence_for_three") is True
