from __future__ import annotations

import json
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

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


def _dummy_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 10), color="white").save(buf, format="PNG")
    return buf.getvalue()


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
    moments = hist["moment_summary"]
    assert moments["mean"] is not None
    assert moments["std"] >= 0
    quantiles = hist.get("quantiles")
    assert quantiles
    assert "p50" in quantiles
    summary_bins = hist["summary_bins"]
    assert summary_bins["bin_count"] == SUMMARY_BINS
    assert len(summary_bins["counts"]) == SUMMARY_BINS
    assert summary_bins["normalized_counts"]
    assert hist.get("sparkline")
    assert len(hist["sparkline"]) == SUMMARY_BINS
    assert hist.get("profile_points")
    assert len(hist["profile_points"]) == SUMMARY_BINS
    assert hist.get("normalized_counts")
    cumulative = hist.get("cumulative_profile")
    assert cumulative
    assert len(cumulative) == SUMMARY_BINS
    assert hist.get("bin_centers")
    assert len(hist["bin_centers"]) == hist["bin_count"]
    assert hist.get("cumulative_counts")
    assert len(hist["cumulative_counts"]) == hist["bin_count"]
    assert hist.get("cumulative_normalized")
    assert len(hist["cumulative_normalized"]) == hist["bin_count"]
    samples = hist.get("profile_samples")
    assert samples
    assert len(samples) == hist["bin_count"]
    first_sample = samples[0]
    assert "x" in first_sample and "count" in first_sample
    assert "smoothed" in first_sample
    assert "cumulative_fraction" in first_sample or hist["bin_count"] == 0
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

    multiscale = payload.get("kde_multiscale")
    assert multiscale
    assert multiscale["grid"]
    assert multiscale["profiles"], "multiscale profiles should not be empty"
    profile0 = multiscale["profiles"][0]
    assert len(profile0["density"]) == len(multiscale["grid"])
    assert profile0.get("scale") is not None
    consensus = multiscale.get("consensus")
    assert consensus
    assert consensus.get("recommended")
    assert "persistent_peaks" in consensus or consensus.get("counts_by_scale")

    vote_summary = payload.get("vote_summary")
    assert vote_summary
    assert vote_summary.get("votes")
    assert vote_summary.get("recommended")

    # JSON serialization should succeed for downstream GPT usage
    json.dumps(payload)

    candidates = payload["candidates"]
    assert candidates["count"] == len(candidates["peaks"])
    assert candidates["shape_description"]
    assert "prominence_threshold" in candidates
    valleys = candidates.get("valleys")
    assert isinstance(valleys, list)
    if valleys:
        assert "relative_height_min" in valleys[0]
    if candidates["count"] >= 1:
        first_peak = candidates["peaks"][0]
        assert "relative_height" in first_peak
        assert "bin_index" in first_peak
        assert "original_order" in first_peak
    if candidates["count"] >= 2:
        assert candidates["pairwise_separations"]


def test_feature_payload_respects_numeric_bandwidth_override():
    counts = _dummy_counts()
    payload = _build_feature_payload(counts, kde_bandwidth=0.42)

    hist = payload["histogram"]
    assert hist["kde_bandwidth_source"] == "provided_value"
    assert hist["kde_bandwidth"] == pytest.approx(0.42)
    assert "kde_bandwidth_rule" not in hist

    kde = payload["kde"]
    assert kde["bandwidth"] == pytest.approx(0.42)
    assert "bandwidth_rule" not in kde


def test_feature_payload_records_bandwidth_rule_override():
    counts = _dummy_counts()
    payload = _build_feature_payload(counts, kde_bandwidth="silverman")

    hist = payload["histogram"]
    assert hist["kde_bandwidth_source"] == "provided_rule"
    assert hist.get("kde_bandwidth_rule") == "silverman"

    kde = payload["kde"]
    assert kde.get("bandwidth_rule") == "silverman"
    assert kde["bandwidth"] is None or kde["bandwidth"] > 0


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
    assert captured.get("kde_multiscale")
    meta = captured.get("meta", {})
    assert meta.get("consensus_peak_count") is not None


def test_ask_gpt_peak_count_passes_bandwidth_metadata():
    counts = _dummy_counts()
    captured: dict | None = None

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            nonlocal captured
            user = next(msg for msg in kwargs["messages"] if msg["role"] == "user")
            captured = json.loads(user["content"])
            response = {"peak_count": 1, "confidence": 0.5, "reason": "ok", "peak_indices": [0]}
            message = SimpleNamespace(content=json.dumps(response))
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    client = DummyClient()
    result = ask_gpt_peak_count(
        client,
        model_name="dummy",
        max_peaks=3,
        counts_full=counts,
        kde_bandwidth="scott",
    )

    assert result == 1
    assert captured is not None
    hist = captured.get("histogram", {})
    assert hist.get("kde_bandwidth_source") == "provided_rule"
    assert hist.get("kde_bandwidth_rule") == "scott"


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
    assert heuristics.get("three_support_gaps") == []
    consensus = heuristics.get("multiscale_consensus")
    assert isinstance(consensus, dict)
    assert consensus.get("recommended") == 3


def test_gpt_peak_count_request_includes_image_payload():
    counts = _dummy_counts()
    image_bytes = _dummy_png()

    captured_content = None

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            nonlocal captured_content
            user_msg = next(msg for msg in kwargs["messages"] if msg["role"] == "user")
            captured_content = user_msg["content"]
            response = {
                "peak_count": 2,
                "confidence": 0.5,
                "reason": "looks bimodal",
                "peak_indices": [0, 1],
            }
            message = SimpleNamespace(content=json.dumps(response))
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    client = DummyClient()

    result = ask_gpt_peak_count(
        client,
        model_name="dummy",
        max_peaks=4,
        counts_full=counts,
        distribution_image=image_bytes,
    )

    assert result == 2
    assert isinstance(captured_content, list)
    assert captured_content
    assert captured_content[0]["type"] == "text"
    image_parts = [part for part in captured_content if part.get("type") == "input_image"]
    assert image_parts, "Expected input_image payload when distribution image provided"
    image_url = image_parts[0].get("image_url", {}).get("url")
    assert isinstance(image_url, str) and image_url.startswith("data:image/png;base64,")


def test_distribution_preview_passes_bandwidth_keyword(monkeypatch):
    import importlib
    import os

    os.environ.setdefault("STREAMLIT_SUPPRESS_RUN_CONTEXT_WARNING", "1")
    app_module = importlib.import_module("app")

    captured_kwargs: dict[str, object] = {}
    captured_plot: dict[str, object] = {}

    def fake_kde_peaks_valleys(values, n_peaks, prominence, **kwargs):
        captured_kwargs.update(kwargs)
        xs = np.linspace(-1.0, 1.0, 8)
        ys = np.linspace(0.0, 1.0, 8)
        return [0.0], [0.5], xs, ys

    def fake_plot_png(stem, xs, ys, peaks, valleys):
        captured_plot.update({
            "stem": stem,
            "xs": xs,
            "ys": ys,
            "peaks": list(peaks),
            "valleys": list(valleys),
        })
        return b"preview"

    monkeypatch.setattr(app_module, "kde_peaks_valleys", fake_kde_peaks_valleys)
    monkeypatch.setattr(app_module, "_plot_png", fake_plot_png)

    preview = app_module._gpt_distribution_preview(
        "demo",
        _dummy_counts(),
        prominence=0.05,
        bandwidth="scott",
        min_width=None,
        grid_size=128,
        drop_fraction=0.1,
        min_separation=0.5,
        curvature=None,
        turning_peak=False,
        first_valley_mode="slope",
    )

    assert preview == b"preview"
    assert captured_kwargs.get("bw") == "scott"
    assert captured_plot["stem"] == "demo"
    assert captured_plot["peaks"] == []
    assert captured_plot["valleys"] == []
