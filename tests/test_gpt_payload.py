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
    _min_separation_window,
    _bandwidth_window,
    _aggregate_multi_marker_windows,
    ask_gpt_parameter_plan,
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


def test_parameter_plan_clamps_and_fills_defaults():
    counts = _dummy_counts()
    features = _build_feature_payload(counts)

    captured_payload = None

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            nonlocal captured_payload
            user = next(msg for msg in kwargs["messages"] if msg["role"] == "user")
            captured_payload = json.loads(user["content"])
            response = {
                "bandwidth": 2.0,  # beyond cap
                "min_separation": 5.0,  # beyond dynamic clamp
                "prominence": 0.9,
                "peak_cap": 9,
                "apply_turning_points": "yes",
                "notes": "stress test",
            }
            message = SimpleNamespace(content=json.dumps(response))
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    defaults = {
        "bandwidth": "scott",
        "min_separation": 0.5,
        "prominence": 0.05,
        "peak_cap": 4,
        "apply_turning_points": False,
    }

    client = DummyClient()
    plan = ask_gpt_parameter_plan(
        client,
        model_name="dummy",
        counts_full=counts,
        max_peaks=6,
        defaults=defaults,
        features=features,
    )

    bw_floor, bw_cap, bw_default = _bandwidth_window(
        counts, features, defaults["bandwidth"]
    )
    min_floor, max_min_sep, inferred_default = _min_separation_window(
        counts, features, defaults["min_separation"], max_peaks=6
    )

    assert plan["bandwidth"] == bw_cap
    assert plan["min_separation"] == max_min_sep
    assert plan["min_separation"] <= defaults["min_separation"]
    assert min_floor <= plan["min_separation"] <= max_min_sep
    assert inferred_default <= max_min_sep
    assert bw_floor <= plan["bandwidth"] <= bw_cap
    assert bw_default <= bw_cap
    assert plan["prominence"] == 0.3
    assert plan["peak_cap"] == 6
    assert plan["apply_turning_points"] is True
    assert plan["notes"] == "stress test"
    assert captured_payload is not None


def test_parameter_plan_backfills_empty_notes():
    counts = _dummy_counts()
    features = _build_feature_payload(counts)

    captured_payload = None

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            nonlocal captured_payload
            user = next(msg for msg in kwargs["messages"] if msg["role"] == "user")
            captured_payload = json.loads(user["content"])
            response = {
                "bandwidth": "scott",
                "min_separation": 0.1,
                "prominence": 0.05,
                "peak_cap": 3,
                "apply_turning_points": False,
                "notes": " ",  # simulate empty explanation
            }
            message = SimpleNamespace(content=json.dumps(response))
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    defaults = {
        "bandwidth": "scott",
        "min_separation": 0.25,
        "prominence": 0.05,
        "peak_cap": 3,
        "apply_turning_points": False,
    }

    client = DummyClient()
    plan = ask_gpt_parameter_plan(
        client,
        model_name="dummy",
        counts_full=counts,
        max_peaks=4,
        defaults=defaults,
        features=features,
    )

    bw_floor, bw_cap, _ = _bandwidth_window(counts, features, defaults["bandwidth"])
    min_floor, max_min_sep, _ = _min_separation_window(
        counts, features, defaults["min_separation"], max_peaks=4
    )

    assert captured_payload is not None
    assert plan["notes"]
    assert f"[{bw_floor:.3g}, {bw_cap:.3g}]" in plan["notes"]
    assert f"[{min_floor:.3g}, {max_min_sep:.3g}]" in plan["notes"]


def test_parameter_plan_uses_multi_marker_windows():
    rng = np.random.default_rng(7)
    marker_a = rng.normal(0.0, 0.12, size=450)
    marker_b = rng.normal(0.0, 0.85, size=450)

    pooled = np.concatenate([marker_a, marker_b])
    marker_entries = []
    for name, values in (("A", marker_a), ("B", marker_b)):
        marker_entries.append(
            {"marker": name, "values": values, "features": _build_feature_payload(values)}
        )

    pooled_features = _build_feature_payload(pooled)
    pooled_features["multi_marker_features"] = marker_entries

    defaults = {
        "bandwidth": "scott",
        "min_separation": 0.4,
        "prominence": 0.05,
        "peak_cap": 4,
        "apply_turning_points": False,
    }

    aggregated_bw, aggregated_min = _aggregate_multi_marker_windows(
        marker_entries, defaults, max_peaks=5
    )
    pooled_bw = _bandwidth_window(pooled, pooled_features, defaults["bandwidth"])
    pooled_min = _min_separation_window(
        pooled, pooled_features, defaults["min_separation"], max_peaks=5
    )

    assert aggregated_bw is not None and aggregated_min is not None
    assert aggregated_bw[1] <= pooled_bw[1]
    assert aggregated_min[1] <= pooled_min[1]

    captured_payload = None

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            nonlocal captured_payload
            user = next(msg for msg in kwargs["messages"] if msg["role"] == "user")
            captured_payload = json.loads(user["content"])
            response = {
                "bandwidth": 5.0,
                "min_separation": 2.0,
                "prominence": 0.07,
                "peak_cap": 9,
                "apply_turning_points": True,
                "notes": "multi", 
            }
            message = SimpleNamespace(content=json.dumps(response))
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    client = DummyClient()
    plan = ask_gpt_parameter_plan(
        client,
        model_name="dummy",
        counts_full=pooled,
        max_peaks=5,
        defaults=defaults,
        features=pooled_features,
    )

    assert captured_payload is not None
    assert captured_payload.get("multi_marker_features")
    assert plan["bandwidth"] == aggregated_bw[1]
    assert plan["min_separation"] == aggregated_min[1]
    assert plan["peak_cap"] == 5


def test_parameter_plan_returns_defaults_without_client():
    defaults = {
        "bandwidth": "roughness",
        "min_separation": 0.75,
        "prominence": 0.07,
        "peak_cap": 5,
        "apply_turning_points": True,
    }

    plan = ask_gpt_parameter_plan(
        client=None,  # type: ignore[arg-type]
        model_name="none",
        counts_full=_dummy_counts(),
        max_peaks=5,
        defaults=defaults,
    )

    assert plan["bandwidth"] == "roughness"
    assert plan["min_separation"] == 0.75
    assert plan["prominence"] == 0.07
    assert plan["peak_cap"] == 5
    assert plan["apply_turning_points"] is True
