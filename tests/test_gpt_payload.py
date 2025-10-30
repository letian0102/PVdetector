from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.gpt_adapter import _build_feature_payload, ask_gpt_peak_count


def _dummy_counts() -> np.ndarray:
    left = np.linspace(-2.0, 0.0, 60)
    right = np.linspace(2.5, 4.0, 60)
    return np.concatenate([left, right])


def test_feature_payload_includes_kde_trace():
    counts = _dummy_counts()
    payload = _build_feature_payload(counts)

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
