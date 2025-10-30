from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from peak_valley.gpt_adapter import _apply_peak_caps, _default_priors


def _feature_payload(
    *,
    weights_k2,
    delta_bic_21,
    ashman_d,
    valley_ratio,
    prominence_ratio,
    right_tail,
    peak_positions,
    peak_width,
    weights_k3=None,
    delta_bic_32=None,
):
    peaks = [{"x": float(x), "width": float(peak_width)} for x in peak_positions]

    payload = {
        "candidates": {
            "peaks": peaks,
            "valley_depth_ratio": valley_ratio,
            "prominence_ratio": prominence_ratio,
            "right_tail_mass_after_first_valley": right_tail,
        },
        "statistics": {
            "gmm": {
                "weights_k2": list(weights_k2),
                "delta_bic_21": float(delta_bic_21),
                "ashmans_d_k2": float(ashman_d),
            }
        },
    }

    if weights_k3 is not None:
        payload["statistics"]["gmm"]["weights_k3"] = list(weights_k3)
    if delta_bic_32 is not None:
        payload["statistics"]["gmm"]["delta_bic_32"] = float(delta_bic_32)

    return payload


def test_apply_peak_caps_allows_three_when_supported():
    payload = _feature_payload(
        weights_k2=[0.5, 0.5],
        delta_bic_21=-14.0,
        ashman_d=2.8,
        valley_ratio=0.55,
        prominence_ratio=0.4,
        right_tail=0.2,
        peak_positions=[0.0, 1.0, 2.0],
        peak_width=0.25,
        weights_k3=[0.34, 0.33, 0.33],
        delta_bic_32=-9.5,
    )

    safe_max, heuristics = _apply_peak_caps(payload, "CD19", 3)

    assert safe_max == 3
    assert heuristics.get("forced_peak_cap") is None
    assert heuristics["evidence_for_three"] is True


def test_apply_peak_caps_caps_to_two_without_three_support():
    payload = _feature_payload(
        weights_k2=[0.46, 0.54],
        delta_bic_21=-12.0,
        ashman_d=2.6,
        valley_ratio=0.6,
        prominence_ratio=0.38,
        right_tail=0.18,
        peak_positions=[0.0, 1.2, 2.4],
        peak_width=0.3,
        weights_k3=[0.34, 0.33, 0.33],
        delta_bic_32=-4.0,
    )

    safe_max, heuristics = _apply_peak_caps(payload, "CD19", 3)

    assert safe_max == 2
    assert heuristics.get("forced_peak_cap") == 2
    assert heuristics["evidence_for_three"] is False


def test_apply_peak_caps_caps_to_one_without_two_support():
    payload = _feature_payload(
        weights_k2=[0.12, 0.88],
        delta_bic_21=-5.0,
        ashman_d=1.5,
        valley_ratio=0.9,
        prominence_ratio=0.1,
        right_tail=0.05,
        peak_positions=[0.0, 0.6],
        peak_width=0.5,
    )

    safe_max, heuristics = _apply_peak_caps(payload, "CD19", 3)

    assert safe_max == 1
    assert heuristics.get("forced_peak_cap") == 1
    assert heuristics["evidence_for_two"] is False


def test_default_priors_reflect_user_cap():
    priors = _default_priors("CD19", 4)

    assert priors["global_max"] == 4
    assert priors["typical_peaks"]["CD19"] == [1, 4]
