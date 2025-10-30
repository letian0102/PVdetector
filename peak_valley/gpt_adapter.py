from __future__ import annotations

import json
import re
import textwrap
from collections import Counter
from typing import Any, Optional

import numpy as np
from openai import AuthenticationError, OpenAI
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

from .kde_detector import kde_peaks_valleys, quick_peak_estimate
from .signature import shape_signature

__all__ = ["ask_gpt_peak_count", "ask_gpt_prominence", "ask_gpt_bandwidth"]

# keep a simple run-time cache; survives a single Streamlit run
_cache: dict[tuple, float | int | str] = {}  # (tag, sig[, extra]) → value


def _prepare_values(counts_full: Optional[np.ndarray]) -> np.ndarray:
    """Return finite values only (empty array if none)."""

    if counts_full is None:
        return np.empty(0, dtype=float)

    x = np.asarray(counts_full, dtype=float)
    if x.size == 0:
        return np.empty(0, dtype=float)

    mask = np.isfinite(x)
    if not np.any(mask):
        return np.empty(0, dtype=float)

    return x[mask]


def _robust_limits(x: np.ndarray) -> tuple[float, float]:
    """Robust (0.5, 99.5) percentile range with fallbacks."""

    if x.size == 0:
        return 0.0, 1.0

    lo, hi = np.percentile(x, [0.5, 99.5])
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x)) if x.size else 0.0
        hi = float(np.max(x)) if x.size else lo + 1.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _smooth_histogram(counts: np.ndarray) -> np.ndarray:
    """Apply a short [1,4,6,4,1] smoothing kernel."""

    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(counts.astype(float), kernel, mode="same")


def _extract_peak_candidates(
    centers: np.ndarray,
    smoothed: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """Return candidate peak descriptors and valley heuristics."""

    if smoothed.size == 0:
        return [], None, None, None, None

    prom_thresh = np.max(smoothed) * 0.05 if np.max(smoothed) > 0 else 0.0
    idx, _ = find_peaks(smoothed, prominence=prom_thresh, width=1)
    if idx.size == 0:
        return [], None, None, None, None

    prominences, left_bases, right_bases = peak_prominences(smoothed, idx)
    widths_samples = peak_widths(smoothed, idx, rel_height=0.5)[0]

    # convert widths to the same units as the histogram centres
    if centers.size > 1:
        step = float(centers[1] - centers[0])
    else:
        step = 1.0
    widths = widths_samples * step

    peaks: list[dict[str, Any]] = []
    for order, (i, prom, width, lb, rb) in enumerate(
        zip(idx, prominences, widths, left_bases, right_bases)
    ):
        seg = smoothed[int(lb) : int(rb) + 1]
        valley_depth = float(seg.min()) if seg.size else None
        peaks.append(
            {
                "x": float(centers[i]),
                "height": float(smoothed[i]),
                "prominence": float(prom),
                "width": float(width),
                "valley_depth": float(valley_depth) if valley_depth is not None else None,
                "index": int(i),
                "order": order,
            }
        )

    # heuristics for first valley/right-tail mass/ratios
    first_valley_x: Optional[float] = None
    valley_depth_ratio: Optional[float] = None
    prominence_ratio: Optional[float] = None
    valley_depth_abs: Optional[float] = None
    if len(idx) >= 2:
        left, right = idx[0], idx[1]
        seg = smoothed[left : right + 1]
        if seg.size:
            rel = int(np.argmin(seg))
            valley_idx = left + rel
            first_valley_x = float(centers[valley_idx])
            valley_depth_abs = float(smoothed[valley_idx])
            denom = min(smoothed[left], smoothed[right])
            if denom > 0:
                valley_depth_ratio = float(valley_depth_abs / denom)
            main_prom = prominences[0]
            sec_prom = prominences[1]
            if main_prom > 0:
                prominence_ratio = float(sec_prom / main_prom)

    return peaks, first_valley_x, valley_depth_ratio, prominence_ratio, valley_depth_abs


def _gmm_statistics(x: np.ndarray, max_components: int = 3) -> dict[str, Any]:
    """Fit Gaussian mixtures (k=1..max_components) and report metrics."""

    stats: dict[str, Any] = {
        "bic": {},
        "weights_k2": None,
        "means_k2": None,
        "stds_k2": None,
        "ashmans_d_k2": None,
        "weights_k3": None,
        "means_k3": None,
        "stds_k3": None,
    }

    if x.size == 0:
        return stats

    data = x.reshape(-1, 1)
    gmms: dict[int, GaussianMixture] = {}
    for k in range(1, max_components + 1):
        try:
            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=0,
                # ``n_init='auto'`` is only available on recent scikit-learn
                # releases.  Older versions (such as the one bundled with the
                # Streamlit app) expect an integer, so fall back to a small
                # explicit count to preserve compatibility across
                # environments.
                n_init=3,
                reg_covar=1e-6,
            )
            gm.fit(data)
            gmms[k] = gm
            bic_val = gm.bic(data)
            stats["bic"][f"k{k}"] = float(bic_val) if np.isfinite(bic_val) else None
        except Exception:
            stats["bic"][f"k{k}"] = None

    if 2 in gmms:
        gm2 = gmms[2]
        means = gm2.means_.ravel()
        stds = np.sqrt(np.clip(gm2.covariances_.reshape(-1), a_min=0.0, a_max=None))
        order = np.argsort(means)
        means = means[order]
        stds = stds[order]
        weights = gm2.weights_[order]
        stats["weights_k2"] = [float(w) for w in weights]
        stats["means_k2"] = [float(m) for m in means]
        stats["stds_k2"] = [float(s) for s in stds]
        denom = float(np.sqrt(stds[0] ** 2 + stds[1] ** 2))
        if denom > 0:
            stats["ashmans_d_k2"] = float(np.sqrt(2.0) * abs(means[1] - means[0]) / denom)

    if 3 in gmms:
        gm3 = gmms[3]
        means = gm3.means_.ravel()
        stds = np.sqrt(np.clip(gm3.covariances_.reshape(-1), a_min=0.0, a_max=None))
        order = np.argsort(means)
        stats["means_k3"] = [float(m) for m in means[order]]
        stats["stds_k3"] = [float(s) for s in stds[order]]
        stats["weights_k3"] = [float(w) for w in gm3.weights_[order]]

    if 1 in stats["bic"] and 2 in stats["bic"]:
        b1 = stats["bic"].get("k1")
        b2 = stats["bic"].get("k2")
        if b1 is not None and b2 is not None:
            stats["delta_bic_21"] = float(b2 - b1)
    if 2 in stats["bic"] and 3 in stats["bic"]:
        b2 = stats["bic"].get("k2")
        b3 = stats["bic"].get("k3")
        if b2 is not None and b3 is not None:
            stats["delta_bic_32"] = float(b3 - b2)

    return stats


def _right_tail_mass(x: np.ndarray, cutoff: Optional[float]) -> Optional[float]:
    if cutoff is None or x.size == 0:
        return None
    denom = float(np.sum(np.isfinite(x)))
    if denom <= 0:
        return None
    return float(np.sum(x > cutoff) / denom)


def _strong_two_peak_signal(
    features: dict[str, Any]
) -> tuple[bool, list[str], Optional[float], dict[str, Any]]:
    """Return whether evidence supports a second peak and explain why."""

    candidates = features.get("candidates") or {}
    peaks = candidates.get("peaks") or []
    valley_ratio = candidates.get("valley_depth_ratio")
    prominence_ratio = candidates.get("prominence_ratio")
    right_tail = candidates.get("right_tail_mass_after_first_valley")

    stats = features.get("statistics") or {}
    gmm = stats.get("gmm") if isinstance(stats, dict) else {}
    weights_k2 = gmm.get("weights_k2") if isinstance(gmm, dict) else None
    min_weight = min(weights_k2) if weights_k2 else None
    delta_bic = gmm.get("delta_bic_21")
    ashman = gmm.get("ashmans_d_k2")

    WEIGHT_THRESHOLD = 0.14
    DELTA_BIC_THRESHOLD = -9.5
    STRONG_DELTA_BIC_THRESHOLD = -13.5
    ASHMAN_THRESHOLD = 2.05
    STRONG_ASHMAN_THRESHOLD = 2.55
    VALLEY_RATIO_THRESHOLD = 0.78
    STRONG_VALLEY_RATIO_THRESHOLD = 0.62
    RIGHT_TAIL_THRESHOLD = 0.11
    STRONG_RIGHT_TAIL_THRESHOLD = 0.18
    PROMINENCE_RATIO_THRESHOLD = 0.24
    STRONG_PROMINENCE_RATIO_THRESHOLD = 0.36
    SEPARATION_RATIO_THRESHOLD = 1.5

    hits: list[str] = []
    has_weight_support = min_weight is not None and min_weight >= WEIGHT_THRESHOLD

    separation_info: dict[str, Any] = {
        "separation": None,
        "separation_ratio": None,
        "separation_ok": None,
    }

    if not has_weight_support:
        return False, ["low_component_weight"], min_weight, separation_info

    if len(peaks) < 2:
        return False, ["insufficient_candidates"], min_weight, separation_info

    x0 = float(peaks[0].get("x", 0.0))
    x1 = float(peaks[1].get("x", 0.0))
    separation = abs(x1 - x0)
    w0 = float(peaks[0].get("width") or 0.0)
    w1 = float(peaks[1].get("width") or 0.0)
    width_scale = max(0.5 * (w0 + w1), 1e-9)
    separation_ratio = separation / width_scale if width_scale > 0 else None
    separation_ok = (
        separation_ratio is not None and separation_ratio >= SEPARATION_RATIO_THRESHOLD
    )
    separation_info.update(
        {
            "separation": separation,
            "separation_ratio": separation_ratio,
            "separation_ok": separation_ok,
        }
    )

    if not separation_ok:
        return False, ["insufficient_separation"], min_weight, separation_info

    stat_votes: list[str] = []
    strong_stat_votes: list[str] = []

    has_delta = False
    delta_strong = False
    if delta_bic is not None and delta_bic <= DELTA_BIC_THRESHOLD:
        has_delta = True
        stat_votes.append("delta_bic")
        if delta_bic <= STRONG_DELTA_BIC_THRESHOLD:
            delta_strong = True
            strong_stat_votes.append("delta_bic")

    has_ashman = False
    ashman_strong = False
    if ashman is not None and ashman >= ASHMAN_THRESHOLD:
        has_ashman = True
        stat_votes.append("ashman_d")
        if ashman >= STRONG_ASHMAN_THRESHOLD:
            ashman_strong = True
            strong_stat_votes.append("ashman_d")

    stat_count = len(stat_votes)
    strong_stat = bool(strong_stat_votes)

    if stat_count == 0:
        return False, ["insufficient_statistical_support"], min_weight, separation_info

    geometry_votes: list[str] = []
    primary_geo_votes = 0
    strong_geo_votes = 0

    valley_vote = False
    if valley_ratio is not None and valley_ratio <= VALLEY_RATIO_THRESHOLD:
        valley_vote = True
        geometry_votes.append("valley_depth")
        primary_geo_votes += 1
        if valley_ratio <= STRONG_VALLEY_RATIO_THRESHOLD:
            strong_geo_votes += 1

    right_tail_vote = False
    if right_tail is not None and right_tail >= RIGHT_TAIL_THRESHOLD:
        right_tail_vote = True
        geometry_votes.append("right_tail")
        primary_geo_votes += 1
        if right_tail >= STRONG_RIGHT_TAIL_THRESHOLD:
            strong_geo_votes += 1

    prominence_vote = False
    if prominence_ratio is not None and prominence_ratio >= PROMINENCE_RATIO_THRESHOLD:
        prominence_vote = True
        geometry_votes.append("prominence_ratio")
        if prominence_ratio >= STRONG_PROMINENCE_RATIO_THRESHOLD:
            strong_geo_votes += 1

    geometry_vote_count = len(geometry_votes)

    # Require at least one geometric indicator beyond separation
    if geometry_vote_count == 0:
        return False, ["insufficient_geometric_support"], min_weight, separation_info

    allow_two = False
    if stat_count >= 2:
        if primary_geo_votes >= 1 or strong_stat or strong_geo_votes >= 1:
            allow_two = True
    else:  # exactly one statistical vote
        if primary_geo_votes >= 1 and geometry_vote_count >= 1:
            allow_two = True
        elif strong_stat and geometry_vote_count >= 2:
            allow_two = True
        elif strong_geo_votes >= 1 and geometry_vote_count >= 2:
            allow_two = True

    if not allow_two:
        return False, [], min_weight, separation_info

    if has_delta:
        hits.append("delta_bic")
        if delta_strong:
            hits.append("delta_bic_strong")
    if has_ashman:
        hits.append("ashman_d")
        if ashman_strong:
            hits.append("ashman_d_strong")
    if valley_vote:
        hits.append("valley_depth")
        if valley_ratio is not None and valley_ratio <= STRONG_VALLEY_RATIO_THRESHOLD:
            hits.append("valley_depth_strong")
    if right_tail_vote:
        hits.append("right_tail")
        if right_tail is not None and right_tail >= STRONG_RIGHT_TAIL_THRESHOLD:
            hits.append("right_tail_strong")
    if prominence_vote:
        hits.append("prominence_ratio")
        if prominence_ratio is not None and prominence_ratio >= STRONG_PROMINENCE_RATIO_THRESHOLD:
            hits.append("prominence_ratio_strong")
    hits.append("separation")

    return True, hits, min_weight, separation_info


def _strong_three_peak_signal(features: dict[str, Any]) -> tuple[bool, list[str]]:
    stats = features.get("statistics") or {}
    gmm = stats.get("gmm") if isinstance(stats, dict) else {}
    delta_bic = gmm.get("delta_bic_32")
    weights_k3 = gmm.get("weights_k3")

    hits: list[str] = []

    if delta_bic is not None and delta_bic <= -8.0:
        if weights_k3 and min(weights_k3) >= 0.08:
            hits.append("delta_bic")

    return bool(hits), hits


def _apply_peak_caps(
    feature_payload: dict[str, Any],
    marker_name: Optional[str],
    requested_max: int,
) -> tuple[int, dict[str, Any]]:
    safe_max = max(1, int(requested_max))
    heuristics: dict[str, Any] = {"requested_max": safe_max}

    has_two, two_hits, min_weight, separation_info = _strong_two_peak_signal(feature_payload)
    heuristics.update(
        {
            "evidence_for_two": has_two,
            "support_two_signals": two_hits,
            "min_component_weight_k2": min_weight,
            "peak_separation": separation_info,
        }
    )

    if safe_max >= 2 and not has_two:
        safe_max = 1
        heuristics["forced_peak_cap"] = 1

    if safe_max >= 3:
        has_three, three_hits = _strong_three_peak_signal(feature_payload)
        heuristics["evidence_for_three"] = has_three
        heuristics["support_three_signals"] = three_hits
        if not has_three:
            safe_max = min(safe_max, 2)
            heuristics["forced_peak_cap"] = heuristics.get("forced_peak_cap", 2)
    else:
        heuristics["evidence_for_three"] = False
        heuristics["support_three_signals"] = []

    heuristics["final_allowed_max"] = safe_max
    return safe_max, heuristics


def _default_priors(marker_name: Optional[str]) -> dict[str, Any]:
    priors = {
        "typical_peaks": {
            "CD4": [1, 3],
            "CD45RA": [1, 3],
            "CD45RO": [1, 3],
        },
        "default_range": [1, 2],
    }
    if marker_name and marker_name.upper() in {"CD4", "CD45RA", "CD45RO"}:
        priors["note"] = "tri_modal_possible"
    return priors


def _coerce_bandwidth_metadata(bandwidth: float | str | None) -> float | str | None:
    """Return a JSON-friendly representation of the bandwidth parameter."""

    if isinstance(bandwidth, (int, float)):
        if np.isfinite(bandwidth) and bandwidth > 0:
            return float(bandwidth)
        return None
    if isinstance(bandwidth, str):
        return bandwidth
    return None


def _effective_bandwidth(values: np.ndarray, bandwidth: float | str | None) -> Optional[float]:
    """Compute the effective Gaussian KDE bandwidth in data units."""

    if values.size <= 1:
        return None

    try:
        kde = gaussian_kde(values, bw_method=bandwidth or "scott")
    except Exception:
        return None

    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std <= 0:
        return None

    eff = float(kde.factor * std)
    return eff if np.isfinite(eff) and eff > 0 else None


def _estimate_peak_masses(xs: np.ndarray, ys: np.ndarray, peaks_x: list[float]) -> list[float]:
    """Approximate relative mass carried by each KDE peak."""

    if xs.size == 0 or not peaks_x:
        return []

    peaks = np.array(sorted(float(p) for p in peaks_x), dtype=float)
    if peaks.size == 0:
        return []

    dx = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
    left_edge = float(xs[0] - 0.5 * dx)
    right_edge = float(xs[-1] + 0.5 * dx)

    boundaries = [left_edge]
    for left, right in zip(peaks[:-1], peaks[1:]):
        boundaries.append(float(0.5 * (left + right)))
    boundaries.append(right_edge)

    total_area = float(np.trapz(ys, xs))
    if not np.isfinite(total_area) or total_area <= 0:
        return [0.0 for _ in range(peaks.size)]

    masses: list[float] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        mask = (xs >= start) & (xs <= stop)
        if not np.any(mask):
            masses.append(0.0)
            continue
        area = float(np.trapz(ys[mask], xs[mask]))
        masses.append(float(max(area, 0.0) / total_area))

    return masses


def _kde_feature_summary(
    values: np.ndarray,
    bandwidth: float | str | None,
) -> dict[str, Any]:
    """Summarise KDE-derived structure for GPT and heuristics."""

    if values.size == 0:
        return {}

    bw_method = bandwidth if bandwidth is not None else "scott"
    prominences = [0.05, 0.035, 0.02]
    profile: dict[str, Any] = {
        "bandwidth_param": _coerce_bandwidth_metadata(bandwidth),
    }

    eff_bw = _effective_bandwidth(values, bw_method)
    if eff_bw is not None:
        profile["effective_bandwidth"] = eff_bw

    sweep: list[dict[str, Any]] = []
    density_samples: Optional[dict[str, list[float]]] = None

    for prom in prominences:
        try:
            peaks, valleys, xs, ys = kde_peaks_valleys(
                values,
                None,
                prominence=prom,
                bw=bw_method,
                min_width=None,
                grid_size=4096,
                min_x_sep=0.5,
            )
        except Exception:
            peaks, valleys, xs, ys = [], [], np.array([]), np.array([])

        entry: dict[str, Any] = {
            "prominence": float(prom),
            "peak_count": int(len(peaks)),
            "peaks": [float(p) for p in peaks],
            "valleys": [float(v) for v in valleys],
        }

        if xs.size:
            peak_indices = [int(np.argmin(np.abs(xs - p))) for p in peaks]
            entry["heights"] = [float(ys[i]) for i in peak_indices]
            valley_indices = [int(np.argmin(np.abs(xs - v))) for v in valleys]
            entry["valley_heights"] = [float(ys[i]) for i in valley_indices]
            entry["masses"] = [float(m) for m in _estimate_peak_masses(xs, ys, peaks)]
            if density_samples is None:
                step = max(1, xs.size // 64)
                density_samples = {
                    "x": [float(v) for v in xs[::step][:64]],
                    "y": [float(v) for v in ys[::step][:64]],
                }
        else:
            entry["heights"] = []
            entry["valley_heights"] = []
            entry["masses"] = []

        sweep.append(entry)

    quick_count, quick_stable = quick_peak_estimate(
        values,
        prominence=0.05,
        bw=bw_method,
        min_width=None,
        grid_size=512,
    )

    profile["prominence_sweep"] = sweep
    profile["quick"] = {"count": int(quick_count), "stable": bool(quick_stable)}

    if density_samples is not None:
        profile["density_samples"] = density_samples

    return profile


def _auto_peak_decision(
    features: dict[str, Any],
    requested_max: int,
) -> tuple[Optional[int], dict[str, Any]]:
    """Derive a deterministic peak-count decision from the evidence."""

    profile = features.get("kde_profile") or {}
    votes: list[dict[str, Any]] = []

    for entry in profile.get("prominence_sweep", []) or []:
        count = entry.get("peak_count")
        if not isinstance(count, int) or count < 1:
            continue
        vote = {
            "source": f"prominence_{entry.get('prominence'):.3f}",
            "count": int(count),
        }
        masses = entry.get("masses") or []
        if masses:
            vote["secondary_mass"] = float(max(masses[1:], default=0.0))
            vote["masses"] = [float(m) for m in masses]
        valley_heights = entry.get("valley_heights") or []
        if valley_heights:
            vote["valley_height"] = float(valley_heights[0])
        votes.append(vote)

    quick = profile.get("quick")
    if isinstance(quick, dict) and isinstance(quick.get("count"), int):
        votes.append(
            {
                "source": "quick",
                "count": int(quick["count"]),
                "stable": bool(quick.get("stable", False)),
            }
        )

    stats = (features.get("statistics") or {}).get("gmm") or {}
    delta21 = stats.get("delta_bic_21")
    if delta21 is not None and delta21 <= -9.5:
        vote = {"source": "gmm_delta21", "count": 2, "delta_bic": float(delta21)}
        weights = stats.get("weights_k2")
        if weights:
            vote["weights"] = [float(w) for w in weights]
        votes.append(vote)

    delta32 = stats.get("delta_bic_32")
    if delta32 is not None and delta32 <= -8.0:
        vote = {"source": "gmm_delta32", "count": 3, "delta_bic": float(delta32)}
        weights3 = stats.get("weights_k3")
        if weights3:
            vote["weights"] = [float(w) for w in weights3]
        votes.append(vote)

    summary: dict[str, Any] = {
        "votes": votes,
        "requested_max": int(requested_max),
    }

    counter = Counter(v["count"] for v in votes if isinstance(v.get("count"), int))
    summary["vote_counts"] = {int(k): int(v) for k, v in counter.items()}
    total_votes = int(sum(counter.values()))
    summary["total_votes"] = total_votes

    top_choice: Optional[tuple[int, int]] = None
    for count, freq in counter.most_common():
        if 1 <= count <= requested_max:
            top_choice = (int(count), int(freq))
            break

    if top_choice is None:
        return None, summary

    top_count, supporters = top_choice
    summary["top_vote"] = {"count": top_count, "supporters": supporters}

    if total_votes == 0:
        return None, summary

    confidence = supporters / total_votes
    summary["confidence_estimate"] = float(confidence)

    multi_masses = [
        float(v.get("secondary_mass", 0.0))
        for v in votes
        if isinstance(v.get("count"), int) and v["count"] >= 2
    ]
    max_multi_mass = max(multi_masses) if multi_masses else 0.0
    summary["max_multi_mass"] = float(max_multi_mass)

    quick_vote = next((v for v in votes if v["source"] == "quick"), None)
    gmm_multi = [v for v in votes if v["source"].startswith("gmm")]

    candidates = features.get("candidates") or {}
    valley_ratio = candidates.get("valley_depth_ratio")
    prominence_ratio = candidates.get("prominence_ratio")

    # --- unimodal shortcut -------------------------------------------------
    unimodal_majority = top_count == 1 and supporters >= total_votes - 1
    quick_unimodal = bool(
        quick_vote and quick_vote.get("count") == 1 and quick_vote.get("stable")
    )
    weak_multi_shape = (
        max_multi_mass < 0.08
        and (valley_ratio is None or valley_ratio > 0.82)
        and (prominence_ratio is None or prominence_ratio < 0.22)
        and not gmm_multi
    )

    if top_count == 1 and (unimodal_majority or quick_unimodal) and weak_multi_shape:
        summary["auto"] = {
            "count": 1,
            "confidence": float(min(1.0, confidence + 0.25)),
            "reason": "consistent unimodal evidence",
        }
        return 1, summary

    # --- multi-peak shortcut -----------------------------------------------
    if top_count >= 2:
        if supporters < 2:
            return None, summary

        strong_mass = max_multi_mass >= 0.12
        valley_support = valley_ratio is not None and valley_ratio <= 0.78
        prominence_support = prominence_ratio is not None and prominence_ratio >= 0.28
        quick_support = bool(quick_vote and quick_vote.get("count") == top_count)
        gmm_support = any(v["count"] == top_count for v in gmm_multi)

        if top_count == 3 and supporters < 3 and not gmm_support:
            return None, summary

        if (
            (strong_mass or valley_support or prominence_support or gmm_support)
            and (quick_support or supporters >= 3 or gmm_support)
        ):
            summary["auto"] = {
                "count": int(top_count),
                "confidence": float(min(1.0, confidence + 0.18)),
                "reason": "stable multi-peak consensus",
            }
            return top_count, summary

    return None, summary


def _build_feature_payload(
    counts_full: Optional[np.ndarray],
    bandwidth: float | str | None,
) -> dict[str, Any]:
    """Construct histogram + analytic features for GPT."""

    payload: dict[str, Any] = {}
    values = _prepare_values(counts_full)
    if values.size == 0:
        return payload

    lo, hi = _robust_limits(values)
    if hi <= lo:
        hi = lo + 1.0

    bins = 64
    clipped = np.clip(values, lo, hi)
    counts, edges = np.histogram(clipped, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = _smooth_histogram(counts)

    peaks, valley_x, valley_depth_ratio, prominence_ratio, valley_depth_abs = _extract_peak_candidates(
        centers, smoothed
    )
    peaks_out = []
    for p in peaks[:3]:
        entry = {
            "x": p["x"],
            "height": float(p["height"]),
            "prominence": float(p["prominence"]),
            "width": float(p["width"]),
            "valley_depth": (float(p["valley_depth"]) if p["valley_depth"] is not None else None),
        }
        peaks_out.append(entry)

    payload["histogram"] = {
        "bin_edges": [float(e) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
    }

    payload["candidates"] = {
        "peaks": peaks_out,
        "right_tail_mass_after_first_valley": _right_tail_mass(values, valley_x),
        "valley_depth_ratio": valley_depth_ratio,
        "valley_depth_abs": valley_depth_abs,
        "prominence_ratio": prominence_ratio,
    }

    gmm_stats = _gmm_statistics(values)
    payload["statistics"] = {
        "dip_test_p": None,
        "silverman_k1_vs_k2_p": None,
        "gmm": gmm_stats,
    }

    kde_profile = _kde_feature_summary(values, bandwidth)
    if kde_profile:
        payload["kde_profile"] = kde_profile

    return payload

def ask_gpt_bandwidth(
    client: OpenAI,
    model_name: str,
    counts_full: np.ndarray,
    peak_amount: int,
    default: float | str = "scott",
) -> float | str:
    """
    Return a KDE bandwidth *scale factor* in **[0.10‥0.50]**.

    The function scans a few candidate scale factors, estimates how many
    peaks each reveals, and asks GPT to pick the value that should yield
    roughly ``peak_amount`` peaks.  The result is memo‑cached by the
    (distribution signature, expected peak count) pair.
    """

    if client is None:
        return default

    sig = shape_signature(counts_full)
    key = ("bw", sig, peak_amount)
    if key in _cache:
        return _cache[key]

    # down-sample for speed and avoid huge prompts
    x = counts_full.astype("float64")
    if x.size > 2000:
        x = np.random.choice(x, 2000, replace=False)

    # evaluate a small grid of candidate bandwidth scale factors
    scales = np.round(np.linspace(0.10, 0.50, 9), 2)
    peak_counts: list[int] = []
    for s in scales:
        try:
            n, _ = quick_peak_estimate(x, prominence=0.05, bw=s,
                                       min_width=None, grid_size=256)
        except Exception:
            n = 0
        peak_counts.append(int(n))

    q = np.percentile(x, [5, 25, 50, 75, 95]).round(2).tolist()
    table = ", ".join(f"{s:.2f}\u2192{p}" for s, p in zip(scales, peak_counts))
    prompt = textwrap.dedent(f"""
        We are tuning the scale factor for a 1-D Gaussian KDE bandwidth.
        Data summary: n={x.size}, p5={q[0]}, p25={q[1]}, median={q[2]},
        p75={q[3]}, p95={q[4]}.

        Candidate scale factors and their estimated peak counts:
        {table}

        Choose a scale factor between 0.10 and 0.50 so that about
        {peak_amount} peaks appear.  Reply with only the number using
        two decimals.
    """).strip()

    try:
        rsp = client.chat.completions.create(
            model=model_name,
            seed=2025,
            timeout=45,
            messages=[{"role": "user", "content": prompt}],
        )
        val = float(re.findall(r"\d*\.?\d+", rsp.choices[0].message.content)[0])
        val = float(np.clip(val, 0.10, 0.50))
    except AuthenticationError:
        raise
    except Exception:
        val = default

    _cache[key] = val
    return val

def ask_gpt_prominence(
    client:     OpenAI,
    model_name: str,
    counts_full: np.ndarray,
    default:    float = 0.05,
) -> float:
    """
    Return a KDE-prominence value in **[0.01 … 0.30]**.
    Result is memo-cached by the distribution *shape signature* so we
    never query GPT twice for the same-looking histogram.
    """
    if client is None:
        return default

    sig = shape_signature(counts_full)
    key = ("prom", sig)
    if key in _cache:                       # ← cached
        return _cache[key]

    # small numeric summary = prompt token-friendly
    q = np.percentile(counts_full, [5,25,50,75,95]).round(2).tolist()
    prompt = (
        "For a 1-D numeric distribution summarised as "
        f"p5={q[0]}, p25={q[1]}, median={q[2]}, p75={q[3]}, p95={q[4]}, "
        "suggest a *prominence* (between 0.01 and 0.30) that would let "
        "a KDE peak-finder isolate the visible modes.  "
        "Reply with one number only."
    )
    try:
        rsp = client.chat.completions.create(
            model=model_name, seed=2025,
            messages=[{"role": "user", "content": prompt}],
        )
        val = float(re.findall(r"\d*\.?\d+", rsp.choices[0].message.content)[0])
        val = float(np.clip(val, 0.01, 0.30))       # clamp to safe range
    except AuthenticationError:
        raise
    except Exception:
        val = default                                # fallback

    _cache[key] = val                                # memoise
    return val

def ask_gpt_peak_count(
    client: OpenAI,
    model_name: str,
    max_peaks: int,
    counts_full: np.ndarray | None = None,
    marker_name: str | None = None,
    *,
    technology: str | None = None,
    transform: str | None = None,
    batch_id: str | None = None,
    features: Optional[dict[str, Any]] = None,
    priors: Optional[dict[str, Any]] = None,
    bandwidth: float | str | None = None,
) -> int | None:
    """Query GPT for the number of visible density peaks using structured output."""

    if client is None:
        return max_peaks

    requested_max = max(1, int(max_peaks))
    values = _prepare_values(counts_full)

    payload: dict[str, Any] = {
        "meta": {
            "marker": marker_name or "unknown",
            "technology": technology or "unknown",
            "transform": transform or "arcsinh(cofactor=5)",
            "allowed_peaks_max": requested_max,
        }
    }

    bw_meta = _coerce_bandwidth_metadata(bandwidth)
    if bw_meta is not None:
        payload["meta"]["bandwidth_param"] = bw_meta

    if batch_id:
        payload["meta"]["batch_id"] = batch_id

    if values.size:
        q = np.percentile(values, [5, 25, 50, 75, 95])
        payload["meta"]["summary"] = {
            "n": int(values.size),
            "p5": float(q[0]),
            "p25": float(q[1]),
            "median": float(q[2]),
            "p75": float(q[3]),
            "p95": float(q[4]),
        }

    if features is not None:
        feature_payload = dict(features)
    else:
        feature_payload = _build_feature_payload(counts_full, bandwidth)

    auto_decision, analysis_info = _auto_peak_decision(feature_payload, requested_max)

    safe_max, heuristic_info = _apply_peak_caps(feature_payload, marker_name, requested_max)
    payload["meta"]["allowed_peaks_max"] = safe_max

    payload.update(feature_payload)

    heuristics_info = heuristic_info
    heuristics_info["auto_decision"] = analysis_info

    if auto_decision is not None:
        final_decision = int(max(1, min(safe_max, auto_decision)))
        heuristics_info["auto_decision"]["decision"] = final_decision
        auto_meta = analysis_info.get("auto") if isinstance(analysis_info, dict) else None
        if isinstance(auto_meta, dict):
            if final_decision != auto_decision:
                auto_meta["raw_count"] = auto_meta.get("count", auto_decision)
            auto_meta["count"] = final_decision
        if final_decision != auto_decision:
            heuristics_info["auto_decision"]["clamped_to_cap"] = auto_decision
        payload["analysis"] = analysis_info
        payload["heuristics"] = heuristics_info
        return final_decision

    payload["analysis"] = analysis_info

    kde_profile = feature_payload.get("kde_profile") or {}
    eff_bw = kde_profile.get("effective_bandwidth")
    if eff_bw is not None:
        payload["meta"]["effective_bandwidth"] = float(eff_bw)

    payload["priors"] = (
        priors.copy() if isinstance(priors, dict) else _default_priors(marker_name)
    )
    payload["heuristics"] = heuristics_info

    system = (
        "You are a cytometry/ADT gating assistant. Infer the number of visible density peaks "
        "using only the provided summaries, KDE profile, and consensus votes. Prefer fewer peaks "
        "unless strong evidence shows additional modes, honour the allowed peak range, and ignore "
        "tiny shoulders that lack prominence, width, or mass support. Output only the JSON object "
        "described by the schema."
    )

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "peak_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "peak_count": {"type": "integer", "minimum": 1, "maximum": safe_max},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "maxLength": 240},
                    "peak_indices": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["peak_count", "confidence", "reason", "peak_indices"],
                "additionalProperties": False,
            },
        },
    }

    try:
        rsp = client.chat.completions.create(
            model=model_name,
            temperature=1,
            seed=2025,
            timeout=45,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format=response_format,
        )
        msg = rsp.choices[0].message
        content = msg.content
        if isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = content or ""
        data = json.loads(text)
        peak_count = int(data["peak_count"])
        if peak_count <= 0:
            return None
        return min(safe_max, peak_count)
    except AuthenticationError:
        raise
    except Exception as exc:
        print(f"GPT peak count query failed: {exc}")
        return safe_max
