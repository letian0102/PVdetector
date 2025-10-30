from __future__ import annotations

import json
import re
import textwrap
from collections import Counter
from typing import Any, Optional, Sequence

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
    heuristics["evidence_for_two"] = has_two
    heuristics["support_two_signals"] = two_hits
    heuristics["min_component_weight_k2"] = min_weight
    heuristics["peak_separation"] = separation_info

    if safe_max >= 3:
        has_three, three_hits = _strong_three_peak_signal(feature_payload)
        heuristics["evidence_for_three"] = has_three
        heuristics["support_three_signals"] = three_hits
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
        }
    }
    return priors


def _round_series(values: Sequence[float], decimals: int = 4) -> list[float]:
    """Return a JSON-friendly rounded list."""

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    return [float(np.round(v, decimals)) for v in arr.tolist()]


def _normalise_bandwidth(bandwidth: Optional[float | str]) -> float | str:
    """Return a usable bandwidth value for downstream heuristics."""

    if bandwidth is None:
        return "scott"

    if isinstance(bandwidth, str):
        bw = bandwidth.strip().lower()
        if bw in {"", "none", "auto"}:
            return "scott"
        try:
            return float(bw)
        except ValueError:
            return bandwidth

    return float(bandwidth)


def _detector_snapshot(
    values: np.ndarray,
    bandwidth: float | str,
    *,
    prominence: float = 0.05,
    grid_size: int = 4096,
) -> dict[str, Any]:
    """Capture the detector's own view of peaks at the plotting bandwidth."""

    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {}

    try:
        peaks_x, valleys_x, xs, ys = kde_peaks_valleys(
            x,
            n_peaks=None,
            prominence=prominence,
            bw=bandwidth,
            min_width=None,
            grid_size=grid_size,
        )
    except Exception:
        return {}

    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if xs_arr.size == 0 or ys_arr.size == 0:
        return {}

    peak_indices: list[int] = []
    peak_heights: list[float] = []
    for px in peaks_x[:6]:
        idx = int(np.argmin(np.abs(xs_arr - px)))
        peak_indices.append(idx)
        peak_heights.append(float(np.round(ys_arr[idx], 6)))

    prominences: list[float] = []
    if peak_indices:
        try:
            prom_vals, _, _ = peak_prominences(ys_arr, np.asarray(peak_indices, dtype=int))
            prominences = [float(np.round(v, 6)) for v in prom_vals.tolist()]
        except Exception:
            prominences = []

    valley_heights: list[float] = []
    for vx in valleys_x[:6]:
        idx = int(np.argmin(np.abs(xs_arr - vx)))
        valley_heights.append(float(np.round(ys_arr[idx], 6)))

    step = float(np.round(xs_arr[1] - xs_arr[0], 4)) if xs_arr.size > 1 else None

    return {
        "bandwidth": str(bandwidth),
        "peak_count": int(len(peaks_x)),
        "peak_positions": [float(np.round(px, 4)) for px in peaks_x[:6]],
        "peak_heights": peak_heights,
        "peak_prominences": prominences,
        "valley_positions": [float(np.round(vx, 4)) for vx in valleys_x[:6]],
        "valley_heights": valley_heights,
        "grid_step": step,
    }


def _quick_estimate_votes(
    values: np.ndarray,
    bandwidth: float | str,
    *,
    prominences: Sequence[float] = (0.02, 0.04, 0.07),
    grid_size: int = 4096,
) -> list[dict[str, Any]]:
    """Run the lightweight KDE heuristic at several prominences."""

    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return []

    votes: list[dict[str, Any]] = []
    for prom in prominences:
        try:
            count, stable = quick_peak_estimate(
                x,
                prominence=float(prom),
                bw=bandwidth,
                min_width=None,
                grid_size=grid_size,
            )
            count_val: Optional[int] = int(count)
            stable_val = bool(stable)
        except Exception:
            count_val = None
            stable_val = False

        votes.append(
            {
                "source": "quick_estimate",
                "prominence": float(np.round(prom, 3)),
                "count": count_val,
                "stable": stable_val,
            }
        )

    return votes


def _evaluate_mass_support(mass_sources: dict[str, list[float]]) -> dict[str, Any]:
    """Summarise how much probability mass secondary peaks hold."""

    cleaned: dict[str, list[float]] = {}
    dominant_mass = 0.0
    secondary_masses: list[float] = []
    sources_with_secondary = 0

    for name, values in mass_sources.items():
        if not values:
            continue
        arr = [float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v) and v > 0]
        if not arr:
            continue
        arr_sorted = sorted(arr, reverse=True)
        cleaned[name] = [float(np.round(v, 6)) for v in arr_sorted]
        dominant_mass = max(dominant_mass, arr_sorted[0])
        if len(arr_sorted) >= 2:
            secondary = arr_sorted[1:]
            secondary_masses.extend(secondary)
            if max(secondary) >= 0.02:
                sources_with_secondary += 1

    max_secondary = max(secondary_masses) if secondary_masses else 0.0
    total_secondary = float(np.sum(secondary_masses)) if secondary_masses else 0.0
    median_secondary = float(np.median(secondary_masses)) if secondary_masses else 0.0

    supports = False
    MAX_THRESHOLD = 0.045
    TOTAL_THRESHOLD = 0.10
    MEDIAN_THRESHOLD = 0.035
    COMBO_MAX_THRESHOLD = 0.035
    COMBO_TOTAL_THRESHOLD = 0.07
    if secondary_masses:
        if max_secondary >= MAX_THRESHOLD or total_secondary >= TOTAL_THRESHOLD:
            supports = True
        elif sources_with_secondary >= 2 and median_secondary >= MEDIAN_THRESHOLD:
            supports = True
        elif (
            max_secondary >= COMBO_MAX_THRESHOLD
            and total_secondary >= COMBO_TOTAL_THRESHOLD
            and len(secondary_masses) >= 2
        ):
            supports = True

    summary = {
        "sources": cleaned,
        "dominant_mass": float(np.round(dominant_mass, 6)) if dominant_mass else 0.0,
        "max_secondary_mass": float(np.round(max_secondary, 6)) if secondary_masses else 0.0,
        "median_secondary_mass": float(np.round(median_secondary, 6)) if secondary_masses else 0.0,
        "total_secondary_mass": float(np.round(total_secondary, 6)) if secondary_masses else 0.0,
        "secondary_count": int(len(secondary_masses)),
        "sources_with_secondary": int(sources_with_secondary),
        "secondary_masses": _round_series(secondary_masses, decimals=6) if secondary_masses else [],
        "supports_multiple": supports,
    }

    return summary


def _aggregate_votes(
    hist_peaks: list[dict[str, Any]],
    kde_summary: Optional[dict[str, Any]],
    quick_votes: list[dict[str, Any]],
    detector_snapshot: dict[str, Any],
    multiscale_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Combine independent estimates into a consensus vote."""

    sources: list[dict[str, Any]] = []
    counts: list[int] = []
    mass_sources: dict[str, list[float]] = {}

    if hist_peaks:
        entry = {
            "source": "histogram_smooth",
            "count": int(len(hist_peaks)),
            "peak_positions": [float(p["x"]) for p in hist_peaks],
            "peak_prominences": [
                float(p["prominence"]) if p.get("prominence") is not None else None
                for p in hist_peaks
            ],
            "peak_widths": [
                float(p["width"]) if p.get("width") is not None else None
                for p in hist_peaks
            ],
        }
        sources.append(entry)
        counts.append(entry["count"])

    if kde_summary and isinstance(kde_summary, dict):
        peaks = kde_summary.get("peaks") or []
        if peaks:
            entry = {
                "source": "kde_grid",
                "count": int(len(peaks)),
                "peak_positions": [float(p.get("x", 0.0)) for p in peaks],
                "peak_prominences": [
                    float(p["prominence"]) if p.get("prominence") is not None else None
                    for p in peaks
                ],
                "peak_widths": [
                    float(p["width"]) if p.get("width") is not None else None
                    for p in peaks
                ],
            }
            masses = [
                float(p.get("mass_fraction", 0.0))
                for p in peaks
                if isinstance(p.get("mass_fraction"), (int, float)) and np.isfinite(p.get("mass_fraction"))
            ]
            if masses:
                entry["peak_masses"] = [float(np.round(m, 6)) for m in masses]
                mass_sources["kde_grid"] = masses
            sources.append(entry)
            counts.append(entry["count"])

    if detector_snapshot:
        count_val = detector_snapshot.get("peak_count")
        try:
            count_int = int(count_val) if count_val is not None else None
        except (TypeError, ValueError):
            count_int = None

        entry = {
            "source": "detector_bandwidth",
            "count": count_int,
            "peak_positions": detector_snapshot.get("peak_positions"),
            "peak_heights": detector_snapshot.get("peak_heights"),
            "peak_prominences": detector_snapshot.get("peak_prominences"),
            "valley_positions": detector_snapshot.get("valley_positions"),
            "valley_heights": detector_snapshot.get("valley_heights"),
        }
        sources.append(entry)
        if count_int is not None:
            counts.append(count_int)

    if multiscale_summary and isinstance(multiscale_summary, dict):
        scale_results = multiscale_summary.get("scale_results") or []
        ms_counts = [
            int(res["peak_count"])
            for res in scale_results
            if isinstance(res.get("peak_count"), int)
        ]
        ms_majority: Optional[int] = None
        if ms_counts:
            tally_ms = Counter(ms_counts)
            ordered_ms = sorted(tally_ms.items(), key=lambda kv: (-kv[1], kv[0]))
            ms_majority = int(ordered_ms[0][0])

        robust_peaks = multiscale_summary.get("robust_peaks") or []
        robust_count = int(len(robust_peaks)) if robust_peaks else None
        robust_masses: list[float] = []
        for rp in robust_peaks:
            mass_val = rp.get("median_mass_fraction")
            if isinstance(mass_val, (int, float)) and np.isfinite(mass_val):
                robust_masses.append(float(mass_val))
                continue
            alt = rp.get("max_mass_fraction")
            if isinstance(alt, (int, float)) and np.isfinite(alt):
                robust_masses.append(float(alt))
        if robust_masses:
            mass_sources["kde_multiscale"] = robust_masses

        entry_count = robust_count if robust_count is not None else ms_majority
        entry = {
            "source": "kde_multiscale",
            "count": entry_count,
            "majority_count": ms_majority,
            "stable_peak_count": robust_count,
            "scale_results": scale_results,
            "robust_peaks": robust_peaks,
            "scales_tested": multiscale_summary.get("scales_tested"),
        }
        sources.append(entry)
        if entry_count is not None:
            counts.append(int(entry_count))

    for vote in quick_votes:
        sources.append(vote)
        if isinstance(vote.get("count"), int):
            counts.append(int(vote["count"]))

    if not sources:
        return {}

    result: dict[str, Any] = {"sources": sources}

    if multiscale_summary and isinstance(multiscale_summary, dict):
        result["multiscale"] = multiscale_summary
        robust_peaks = multiscale_summary.get("robust_peaks") or []
        if robust_peaks:
            result["robust_peak_count"] = int(len(robust_peaks))
            result["robust_peaks"] = robust_peaks
            if "kde_multiscale" not in mass_sources:
                fallback_masses: list[float] = []
                for rp in robust_peaks:
                    mass_val = rp.get("median_mass_fraction")
                    if isinstance(mass_val, (int, float)) and np.isfinite(mass_val):
                        fallback_masses.append(float(mass_val))
                        continue
                    alt = rp.get("max_mass_fraction")
                    if isinstance(alt, (int, float)) and np.isfinite(alt):
                        fallback_masses.append(float(alt))
                if fallback_masses:
                    mass_sources["kde_multiscale"] = fallback_masses
        collapse_adjustment = multiscale_summary.get("collapse_adjustment")
        if isinstance(collapse_adjustment, dict):
            result["smoothing_adjustment"] = collapse_adjustment

    tally = Counter(counts)
    if tally:
        ordered = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
        best_count, best_votes = ordered[0]
        support = best_votes / max(len(counts), 1)
        result["consensus"] = {
            "preferred_count": int(best_count),
            "support_fraction": float(np.round(support, 3)),
            "vote_tally": [
                {"count": int(k), "votes": int(v)} for k, v in sorted(tally.items())
            ],
        }

        if "robust_peak_count" in result:
            result["recommended_peak_count"] = int(result["robust_peak_count"])
        else:
            result["recommended_peak_count"] = int(best_count)

    if "recommended_peak_count" not in result and "robust_peak_count" in result:
        result["recommended_peak_count"] = int(result["robust_peak_count"])

    collapse_adjustment = result.get("smoothing_adjustment")
    if isinstance(collapse_adjustment, dict) and collapse_adjustment.get("triggered"):
        preferred = collapse_adjustment.get("preferred_count")
        try:
            preferred_int = int(preferred) if preferred is not None else 1
        except (TypeError, ValueError):
            preferred_int = 1
        current = result.get("recommended_peak_count")
        if not isinstance(current, int) or preferred_int < current:
            result["recommended_peak_count"] = max(1, preferred_int)
            collapse_adjustment["applied"] = True

    mass_metrics = _evaluate_mass_support(mass_sources)
    result["mass_metrics"] = mass_metrics

    recommended = result.get("recommended_peak_count")
    if isinstance(recommended, int) and recommended > 1:
        supports_mass = bool(mass_metrics.get("supports_multiple"))
        if not supports_mass:
            max_secondary = float(mass_metrics.get("max_secondary_mass") or 0.0)
            total_secondary = float(mass_metrics.get("total_secondary_mass") or 0.0)
            secondary_count = int(mass_metrics.get("secondary_count") or 0)
            if secondary_count == 0 or (max_secondary < 0.03 and total_secondary < 0.06):
                adjustment = {
                    "triggered": True,
                    "reason": "insufficient_mass",
                    "previous_count": int(recommended),
                    "max_secondary_mass": mass_metrics.get("max_secondary_mass"),
                    "total_secondary_mass": mass_metrics.get("total_secondary_mass"),
                }
                result["mass_adjustment"] = adjustment
                result["recommended_peak_count"] = 1
            else:
                result["mass_adjustment"] = {"triggered": False, "suppressed": True}
        else:
            result["mass_adjustment"] = {"triggered": False}
    else:
        result["mass_adjustment"] = {"triggered": False}

    vote_sources = [
        entry
        for entry in sources
        if isinstance(entry.get("count"), int)
    ]
    if vote_sources:
        result["vote_sources"] = [entry.get("source") for entry in vote_sources]

    recommended = result.get("recommended_peak_count")
    if isinstance(recommended, int) and vote_sources:
        agreeing = [
            entry.get("source")
            for entry in vote_sources
            if int(entry.get("count", -1)) == recommended
        ]
        disagreeing = [
            entry.get("source")
            for entry in vote_sources
            if int(entry.get("count", -1)) != recommended
        ]
        total_votes = len(vote_sources)
        agreement_fraction = len(agreeing) / max(total_votes, 1)
        result["agreement"] = {
            "agreeing_sources": agreeing,
            "disagreeing_sources": disagreeing,
            "fraction": float(np.round(agreement_fraction, 3)),
            "total_sources": total_votes,
        }

    return result


def _summarize_kde(
    values: np.ndarray,
    lo: float,
    hi: float,
    bandwidth: Optional[float | str],
    *,
    grid_points: int = 144,
    detector_snapshot: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a compact KDE summary aligned with the plotting bandwidth."""

    if bandwidth in (None, "", "none"):
        return {}

    bw_param: Any
    if isinstance(bandwidth, str):
        bw_param = bandwidth
    elif np.isscalar(bandwidth):
        bw_param = float(bandwidth)
        if not np.isfinite(bw_param):
            return {}
    else:
        bw_param = None

    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {}

    # Match KDE heuristics used in the main detector: limit extremely large
    # samples to a manageable size for the expensive covariance step.
    if x.size > 10_000:
        x = np.random.choice(x, 10_000, replace=False)

    try:
        kde = gaussian_kde(x, bw_method=bandwidth)
    except Exception:
        # Report the requested bandwidth so GPT knows what we intended, even
        # if KDE fitting failed (e.g., degenerate sample).
        return {
            "bandwidth_param": bw_param if bw_param is not None else str(bandwidth),
            "bandwidth_value": None,
        }

    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    bw_value = float(kde.factor * std)

    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0:
        lo = float(np.min(x))
        hi = float(np.max(x)) if x.size else lo + 1.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1.0
        span = float(hi - lo)

    margin = 0.05 * span if span > 0 else 1.0
    grid_lo = lo - margin
    grid_hi = hi + margin
    if not np.isfinite(grid_lo) or not np.isfinite(grid_hi) or grid_hi <= grid_lo:
        grid_lo = lo
        grid_hi = hi if hi > lo else lo + 1.0

    xs = np.linspace(grid_lo, grid_hi, int(max(32, grid_points)))
    ys = np.asarray(kde(xs), dtype=float)
    ys[~np.isfinite(ys)] = 0.0

    peaks_info: list[dict[str, Any]] = []
    valleys_info: list[dict[str, Any]] = []
    density_stats: dict[str, Any] = {}
    density_percentiles: list[float] = []

    if ys.size and np.max(ys) > 0:
        max_density = float(np.max(ys))
        prom_thresh = 0.05 * max_density
        step = float(xs[1] - xs[0]) if xs.size > 1 else 1.0
        total_mass = float(np.trapz(ys, xs)) if xs.size > 1 else float(np.sum(ys))

        idx, _ = find_peaks(ys, prominence=prom_thresh, width=1)
        if idx.size:
            prominences, left_bases, right_bases = peak_prominences(ys, idx)
            widths_samples = peak_widths(ys, idx, rel_height=0.5)[0]
            order = np.argsort(ys[idx])[::-1]
            for pos in order[:3]:
                j = idx[pos]
                left = int(max(0, np.floor(left_bases[pos])))
                right = int(min(len(ys) - 1, np.ceil(right_bases[pos])))
                seg = ys[left : right + 1]
                valley_depth = float(seg.min()) if seg.size else None
                width_val = float(widths_samples[pos] * step)
                record = {
                    "x": float(np.round(xs[j], 4)),
                    "height": float(np.round(ys[j], 6)),
                    "prominence": float(np.round(prominences[pos], 6)),
                    "width": float(np.round(width_val, 4)),
                    "valley_depth": (
                        float(np.round(valley_depth, 6)) if valley_depth is not None else None
                    ),
                    "relative_height": float(np.round(ys[j] / max_density, 6))
                    if max_density > 0
                    else None,
                }
                if total_mass > 0 and right >= left:
                    baseline = 0.0
                    if valley_depth is not None and np.isfinite(valley_depth):
                        baseline = max(0.0, valley_depth)
                    base_level = float(
                        min(ys[left], ys[right], baseline)
                    ) if right >= left else 0.0
                    trimmed = np.maximum(seg - base_level, 0.0)
                    area = float(np.trapz(trimmed, xs[left : right + 1])) if trimmed.size else 0.0
                    record["mass_fraction"] = (
                        float(np.round(area / total_mass, 6)) if area > 0 else 0.0
                    )
                    record["mass_baseline"] = float(np.round(base_level, 6))
                peaks_info.append(record)

        valley_idx, _ = find_peaks(-ys, prominence=prom_thresh * 0.5, width=1)
        if valley_idx.size:
            valley_prom, _, _ = peak_prominences(-ys, valley_idx)
            widths_samples = None
            try:
                widths_samples = peak_widths(-ys, valley_idx, rel_height=0.5)[0]
            except Exception:
                widths_samples = None
            order_v = np.argsort(valley_prom)[::-1]
            for pos in order_v[:3]:
                j = valley_idx[pos]
                width_val = (
                    float(widths_samples[pos] * step)
                    if widths_samples is not None
                    else None
                )
                valleys_info.append(
                    {
                        "x": float(np.round(xs[j], 4)),
                        "depth": float(np.round(ys[j], 6)),
                        "prominence": float(np.round(valley_prom[pos], 6)),
                        "width": float(np.round(width_val, 4)) if width_val is not None else None,
                    }
                )

        density_stats = {
            "max": float(np.round(np.max(ys), 6)),
            "mean": float(np.round(np.mean(ys), 6)),
            "median": float(np.round(np.median(ys), 6)),
            "integral": float(np.round(np.trapz(ys, xs), 6)),
        }
        density_percentiles = _round_series(np.percentile(ys, [5, 25, 50, 75, 95]), 6)

    summary = {
        "bandwidth_param": bw_param if bw_param is not None else str(bandwidth),
        "bandwidth_value": float(bw_value) if np.isfinite(bw_value) else None,
        "grid": {
            "x": _round_series(xs, decimals=4),
            "density": _round_series(ys, decimals=6),
            "step": float(np.round(xs[1] - xs[0], 4)) if xs.size > 1 else None,
            "lo": float(np.round(grid_lo, 4)),
            "hi": float(np.round(grid_hi, 4)),
            "percentiles": density_percentiles,
        },
        "peaks": peaks_info,
        "valleys": valleys_info,
        "density_stats": density_stats,
    }

    if detector_snapshot:
        summary["detector_view"] = detector_snapshot

    return summary


def _multiscale_peak_profile(
    values: np.ndarray,
    bandwidth: Optional[float | str],
    lo: float,
    hi: float,
    *,
    scales: Sequence[float] = (0.55, 0.75, 1.0, 1.3, 1.7),
    grid_points: int = 192,
    base_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Probe multiple bandwidth scales and retain persistent peaks."""

    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {}

    try:
        base_kde = gaussian_kde(x, bw_method=bandwidth)
    except Exception:
        return {}

    base_factor = float(base_kde.factor)
    if not np.isfinite(base_factor) or base_factor <= 0:
        return {}

    std = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0:
        return {}

    margin = 0.05 * span if span > 0 else 1.0
    grid_lo = float(lo - margin)
    grid_hi = float(hi + margin)
    if base_summary and isinstance(base_summary, dict):
        grid = base_summary.get("grid") or {}
        grid_lo = float(grid.get("lo", grid_lo))
        grid_hi = float(grid.get("hi", grid_hi))

    if not (np.isfinite(grid_lo) and np.isfinite(grid_hi)) or grid_hi <= grid_lo:
        return {}

    xs = np.linspace(grid_lo, grid_hi, int(max(grid_points, 96)))
    base_step = float(xs[1] - xs[0]) if xs.size > 1 else span / max(grid_points, 1)

    scale_results: list[dict[str, Any]] = []
    all_peaks: list[dict[str, Any]] = []
    scales_used: list[float] = []

    for scale in scales:
        scale_factor = float(base_factor * float(scale))
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            continue

        try:
            kde = gaussian_kde(x, bw_method=scale_factor)
        except Exception:
            continue

        ys = np.asarray(kde(xs), dtype=float)
        if ys.size == 0:
            continue
        ys[~np.isfinite(ys)] = 0.0
        max_density = float(np.max(ys))
        if max_density <= 0:
            continue

        prom_thresh = max_density * 0.035
        idx, _ = find_peaks(ys, prominence=prom_thresh, width=1)
        prominences: np.ndarray
        widths_samples: np.ndarray
        left_bases: np.ndarray
        right_bases: np.ndarray
        if idx.size:
            prominences, left_bases, right_bases = peak_prominences(ys, idx)
            widths_samples = peak_widths(ys, idx, rel_height=0.5)[0]
        else:
            prominences = np.empty(0)
            left_bases = np.empty(0)
            right_bases = np.empty(0)
            widths_samples = np.empty(0)

        peaks_for_scale: list[dict[str, Any]] = []
        total_mass_scale = float(np.trapz(ys, xs)) if xs.size > 1 else float(np.sum(ys))
        for pos, j in enumerate(idx):
            left = int(max(0, np.floor(left_bases[pos])))
            right = int(min(len(ys) - 1, np.ceil(right_bases[pos])))
            seg = ys[left : right + 1]
            valley_depth = float(seg.min()) if seg.size else None
            width_val = float(widths_samples[pos] * base_step)
            area = 0.0
            peak_record = {
                "x": float(np.round(xs[j], 4)),
                "height": float(np.round(ys[j], 6)),
                "prominence": float(np.round(prominences[pos], 6)),
                "width": float(np.round(width_val, 4)),
                "valley_depth": float(np.round(valley_depth, 6)) if valley_depth is not None else None,
                "relative_height": float(np.round(ys[j] / max_density, 6)) if max_density > 0 else None,
            }
            if total_mass_scale > 0 and right >= left:
                baseline = 0.0
                if valley_depth is not None and np.isfinite(valley_depth):
                    baseline = max(0.0, valley_depth)
                base_level = float(min(ys[left], ys[right], baseline)) if right >= left else 0.0
                trimmed = np.maximum(seg - base_level, 0.0)
                area = float(np.trapz(trimmed, xs[left : right + 1])) if trimmed.size else 0.0
                peak_record["mass_fraction"] = (
                    float(np.round(area / total_mass_scale, 6)) if area > 0 else 0.0
                )
                peak_record["mass_baseline"] = float(np.round(base_level, 6))
            peaks_for_scale.append(peak_record)
            all_peaks.append(
                {
                    "scale": float(scale),
                    "x": float(xs[j]),
                    "prominence": float(prominences[pos]),
                    "width": float(width_val),
                    "height": float(ys[j]),
                    "mass_fraction": float(area / total_mass_scale) if total_mass_scale > 0 and area > 0 else 0.0,
                }
            )

        scale_results.append(
            {
                "scale": float(np.round(scale, 3)),
                "bandwidth_factor": float(np.round(kde.factor, 6)),
                "bandwidth_value": float(np.round(kde.factor * std, 6)),
                "peak_count": int(len(peaks_for_scale)),
                "peaks": peaks_for_scale,
                "max_density": float(np.round(max_density, 6)),
            }
        )
        scales_used.append(float(np.round(scale, 3)))

    if not scale_results:
        return {}

    tolerance = max(0.02 * span, base_step * 4.0)
    clusters: list[dict[str, Any]] = []

    for peak in sorted(all_peaks, key=lambda item: item["prominence"], reverse=True):
        inserted = False
        for cluster in clusters:
            if abs(peak["x"] - cluster["center"]) <= tolerance:
                cluster["peaks"].append(peak)
                xs_cluster = [p["x"] for p in cluster["peaks"]]
                cluster["center"] = float(np.mean(xs_cluster))
                cluster["max_prominence"] = float(
                    max(cluster["max_prominence"], peak["prominence"])
                )
                inserted = True
                break
        if not inserted:
            clusters.append(
                {
                    "center": float(peak["x"]),
                    "max_prominence": float(peak["prominence"]),
                    "peaks": [peak],
                }
            )

    robust_peaks: list[dict[str, Any]] = []
    total_scales = len(scales_used)
    collapse_info: Optional[dict[str, Any]] = None

    for cluster in clusters:
        scales_present = sorted({float(np.round(p["scale"], 3)) for p in cluster["peaks"]})
        if not scales_present:
            continue
        support_fraction = len(scales_present) / max(total_scales, 1)
        if support_fraction < 0.6:
            continue

        prominences = [float(p["prominence"]) for p in cluster["peaks"]]
        widths = [float(p["width"]) for p in cluster["peaks"] if np.isfinite(p["width"])]
        masses = [
            float(p.get("mass_fraction", 0.0))
            for p in cluster["peaks"]
            if isinstance(p.get("mass_fraction"), (int, float))
            and np.isfinite(p.get("mass_fraction"))
        ]

        robust_peaks.append(
            {
                "x": float(np.round(cluster["center"], 4)),
                "support_fraction": float(np.round(support_fraction, 3)),
                "scales": scales_present,
                "median_prominence": float(np.round(float(np.median(prominences)), 6))
                if prominences
                else None,
                "max_prominence": float(np.round(cluster["max_prominence"], 6)),
                "median_width": float(np.round(float(np.median(widths)), 4)) if widths else None,
                "median_mass_fraction": float(np.round(float(np.median(masses)), 6))
                if masses
                else None,
                "max_mass_fraction": float(np.round(float(np.max(masses)), 6)) if masses else None,
            }
        )

    if scale_results and robust_peaks:
        sorted_scales = sorted(scales_used)
        if sorted_scales:
            high_group = max(1, len(sorted_scales) // 3)
            high_span = max(2, high_group)
            high_scales = sorted_scales[-high_span:]
            high_threshold = high_scales[0]
            high_counts = [
                res
                for res in scale_results
                if isinstance(res.get("scale"), (int, float))
                and float(res["scale"]) >= high_threshold - 1e-6
            ]
            if high_counts and all(int(res.get("peak_count", 0)) <= 1 for res in high_counts):
                filtered_peaks: list[dict[str, Any]] = []
                dropped_peaks: list[dict[str, Any]] = []
                for rp in robust_peaks:
                    rp_scales = rp.get("scales") or []
                    if any(float(s) >= high_threshold - 1e-6 for s in rp_scales):
                        filtered_peaks.append(rp)
                    else:
                        dropped_peaks.append(rp)

                if filtered_peaks:
                    if dropped_peaks:
                        collapse_info = {
                            "triggered": True,
                            "reason": "high_bandwidth_single_peak",
                            "high_scale_min": float(np.round(high_threshold, 3)),
                            "high_scale_max": float(np.round(sorted_scales[-1], 3)),
                            "high_scale_counts": [
                                {
                                    "scale": float(res.get("scale", 0.0)),
                                    "peak_count": int(res.get("peak_count", 0)),
                                }
                                for res in high_counts
                            ],
                            "dropped_peaks": int(len(dropped_peaks)),
                            "preferred_count": 1,
                        }
                    robust_peaks = filtered_peaks
                elif dropped_peaks:
                    # If every robust peak was confined to small scales, retain the
                    # strongest peak but mark the adjustment so downstream logic
                    # can treat the distribution as effectively unimodal.
                    best_peak = max(
                        dropped_peaks,
                        key=lambda item: (
                            float(item.get("support_fraction") or 0.0),
                            float(item.get("median_mass_fraction") or 0.0),
                            float(item.get("max_prominence") or 0.0),
                        ),
                    )
                    robust_peaks = [best_peak]
                    collapse_info = {
                        "triggered": True,
                        "reason": "high_bandwidth_single_peak",
                        "high_scale_min": float(np.round(high_threshold, 3)),
                        "high_scale_max": float(np.round(sorted_scales[-1], 3)),
                        "high_scale_counts": [
                            {
                                "scale": float(res.get("scale", 0.0)),
                                "peak_count": int(res.get("peak_count", 0)),
                            }
                            for res in high_counts
                        ],
                        "dropped_peaks": int(len(dropped_peaks) - 1),
                        "forced_retain": True,
                        "preferred_count": 1,
                    }

    profile = {
        "scales_tested": scales_used,
        "scale_results": scale_results,
        "robust_peaks": robust_peaks,
        "stable_peak_count": int(len(robust_peaks)),
    }

    if std > 0 and np.isfinite(std):
        profile["base_bandwidth_value"] = float(np.round(base_factor * std, 6))

    if collapse_info:
        profile["collapse_adjustment"] = collapse_info

    return profile


def _auto_peak_decision(
    analysis: Optional[dict[str, Any]],
    allowed_max: int,
) -> tuple[Optional[int], dict[str, Any]]:
    """Decide whether detector consensus is strong enough to skip GPT."""

    info: dict[str, Any] = {"triggered": False}

    if not isinstance(analysis, dict) or not analysis:
        info["reason"] = "no_analysis"
        return None, info

    recommended = analysis.get("recommended_peak_count")
    info["recommended"] = recommended
    if not isinstance(recommended, int) or recommended <= 0:
        info["reason"] = "no_recommendation"
        return None, info

    if recommended > max(1, int(allowed_max)):
        info["reason"] = "over_cap"
        return None, info

    mass_metrics = analysis.get("mass_metrics") if isinstance(analysis, dict) else None
    if isinstance(recommended, int) and recommended > 1 and isinstance(mass_metrics, dict):
        supports_mass = mass_metrics.get("supports_multiple")
        if supports_mass is False:
            info["reason"] = "insufficient_mass"
            info["mass_metrics"] = mass_metrics
            return None, info

    sources = analysis.get("sources") or []
    vote_sources = [s for s in sources if isinstance(s.get("count"), int)]
    agree_sources = [
        s.get("source")
        for s in vote_sources
        if int(s.get("count", -1)) == recommended
    ]
    disagree_sources = [
        s.get("source")
        for s in vote_sources
        if int(s.get("count", -1)) != recommended
    ]
    agreement_fraction = len(agree_sources) / max(len(vote_sources), 1)

    info["agreeing_sources"] = agree_sources
    info["disagreeing_sources"] = disagree_sources
    info["agreement_fraction"] = float(np.round(agreement_fraction, 3))
    info["vote_source_count"] = len(vote_sources)

    consensus = analysis.get("consensus") or {}
    support = consensus.get("support_fraction")
    vote_tally = consensus.get("vote_tally") or []
    votes_for = 0
    total_votes = 0
    for entry in vote_tally:
        try:
            count_val = int(entry.get("count"))
            votes_val = int(entry.get("votes"))
        except (TypeError, ValueError):
            continue
        total_votes += max(votes_val, 0)
        if count_val == recommended:
            votes_for += max(votes_val, 0)

    info["consensus_support"] = support
    info["consensus_votes_for"] = votes_for
    info["consensus_votes_total"] = total_votes

    quick_votes = [s for s in sources if s.get("source") == "quick_estimate"]
    quick_total = len(quick_votes)
    quick_match = sum(1 for s in quick_votes if s.get("count") == recommended)
    quick_stable = sum(
        1 for s in quick_votes if s.get("count") == recommended and s.get("stable")
    )
    info["quick_votes_total"] = quick_total
    info["quick_votes_match"] = quick_match
    info["quick_votes_stable"] = quick_stable

    detector_vote = next(
        (s for s in sources if s.get("source") == "detector_bandwidth"),
        None,
    )
    histogram_vote = next(
        (s for s in sources if s.get("source") == "histogram_smooth"),
        None,
    )
    detector_agrees = bool(
        detector_vote and detector_vote.get("count") == recommended
    )
    histogram_agrees = bool(
        histogram_vote and histogram_vote.get("count") == recommended
    )
    info["detector_agrees"] = detector_agrees
    info["histogram_agrees"] = histogram_agrees

    robust_count = analysis.get("robust_peak_count")
    info["robust_peak_count"] = robust_count

    multiscale = analysis.get("multiscale") or {}
    ms_stable = multiscale.get("stable_peak_count")
    info["multiscale_stable_peak_count"] = ms_stable

    strong_consensus = support is not None and support >= 0.72
    overwhelming_consensus = support is not None and support >= 0.85
    robust_match = isinstance(robust_count, int) and robust_count == recommended and robust_count > 0
    multiscale_match = isinstance(ms_stable, int) and ms_stable == recommended and ms_stable > 0
    quick_confident = quick_stable >= 2 or (quick_total >= 2 and quick_match == quick_total)
    broad_agreement = agreement_fraction >= 0.8 and len(agree_sources) >= max(2, len(vote_sources) - 1)
    detector_histogram_pair = detector_agrees and histogram_agrees

    signals: list[str] = []
    if strong_consensus:
        signals.append("consensus_strong")
    if overwhelming_consensus:
        signals.append("consensus_overwhelming")
    if robust_match:
        signals.append("robust_match")
    if multiscale_match:
        signals.append("multiscale_match")
    if quick_confident:
        signals.append("quick_confident")
    if broad_agreement:
        signals.append("broad_agreement")
    if detector_histogram_pair:
        signals.append("detector_histogram_agree")

    info["signals"] = signals

    trigger = False

    if robust_match and (strong_consensus or multiscale_match or quick_confident):
        trigger = True
    elif multiscale_match and strong_consensus and agreement_fraction >= 0.7:
        trigger = True
    elif overwhelming_consensus and broad_agreement and (quick_confident or detector_histogram_pair):
        trigger = True
    elif recommended == 1 and agreement_fraction >= 0.9 and (support is None or support >= 0.75):
        if quick_match >= max(1, quick_total - 1) or detector_histogram_pair:
            trigger = True

    if trigger:
        info["triggered"] = True
        info["decision"] = int(recommended)
        return int(recommended), info

    info["reason"] = "insufficient_evidence"
    return None, info
def _build_feature_payload(
    counts_full: Optional[np.ndarray],
    *,
    kde_bandwidth: Optional[float | str] = None,
) -> dict[str, Any]:
    """Construct histogram + analytic features for GPT."""

    payload: dict[str, Any] = {}
    values = _prepare_values(counts_full)
    if values.size == 0:
        return payload

    lo, hi = _robust_limits(values)
    if hi <= lo:
        hi = lo + 1.0

    bins = 128
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
        "bin_centers": _round_series(centers, decimals=4),
        "counts": [int(c) for c in counts.tolist()],
        "smoothed": _round_series(smoothed, decimals=3),
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

    normalised_bw = _normalise_bandwidth(kde_bandwidth)
    detector_snapshot = _detector_snapshot(values, normalised_bw)
    quick_votes = _quick_estimate_votes(values, normalised_bw)

    kde_summary = _summarize_kde(
        values,
        lo,
        hi,
        normalised_bw,
        detector_snapshot=detector_snapshot,
    )
    if kde_summary:
        payload["kde"] = kde_summary

    multiscale_summary = _multiscale_peak_profile(
        values,
        normalised_bw,
        lo,
        hi,
        base_summary=kde_summary,
    )
    if multiscale_summary:
        payload["kde_multiscale"] = multiscale_summary

    analysis = _aggregate_votes(
        peaks_out,
        kde_summary,
        quick_votes,
        detector_snapshot,
        multiscale_summary,
    )
    if analysis:
        analysis["bandwidth_used"] = (
            normalised_bw if isinstance(normalised_bw, str) else float(normalised_bw)
        )
        if kde_summary and isinstance(kde_summary, dict):
            bw_val = kde_summary.get("bandwidth_value")
            if bw_val is not None:
                analysis["bandwidth_value"] = float(bw_val)
        if kde_bandwidth is not None:
            analysis["requested_bandwidth"] = kde_bandwidth
        if detector_snapshot:
            analysis.setdefault("detector_snapshot", detector_snapshot)
        payload["analysis"] = analysis


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
    kde_bandwidth: Optional[float | str] = None,
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
        feature_payload = _build_feature_payload(counts_full, kde_bandwidth=kde_bandwidth)

    safe_max, heuristic_info = _apply_peak_caps(feature_payload, marker_name, requested_max)

    analysis_payload = feature_payload.get("analysis")
    auto_count, auto_info = _auto_peak_decision(analysis_payload, safe_max)
    if isinstance(analysis_payload, dict):
        analysis_payload["auto_decision"] = auto_info
    heuristic_info["auto_peak_decision"] = auto_info

    payload["meta"]["allowed_peaks_max"] = safe_max

    payload.update(feature_payload)

    payload["priors"] = (priors.copy() if isinstance(priors, dict) else _default_priors(marker_name))
    payload["heuristics"] = heuristic_info

    if auto_count is not None:
        return int(min(safe_max, max(1, auto_count)))

    system = (
        "You are a cytometry/ADT gating assistant. Infer the number of visible density peaks "
        "(modes) using only the provided features. Prefer fewer peaks unless strong evidence "
        "suggests more, but never exceed the allowed peak cap. Tiny shoulders are not peaks unless "
        "prominence and width thresholds are met. Output only the JSON object described by the schema."
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
