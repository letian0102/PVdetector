from __future__ import annotations

import json
import math
import re
import textwrap
from typing import Any, Optional

import numpy as np
from openai import AuthenticationError, OpenAI
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

from .kde_detector import quick_peak_estimate
from .signature import shape_signature

__all__ = ["ask_gpt_peak_count", "ask_gpt_prominence", "ask_gpt_bandwidth"]

HISTOGRAM_PROMPT_CHAR_BUDGET = 4800
MIN_ADAPTIVE_SAMPLES = 512
MAX_HISTOGRAM_BINS = 160
MIN_HISTOGRAM_BINS = 80
BASELINE_BINS = 64
SUMMARY_BINS = 48

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


def _adaptive_bin_count(
    values: np.ndarray,
    lo: float,
    hi: float,
    *,
    baseline: int = BASELINE_BINS,
    min_bins: int = MIN_HISTOGRAM_BINS,
    max_bins: int = MAX_HISTOGRAM_BINS,
) -> tuple[int, dict[str, Any]]:
    """Return an adaptive bin count (Freedman–Diaconis/Scott) and metadata."""

    info: dict[str, Any] = {
        "samples": int(values.size),
        "method": "baseline",
        "raw_bins": None,
        "clipped_bins": None,
    }

    span = float(hi - lo)
    if values.size < MIN_ADAPTIVE_SAMPLES or not np.isfinite(span) or span <= 0:
        return baseline, info

    n = float(values.size)

    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    fd_bins = math.inf
    if iqr > 0:
        width_fd = 2.0 * iqr / np.cbrt(n)
        if width_fd > 0:
            fd_bins = span / width_fd

    std = float(np.std(values))
    scott_bins = math.inf
    if std > 0:
        width_scott = 3.5 * std / np.cbrt(n)
        if width_scott > 0:
            scott_bins = span / width_scott

    candidates: list[tuple[str, float]] = []
    if math.isfinite(fd_bins) and fd_bins > baseline:
        candidates.append(("freedman_diaconis", fd_bins))
    if math.isfinite(scott_bins) and scott_bins > baseline:
        candidates.append(("scott", scott_bins))

    if not candidates:
        return baseline, info

    method, raw_bins = min(candidates, key=lambda item: item[1])
    clipped = int(np.clip(round(raw_bins), min_bins, max_bins))
    if clipped <= baseline:
        return baseline, info

    info.update({
        "method": method,
        "raw_bins": float(raw_bins),
        "clipped_bins": clipped,
    })
    return clipped, info


def _run_length_encode(arr: np.ndarray) -> list[list[int]]:
    """Return simple run-length encoding for prompt-friendly payloads."""

    if arr.size == 0:
        return []

    encoded: list[list[int]] = []
    current_val = int(arr[0])
    run_length = 1
    for val in arr[1:]:
        intval = int(val)
        if intval == current_val:
            run_length += 1
        else:
            encoded.append([current_val, run_length])
            current_val = intval
            run_length = 1
    encoded.append([current_val, run_length])
    return encoded


def _estimate_prompt_chars(data: dict[str, Any]) -> int:
    """Estimate prompt character count for a JSON serialisation."""

    try:
        return len(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    except TypeError:
        return 0


def _downsample_histogram(
    values: np.ndarray,
    lo: float,
    hi: float,
    target_bins: int,
) -> dict[str, Any]:
    """Generate a compact histogram representation for prompts."""

    if target_bins <= 1 or values.size == 0:
        return {
            "bin_count": int(max(target_bins, 0)),
            "bin_edges": [],
            "counts": [],
        }

    counts, edges = np.histogram(values, bins=target_bins, range=(lo, hi))
    return {
        "bin_count": int(target_bins),
        "bin_edges": [float(round(e, 4)) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
    }


def _second_derivative_summary(smoothed: np.ndarray) -> dict[str, Any]:
    """Summarise curvature structure from a smoothed histogram."""

    metrics = {
        "zero_crossings": 0,
        "maxima": 0,
        "minima": 0,
        "mean_abs": 0.0,
        "max_abs": 0.0,
    }

    if smoothed.size < 5:
        return metrics

    first = np.diff(smoothed)
    second = np.diff(first)
    if second.size == 0:
        return metrics

    nonzero_mask = second != 0
    filtered = second[nonzero_mask]
    if filtered.size > 1:
        signs = np.sign(filtered)
        metrics["zero_crossings"] = int(np.sum(signs[1:] != signs[:-1]))

    # Local extrema within the second derivative sequence
    core = second[1:-1]
    if core.size > 0:
        left = second[:-2]
        right = second[2:]
        metrics["maxima"] = int(np.sum((core > left) & (core > right)))
        metrics["minima"] = int(np.sum((core < left) & (core < right)))

    abs_second = np.abs(second)
    metrics["mean_abs"] = float(abs_second.mean())
    metrics["max_abs"] = float(abs_second.max())
    return metrics


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

    if "k1" in stats["bic"] and "k2" in stats["bic"]:
        b1 = stats["bic"].get("k1")
        b2 = stats["bic"].get("k2")
        if b1 is not None and b2 is not None:
            stats["delta_bic_21"] = float(b2 - b1)
    if "k2" in stats["bic"] and "k3" in stats["bic"]:
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
    heuristics: dict[str, Any] = {}

    heuristics["requested_max"] = safe_max

    has_two, two_hits, min_weight, separation_info = _strong_two_peak_signal(feature_payload)
    heuristics["evidence_for_two"] = has_two
    heuristics["support_two_signals"] = two_hits
    heuristics["min_component_weight_k2"] = min_weight
    heuristics["peak_separation"] = separation_info

    if safe_max >= 2 and not has_two:
        safe_max = 1
        heuristics["forced_peak_cap"] = 1

    if safe_max >= 3:
        has_three, three_hits = _strong_three_peak_signal(feature_payload)
        heuristics["evidence_for_three"] = has_three
        heuristics["support_three_signals"] = three_hits
        if not has_three:
            safe_max = 2
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
    }
    if marker_name:
        priors["marker"] = marker_name.upper()
    return priors


def _build_feature_payload(
    counts_full: Optional[np.ndarray],
) -> dict[str, Any]:
    """Construct histogram + analytic features for GPT."""

    payload: dict[str, Any] = {}
    values = _prepare_values(counts_full)
    if values.size == 0:
        return payload

    lo, hi = _robust_limits(values)
    if hi <= lo:
        hi = lo + 1.0

    bins, adaptive_info = _adaptive_bin_count(values, lo, hi)
    if bins <= 0:
        bins = BASELINE_BINS
    clipped = np.clip(values, lo, hi)
    counts, edges = np.histogram(clipped, bins=bins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    smoothed = _smooth_histogram(counts)

    kde_section: dict[str, Any] | None = None
    kde_points = 200
    if values.size >= 2:
        try:
            kde = gaussian_kde(values, bw_method="scott")
            scale = float(kde.factor)
            sample_std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            bandwidth = scale * sample_std if sample_std > 0 else None
            xs = np.linspace(lo, hi, kde_points)
            ys = kde(xs)
            kde_section = {
                "bandwidth": float(bandwidth) if bandwidth is not None and np.isfinite(bandwidth) else None,
                "scale": scale if np.isfinite(scale) else None,
                "x": [float(v) for v in xs.tolist()],
                "density": [float(v) for v in ys.tolist()],
            }
        except Exception:
            kde_section = {
                "bandwidth": None,
                "scale": None,
                "x": [],
                "density": [],
            }
    else:
        kde_section = {
            "bandwidth": None,
            "scale": None,
            "x": [],
            "density": [],
        }

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

    histogram_payload: dict[str, Any] = {
        "bin_count": int(bins),
        "bin_edges": [float(round(e, 4)) for e in edges.tolist()],
        "counts": [int(c) for c in counts.tolist()],
        "kde_bandwidth": kde_section.get("bandwidth") if kde_section else None,
        "kde_scale": kde_section.get("scale") if kde_section else None,
    }

    if kde_section:
        payload["kde"] = kde_section

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
        feature_payload = _build_feature_payload(counts_full)

    safe_max, heuristic_info = _apply_peak_caps(feature_payload, marker_name, requested_max)
    payload["meta"]["allowed_peaks_max"] = safe_max

    payload.update(feature_payload)

    payload["priors"] = (priors.copy() if isinstance(priors, dict) else _default_priors(marker_name))
    payload["heuristics"] = heuristic_info

    system = (
        "You are a cytometry/ADT gating assistant. Infer the number of visible density peaks "
        "(modes) using only the provided features. Prefer fewer peaks unless strong evidence "
        "suggests more, while respecting the user-specified maximum. Histogram bins and KDE traces "
        "summarise the distribution; prefer the smoother KDE profile and its bandwidth hints when judging "
        "shoulders versus true peaks. Tiny shoulders are not peaks unless prominence and width thresholds "
        "are met. Output only the JSON object described by the schema."
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
