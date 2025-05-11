from __future__ import annotations
import re, textwrap
from openai import OpenAI, OpenAIError
import numpy as np
from .signature import shape_signature

__all__ = ["ask_gpt_peak_count", "ask_gpt_prominence", "ask_gpt_bandwidth"]

# keep a simple run-time cache; survives a single Streamlit run
_cache: dict[tuple[str, str], float | int] = {}  # (tag, sig) → value

def ask_gpt_bandwidth(
    client:      OpenAI,
    model_name:  str,
    counts_full: np.ndarray,
    peak_amount: int,               # 🔸 NEW – expected modes
    default:     float = 0.5,
) -> float:
    """
    Return a KDE bandwidth *scale factor* in **[0.1 … 0.5]** that makes the
    KDE reveal *about* ``peak_amount`` peaks.
    The answer is memo-cached by the (signature, expected_peaks) pair.
    """
    sig = shape_signature(counts_full)
    key = ("bw", sig, peak_amount)
    if key in _cache:                      # reuse identical call
        return _cache[key]

    # ── construct a compact prompt ───────────────────────────────────────
    #  • Give GPT a concise numeric summary (5-number summary + length)
    #  • State the desired number of peaks explicitly
    prompt = textwrap.dedent(f"""
        You are tuning the bandwidth for a 1-D Gaussian KDE.
        The data is {counts_full}

        Choose a *scale factor* in the **0.1-0.5** range
        so that the KDE curve shows **≈ {peak_amount} distinct peaks**:
        ─ if the bandwidth is too small the curve will be noisy (too many peaks),
        ─ if it is too large it will merge peaks.

        Reply with just one decimal number, nothing else.
    """).strip()

    try:
        rsp  = client.chat.completions.create(
            model=model_name,
            seed=2025,
            timeout=45,
            messages=[{"role": "user", "content": prompt}],
        )
        val = float(re.findall(r"\d*\.?\d+", rsp.choices[0].message.content)[0])
        val = float(np.clip(val, 0.1, 0.7))      # final safety clamp
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
    sig = shape_signature(counts_full)
    key = ("prom", sig)
    if key in _cache:                       # ← cached ✔
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
    except Exception:
        val = default                                # fallback

    _cache[key] = val                                # memoise
    return val

def ask_gpt_peak_count(
    client:     OpenAI,
    model_name: str,
    max_peaks:  int,
    counts_full: np.ndarray | None = None,
    marker_name: str | None = None,
) -> int | None:
    """As before but with a memoization layer."""
    if counts_full is not None:
        sig = shape_signature(counts_full)
        if sig in _cache:                    # seen → reuse
            return min(_cache[sig], max_peaks)

    # ---------- GPT call (unchanged) ----------
    marker_txt = f"for the protein marker **{marker_name}** " if marker_name else ""
    prompt = (
        f"How many density peaks (modes) should be visible in the following raw protein-count list? Remember this is  {marker_txt} (Give ONE integer ≤ {max_peaks}.)\n\n"
        f"{counts_full}"
    )
    try:
        rsp = client.chat.completions.create(
            model=model_name,
            seed=2025,
            messages=[{"role": "user", "content": prompt}],
        )
        n = int(re.findall(r"\d+", rsp.choices[0].message.content)[0])
        if counts_full is not None:
            _cache[sig] = n                  # store for future
        return min(max_peaks, n) if n > 0 else None
    except (OpenAIError, ValueError, IndexError):
        return None
