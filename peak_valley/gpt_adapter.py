from __future__ import annotations
import re
from openai import OpenAI, OpenAIError
import numpy as np
from .signature import shape_signature

__all__ = ["ask_gpt_peak_count", "ask_gpt_prominence"]

# keep a simple run-time cache; survives a single Streamlit run
_cache: dict[tuple[str, str], float | int] = {}  # (tag, sig) → value

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
            model=model_name, seed=2025, timeout=60,
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
    counts_full: np.ndarray | None = None,   # NEW
) -> int | None:
    """As before but with a memoization layer."""
    if counts_full is not None:
        sig = shape_signature(counts_full)
        if sig in _cache:                    # seen → reuse
            return min(_cache[sig], max_peaks)

    # ---------- GPT call (unchanged) ----------
    prompt = (
        "How many distinct density peaks (modes) does this list show? "
        "Answer with a single integer.\n\n"
        f"{counts_full}"
    )
    try:
        rsp = client.chat.completions.create(
            model=model_name,
            seed=2025,
            timeout=60,
            messages=[{"role": "user", "content": prompt}],
        )
        n = int(re.findall(r"\d+", rsp.choices[0].message.content)[0])
        if counts_full is not None:
            _cache[sig] = n                  # store for future
        return min(max_peaks, n) if n > 0 else None
    except (OpenAIError, ValueError, IndexError):
        return None
