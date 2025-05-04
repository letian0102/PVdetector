from __future__ import annotations
import re
from typing import Optional
from openai import OpenAI, OpenAIError
# ... existing imports ...
from .signature import shape_signature      # NEW

__all__ = ["ask_gpt_peak_count"]

# keep a simple run-time cache; survives a single Streamlit run
_cache: dict[str, int] = {}

def ask_gpt_peak_count(
    client:     OpenAI,
    model_name: str,
    sample:     list[float],
    max_peaks:  int,
    counts_full: np.ndarray | None = None,   # NEW
) -> int | None:
    """As before but with a memoisation layer."""
    if counts_full is not None:
        sig = shape_signature(counts_full)
        if sig in _cache:                    # seen → reuse
            return min(_cache[sig], max_peaks)

    # ---------- GPT call (unchanged) ----------
    prompt = (
        "How many distinct density peaks (modes) does this list show? "
        "Answer with a single integer.\n\n"
        f"{sample}"
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
