from __future__ import annotations
import numpy as np
import hashlib

__all__ = ["shape_signature"]

def shape_signature(counts: np.ndarray) -> str:
    """
    Return a 10-byte hex digest that represents the *shape* of the distribution
    (mean-free, scale-free).

    • 16-bin histogram of z-scored data, normalised to sum-1  
    • 3 robust quantiles (q25,q50,q75)  
    • md5 of the concatenated float32 vector -> 10-byte digest
    """
    x = counts.astype("float32")
    if x.size > 2000:                         # subsample for speed
        x = np.random.choice(x, 2000, False)
    x = (x - x.mean()) / x.std(ddof=1)        # z-score
    hist, _ = np.histogram(x, bins=16, range=(-4, 4), density=True)
    qs = np.percentile(x, [25, 50, 75]) / 10  # keep <1 in magnitude
    vec = np.hstack([hist, qs]).astype("float32").tobytes()
    return hashlib.md5(vec).hexdigest()[:20]  # short but collision-safe
