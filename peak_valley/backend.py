"""Array backend utilities for optional CUDA acceleration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import signal as scipy_signal


@dataclass(frozen=True)
class ArrayBackend:
    """Container describing the active array backend."""

    xp: Any
    signal: Any
    is_gpu: bool

    def to_cpu(self, array: Any) -> np.ndarray:
        """Convert ``array`` to a NumPy ``ndarray`` if necessary."""

        if self.is_gpu:
            import cupy as cp  # type: ignore[import-not-found]

            if isinstance(array, cp.ndarray):
                return cp.asnumpy(array)
        return np.asarray(array)

    @property
    def name(self) -> str:
        return "CUDA" if self.is_gpu else "CPU"


def _create_gpu_backend() -> ArrayBackend | None:
    """Return a CUDA-capable backend or ``None`` if unavailable."""

    try:
        import cupy as cp  # type: ignore[import-not-found]
        from cupyx.scipy import signal as cupy_signal  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            return None
        # Lightweight probe to ensure the driver/context is usable.
        cp.zeros((1,), dtype=cp.float32).sum().item()
    except Exception:
        return None

    return ArrayBackend(cp, cupy_signal, True)


@lru_cache(maxsize=1)
def get_array_backend() -> ArrayBackend:
    """Return the preferred array backend for heavy computations."""

    mode = os.getenv("PV_USE_CUDA", "auto").strip().lower()

    if mode in {"0", "false", "off", "cpu"}:
        return ArrayBackend(np, scipy_signal, False)

    if mode in {"1", "true", "on", "cuda", "gpu"}:
        gpu_backend = _create_gpu_backend()
        if gpu_backend is not None:
            return gpu_backend
        return ArrayBackend(np, scipy_signal, False)

    # auto-detect
    gpu_backend = _create_gpu_backend()
    if gpu_backend is not None:
        return gpu_backend

    return ArrayBackend(np, scipy_signal, False)


def backend_description() -> str:
    """Human-readable description of the selected backend."""

    backend = get_array_backend()
    if backend.is_gpu:
        return "CUDA (CuPy)"
    return "CPU (NumPy)"
