"""Lightweight peak classifier utilities.

A gradient-boosted classifier turns local KDE windows into peak/non-peak
probabilities.  The scorer expects a label per grid point and can persist its
state via joblib for reuse in batch or CLI workflows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib


def _window_features(trace: np.ndarray, radius: int) -> np.ndarray:
    """Return normalised sliding windows for a 1D trace."""

    radius = max(1, int(radius))
    padded = np.pad(trace.astype(float), radius, mode="edge")
    windows = []
    width = 2 * radius + 1
    for start in range(trace.size):
        window = padded[start : start + width]
        centred = window - float(window.mean())
        scale = float(np.max(np.abs(centred))) or 1.0
        windows.append(centred / scale)
    return np.vstack(windows)


def _prepare_training_set(
    traces: Sequence[np.ndarray], labels: Sequence[np.ndarray], radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten labelled traces into (features, targets)."""

    feats: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for trace, lbl in zip(traces, labels):
        arr = np.asarray(trace, float).ravel()
        lab = np.asarray(lbl, int).ravel()
        if arr.size != lab.size:
            raise ValueError("Trace and label lengths must match for training")
        feats.append(_window_features(arr, radius))
        targets.append(lab)
    return np.vstack(feats), np.concatenate(targets)


@dataclass
class GradientBoostingPeakScorer:
    """Score KDE grid points as peak/non-peak using gradient boosting."""

    window_radius: int = 5
    model: GradientBoostingClassifier | None = None

    def fit(
        self, traces: Sequence[np.ndarray], labels: Sequence[np.ndarray]
    ) -> "GradientBoostingPeakScorer":
        X, y = _prepare_training_set(traces, labels, self.window_radius)
        clf = GradientBoostingClassifier(random_state=0)
        clf.fit(X, y)
        self.model = clf
        return self

    def score_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Return peak probabilities for each grid cell."""

        _ = xs  # only ys is needed to derive local windows
        if self.model is None:
            raise RuntimeError("Peak scorer has not been trained")
        feats = _window_features(np.asarray(ys, float), self.window_radius)
        probs = self.model.predict_proba(feats)
        if probs.shape[1] == 1:
            return np.zeros(ys.shape, float)
        return probs[:, 1]

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Peak scorer has not been trained")
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"radius": self.window_radius, "model": self.model}, dest)
        return dest


def load_peak_scorer(path: str | Path) -> GradientBoostingPeakScorer:
    """Load a persisted :class:`GradientBoostingPeakScorer`."""

    saved = joblib.load(Path(path))
    scorer = GradientBoostingPeakScorer(saved.get("radius", 5))
    scorer.model = saved.get("model")
    if scorer.model is None:
        raise RuntimeError("Loaded peak scorer is missing the underlying model")
    return scorer


def training_rows_from_labels(
    labelled_traces: Iterable[tuple[np.ndarray, np.ndarray]], window_radius: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Helper to build training matrices from labelled windows.

    Parameters
    ----------
    labelled_traces:
        Iterable of ``(trace, labels)`` pairs where ``labels`` is a binary array
        (1=peak, 0=non-peak) aligned to ``trace``.
    window_radius:
        Half-width of the sliding window used to derive features.
    """

    traces, labels = zip(*[(np.asarray(t), np.asarray(l)) for t, l in labelled_traces])
    return _prepare_training_set(traces, labels, window_radius)
