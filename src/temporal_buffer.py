"""
src/temporal_buffer.py
Rolling temporal window that accumulates per-frame feature dicts
and exposes aggregated statistics for model inference.
"""

import sys
from pathlib import Path
from collections import deque

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class TemporalBuffer:
    """
    Fixed-size rolling buffer of feature dicts.

    Aggregates into a flat vector:
        - mean, std, min, max, trend (slope) for each numeric feature
    Ready to be fed directly into the trained ensemble.
    """

    def __init__(self, window: int = None, feature_cols: list[str] = None):
        self.window      = window       or config.TEMPORAL_WINDOW
        self.feature_cols = feature_cols or config.FEATURE_COLS
        self._buffer: deque = deque(maxlen=self.window)

    # ── Public API ────────────────────────────────────────────────────────────
    def push(self, feature_dict: dict) -> None:
        """Append one frame's features to the rolling buffer."""
        self._buffer.append({k: float(feature_dict.get(k, 0.0))
                              for k in self.feature_cols})

    def is_ready(self) -> bool:
        """True when at least half the window is filled."""
        return len(self._buffer) >= max(1, self.window // 2)

    def aggregate(self) -> dict | None:
        """
        Return a flat dict of aggregated features.
        Returns None if buffer is not ready.
        """
        if not self.is_ready():
            return None

        arr = np.array([[row[k] for k in self.feature_cols]
                        for row in self._buffer])           # (T, F)

        agg = {}
        for i, col in enumerate(self.feature_cols):
            vals = arr[:, i]
            agg[f"{col}_mean"]  = float(np.mean(vals))
            agg[f"{col}_std"]   = float(np.std(vals))
            agg[f"{col}_min"]   = float(np.min(vals))
            agg[f"{col}_max"]   = float(np.max(vals))
            # Linear trend (slope of least-squares fit)
            if len(vals) > 1:
                x = np.arange(len(vals))
                agg[f"{col}_trend"] = float(np.polyfit(x, vals, 1)[0])
            else:
                agg[f"{col}_trend"] = 0.0

        return agg

    def latest(self) -> dict | None:
        """Return the most recent raw feature dict, or None."""
        if not self._buffer:
            return None
        return dict(self._buffer[-1])

    def reset(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
