"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012), vectorised over points."""

from __future__ import annotations

import math

import numpy as np

from config import OneEuroConfig


class OneEuroFilter:
    """Adaptive low-pass filter: smooth while still, responsive while moving.

    Filters arrays of shape ``(..., D)``. The cutoff adapts per point using the
    speed of the whole D-dimensional vector, so x/y/z of a landmark always get
    the same amount of smoothing.
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t = 0.0

    @classmethod
    def from_config(cls, cfg: OneEuroConfig) -> OneEuroFilter:
        return cls(cfg.min_cutoff, cfg.beta, cfg.d_cutoff)

    @staticmethod
    def _alpha(dt: float, cutoff: float | np.ndarray) -> float | np.ndarray:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x = None
        self._dx = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filter sample ``x`` taken at time ``t`` (seconds) and return the estimate."""
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._dx is None or self._x.shape != x.shape:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x.copy()

        dt = t - self._t
        if dt <= 0.0:  # duplicate or out-of-order timestamp: keep the last estimate
            return self._x.copy()
        self._t = t

        raw_dx = (x - self._x) / dt
        self._dx += self._alpha(dt, self.d_cutoff) * (raw_dx - self._dx)
        speed = np.linalg.norm(self._dx, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        self._x += self._alpha(dt, cutoff) * (x - self._x)
        return self._x.copy()
