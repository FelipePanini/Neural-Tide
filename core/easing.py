"""Frame-rate independent easing helpers."""

from __future__ import annotations

import math
from typing import TypeVar

import numpy as np

T = TypeVar("T", float, np.ndarray)


def damp_factor(dt: float, tau: float) -> float:
    """Blend factor that closes ~63% of the gap to a target every ``tau`` seconds."""
    if tau <= 0.0:
        return 1.0
    return 1.0 - math.exp(-dt / tau)


def damp(current: T, target: T, tau: float, dt: float) -> T:
    """Exponentially move ``current`` towards ``target``; identical at any frame rate."""
    return current + (target - current) * damp_factor(dt, tau)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite ramp from 0 at ``edge0`` to 1 at ``edge1`` (edges may be reversed)."""
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)
