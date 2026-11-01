"""Smooth velocity fields used to stir particles (all vectorised over ``(N, 2)``)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-6


@dataclass(frozen=True)
class Stirrer:
    """Something that stirs the fluid: a hand, or the idle phantom."""

    center: np.ndarray  # (2,) world space
    velocity: np.ndarray  # (2,) world units per second
    weight: float  # 0..1 (hand presence)
    spin: float  # +1 counter-clockwise, -1 clockwise
    pinch: float = 0.0  # 0..1


def gaussian(r2: np.ndarray, reach: float) -> np.ndarray:
    """``exp(-r²/reach²)``: 1 at the centre, ~0 beyond ``2 × reach``."""
    return np.exp(-r2 / (reach * reach))


def vortex(rel: np.ndarray, r2: np.ndarray, circulation: float, core: float, reach: float) -> np.ndarray:
    """Lamb–Oseen vortex: solid-body spin inside ``core``, ~1/r outside, faded beyond ``reach``.

    The profile is finite at the centre (no singularity) and divergence-free,
    which is what makes the swirl read as fluid rather than as a rigid spin.
    """
    factor = circulation / (2.0 * math.pi) * (1.0 - np.exp(-r2 / (core * core))) / (r2 + _EPS) * gaussian(r2, reach)
    return np.stack((-rel[:, 1] * factor, rel[:, 0] * factor), axis=1)


def suction(rel: np.ndarray, r2: np.ndarray, speed: float, core: float, reach: float) -> np.ndarray:
    """Inward pull of about ``speed`` around ``core``, zero at the very centre."""
    r = np.sqrt(r2) + _EPS
    profile = (1.0 - np.exp(-r2 / (core * core))) * gaussian(r2, reach)
    return rel * (-speed * profile / r)[:, None]
