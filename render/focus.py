"""Spotlights that limit the effect layer to the area around the hands."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from config import FocusConfig
from core.easing import damp
from core.hand_state import HandState


@dataclass(frozen=True)
class Spot:
    center: np.ndarray  # world space
    radius: float  # world units; brightness falls to exp(-falloff) here
    weight: float  # 0..1


def hand_spot(hand: HandState, cfg: FocusConfig, scale: float = 1.0) -> Spot:
    """A spotlight around the whole hand, sized by how spread out it is on screen."""
    center = hand.points.mean(axis=0)
    size = float(np.linalg.norm(hand.points - center, axis=1).max())
    radius = min(max(size * cfg.hand_scale, cfg.min_radius), cfg.max_radius) * scale
    return Spot(center=center, radius=radius, weight=hand.presence)


class FocusState:
    """Eased on/off amount plus the spotlights for the current frame."""

    def __init__(self, cfg: FocusConfig) -> None:
        self._cfg = cfg
        self.enabled = cfg.start_enabled
        self.amount = 1.0 if cfg.start_enabled else 0.0
        self._idle_weight = 1.0
        self._time = 0.0
        self.spots: list[Spot] = []

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def update(self, hands: Sequence[HandState], dt: float, half_width: float, scale: float = 1.0) -> list[Spot]:
        cfg = self._cfg
        self._time += dt
        self.amount = damp(self.amount, 1.0 if self.enabled else 0.0, cfg.toggle_tau_s, dt)
        spots = [hand_spot(h, cfg, scale) for h in hands if h.presence > 0.01]
        presence = max((h.presence for h in hands), default=0.0)
        self._idle_weight = damp(self._idle_weight, 1.0 - presence, cfg.idle_fade_tau_s, dt)
        if self._idle_weight > 0.01:
            phase = 2.0 * math.pi * self._time / cfg.idle_period_s
            ax, ay = cfg.idle_path
            center = np.array((ax * half_width * math.sin(phase), ay * math.sin(2.0 * phase + 1.3)))
            spots.append(Spot(center, cfg.idle_radius * scale, self._idle_weight * cfg.idle_brightness))
        self.spots = sorted(spots, key=lambda s: -s.weight)[: cfg.max_spots]
        return self.spots
