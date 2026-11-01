"""Render-side hand state: world-space hands eased every frame.

The tracker runs at camera rate (~30 Hz) while rendering runs at 60+ Hz, so
the render loop eases towards the latest filtered landmarks instead of
stepping. Hands also fade in and out (``presence``), which is what lets
effects drift back to their idle state instead of snapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import HandStateConfig
from core.easing import damp
from core.hand import FINGERTIPS, INDEX_TIP, PALM_POINTS, THUMB_TIP, TrackingFrame, palm_center
from core.viewport import Viewport


@dataclass
class HandState:
    """What effects consume: one hand in world space."""

    slot: int
    handedness: str
    points: np.ndarray  # (21, 2) world space, eased
    raw_points: np.ndarray  # (21, 2) world space, unfiltered (debug overlay)
    velocity: np.ndarray  # (21, 2) world units per second, smoothed
    depth: np.ndarray  # (21,) relative depth, negative = closer to the camera
    world: np.ndarray  # (21, 3) metres, latest filtered landmarks (not eased)
    fingers: tuple[bool, ...]
    pinch: float  # eased 0..1
    presence: float  # eased 0..1
    visible: bool  # reported by the tracker this frame

    @property
    def finger_count(self) -> int:
        return sum(self.fingers)

    @property
    def palm(self) -> np.ndarray:
        return palm_center(self.points)

    @property
    def palm_velocity(self) -> np.ndarray:
        return self.velocity[list(PALM_POINTS)].mean(axis=0)

    @property
    def tips(self) -> np.ndarray:
        return self.points[list(FINGERTIPS)]

    @property
    def pinch_point(self) -> np.ndarray:
        return 0.5 * (self.points[THUMB_TIP] + self.points[INDEX_TIP])


class HandStateSmoother:
    """Keeps one :class:`HandState` per tracker slot and eases it every frame."""

    def __init__(self, cfg: HandStateConfig) -> None:
        self._cfg = cfg
        self._states: dict[int, HandState] = {}

    def update(self, frame: TrackingFrame, viewport: Viewport, dt: float) -> list[HandState]:
        cfg = self._cfg
        seen: set[int] = set()
        for hand in frame.hands:
            target = viewport.image_to_world(hand.landmarks)
            state = self._states.get(hand.slot)
            if state is None:
                state = HandState(
                    slot=hand.slot,
                    handedness=hand.handedness,
                    points=target,
                    raw_points=viewport.image_to_world(hand.raw_landmarks),
                    velocity=np.zeros_like(target),
                    depth=hand.landmarks[:, 2].copy(),
                    world=hand.world,
                    fingers=hand.fingers,
                    pinch=hand.pinch,
                    presence=0.0,
                    visible=True,
                )
                self._states[hand.slot] = state
            elif dt > 0.0:
                previous = state.points
                state.points = damp(previous, target, cfg.position_tau_s, dt)
                state.velocity = damp(state.velocity, (state.points - previous) / dt, cfg.velocity_tau_s, dt)
                state.raw_points = viewport.image_to_world(hand.raw_landmarks)
                state.depth = damp(state.depth, hand.landmarks[:, 2], cfg.position_tau_s, dt)
                state.pinch = damp(state.pinch, hand.pinch, cfg.pinch_tau_s, dt)
                state.world = hand.world
                state.handedness = hand.handedness
                state.fingers = hand.fingers
            state.visible = True
            state.presence = damp(state.presence, 1.0, cfg.presence_in_tau_s, dt)
            seen.add(hand.slot)

        for slot, state in list(self._states.items()):
            if slot in seen:
                continue
            state.visible = False
            state.presence = damp(state.presence, 0.0, cfg.presence_out_tau_s, dt)
            state.velocity = damp(state.velocity, np.zeros_like(state.velocity), cfg.velocity_tau_s, dt)
            state.pinch = damp(state.pinch, 0.0, cfg.pinch_tau_s, dt)
            if state.presence < cfg.presence_epsilon:
                del self._states[slot]

        return sorted(self._states.values(), key=lambda s: s.slot)
