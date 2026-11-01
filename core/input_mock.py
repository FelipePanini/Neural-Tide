"""Synthetic hands: the mouse-driven mock source and a landmark generator for tests.

The generator builds a simple articulated hand (metres) with a curl value per
finger, so the mock exercises exactly the same filtering, finger counting and
pinch code as real MediaPipe detections.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from config import Config
from core.camera import CameraFrame
from core.easing import damp
from core.hand import (
    EMPTY_FRAME,
    INDEX_DIP,
    INDEX_TIP,
    LEFT,
    PALM_POINTS,
    RIGHT,
    THUMB_IP,
    THUMB_TIP,
    HandObservation,
    SourceStatus,
    TrackingFrame,
)
from core.stats import RateMeter
from core.tracker import HandFilterBank

# Right hand, palm facing the camera, as seen in the mirrored (selfie) view:
# x right, y up, z towards the camera, wrist at the origin.
_BASES = np.array([(-0.022, 0.028), (-0.030, 0.088), (-0.008, 0.094), (0.013, 0.089), (0.031, 0.078)])
_BONES = np.array(
    [(0.040, 0.032, 0.028), (0.042, 0.025, 0.021), (0.046, 0.029, 0.023), (0.043, 0.027, 0.021), (0.033, 0.021, 0.019)]
)
_SPLAY_DEG = np.array([52.0, 8.0, 0.0, -7.0, -16.0])  # from +y; positive leans to the thumb side
_FINGER_FLEX_DEG = np.array([80.0, 100.0, 70.0])  # MCP, PIP, DIP bend at full curl
_THUMB_SWEEP_DEG = np.array([40.0, 45.0, 40.0])  # thumb sweeps across the palm when curled
_THUMB_LIFT_DEG = 25.0  # ...and rises towards the camera to pass over it
_PINCH_CURL = 0.3  # extra curl of thumb and index while pinching
_PINCH_JOINT_PULL = 0.45  # how far the joints below the tips follow the pinch
HAND_SPAN_M = 0.19  # wrist to middle fingertip

# Curl per finger (thumb → pinky): 0 = straight, 1 = fully curled.
FINGER_PRESETS: dict[int, tuple[float, ...]] = {
    0: (1.0, 1.0, 1.0, 1.0, 1.0),
    1: (1.0, 0.0, 1.0, 1.0, 1.0),
    2: (1.0, 0.0, 0.0, 1.0, 1.0),
    3: (1.0, 0.0, 0.0, 0.0, 1.0),
    4: (1.0, 0.0, 0.0, 0.0, 0.0),
    5: (0.0, 0.0, 0.0, 0.0, 0.0),
}
HORNS_PRESET = (1.0, 0.0, 1.0, 1.0, 0.0)

_Z = np.array([0.0, 0.0, 1.0])


def _long_finger(base: np.ndarray, bones: np.ndarray, splay_deg: float, curl: float) -> list[np.ndarray]:
    splay = math.radians(splay_deg)
    forward = np.array([-math.sin(splay), math.cos(splay), 0.0])
    points = [np.array([base[0], base[1], 0.0])]
    bend = 0.0
    for length, flex in zip(bones, _FINGER_FLEX_DEG, strict=True):
        bend += math.radians(curl * flex)
        points.append(points[-1] + length * (math.cos(bend) * forward + math.sin(bend) * _Z))
    return points


def _thumb(base: np.ndarray, bones: np.ndarray, splay_deg: float, curl: float) -> list[np.ndarray]:
    lift = math.radians(curl * _THUMB_LIFT_DEG)
    points = [np.array([base[0], base[1], 0.0])]
    angle = splay_deg
    for length, sweep in zip(bones, _THUMB_SWEEP_DEG, strict=True):
        angle -= curl * sweep
        a = math.radians(angle)
        direction = np.array([-math.sin(a) * math.cos(lift), math.cos(a) * math.cos(lift), math.sin(lift)])
        points.append(points[-1] + length * direction)
    return points


def hand_model(curls: Sequence[float], pinch: float = 0.0) -> np.ndarray:
    """21 landmarks (metres) of a right hand in the local frame described above."""
    curls = list(curls)
    curls[0] = max(curls[0], _PINCH_CURL * pinch)
    curls[1] = max(curls[1], _PINCH_CURL * pinch)
    points = [np.zeros(3), *_thumb(_BASES[0], _BONES[0], _SPLAY_DEG[0], curls[0])]
    for finger in range(1, 5):
        points.extend(_long_finger(_BASES[finger], _BONES[finger], _SPLAY_DEG[finger], curls[finger]))
    pts = np.array(points)

    if pinch > 0.0:
        meet = 0.5 * (pts[THUMB_TIP] + pts[INDEX_TIP])
        for index, weight in ((THUMB_TIP, 1.0), (INDEX_TIP, 1.0), (THUMB_IP, _PINCH_JOINT_PULL), (INDEX_DIP, _PINCH_JOINT_PULL)):
            pts[index] += (meet - pts[index]) * pinch * weight
    return pts


def synthetic_hand(
    center: tuple[float, float],
    curls: Sequence[float],
    *,
    pinch: float = 0.0,
    handedness: str = RIGHT,
    height: float = 0.34,
    image_aspect: float = 16.0 / 9.0,
    roll_deg: float = 0.0,
    noise_m: float = 0.0,
    rng: np.random.Generator | None = None,
) -> HandObservation:
    """A hand observation with its palm centre at ``center`` (image space).

    ``height`` is the wrist-to-fingertip span as a fraction of the image height.
    """
    pts = hand_model(curls, pinch)
    if noise_m > 0.0:
        pts = pts + (rng or np.random.default_rng()).normal(0.0, noise_m, pts.shape)
    if handedness == LEFT:
        pts[:, 0] = -pts[:, 0]

    anchor = pts[list(PALM_POINTS)].mean(axis=0)
    roll = math.radians(roll_deg)
    rot = np.array([[math.cos(roll), -math.sin(roll)], [math.sin(roll), math.cos(roll)]])
    rel = pts - anchor
    rel[:, :2] = rel[:, :2] @ rot.T

    scale = height / HAND_SPAN_M  # image-height units per metre
    image = np.empty_like(rel)
    image[:, 0] = center[0] + rel[:, 0] * scale / image_aspect
    image[:, 1] = center[1] - rel[:, 1] * scale
    image[:, 2] = -(rel[:, 2] - rel[0, 2]) * scale  # MediaPipe: wrist = 0, closer = negative

    world = rel - rel.mean(axis=0)
    world[:, 1:] = -world[:, 1:]  # MediaPipe world axes: y down, z away from the camera
    return HandObservation(landmarks=image, world=world, handedness=handedness, score=1.0)


class MockHandSource:
    """Turns the mouse into a hand so everything can be tested without a webcam.

    Mouse wheel / trackpad scroll or arrow keys: raised fingers (0–5) · left
    button: pinch · right button or F: fist · H: mirrored second hand · moving
    the pointer out of the window: hand leaves.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg.mock
        self._bank = HandFilterBank(cfg.tracker, cfg.geometry)
        self._rng = np.random.default_rng(cfg.mock.seed)
        self._meter = RateMeter(cfg.debug.rate_window_s)
        self._pointer: tuple[float, float] | None = None
        self._pinch_target = 0.0
        self._pinch = 0.0
        self._fist = False
        self._finger_count = cfg.mock.default_fingers
        self._curls = np.array(FINGER_PRESETS[self._finger_count])
        self._second_hand = False
        self._scroll_accum = 0.0
        self._scroll_blocked_until = 0.0
        self._frame = EMPTY_FRAME

    @property
    def finger_count(self) -> int:
        return self._finger_count

    @property
    def second_hand(self) -> bool:
        return self._second_hand

    def start(self) -> None:
        """Nothing to start: the mock is advanced from the render loop."""

    def stop(self) -> None:
        """Nothing to stop."""

    def latest(self) -> TrackingFrame:
        return self._frame

    def camera_frame(self) -> CameraFrame | None:
        return None

    def status(self) -> SourceStatus:
        return SourceStatus(label="mock (mouse)", tracking_fps=self._meter.rate)

    def set_pointer(self, uv: tuple[float, float] | None) -> None:
        self._pointer = uv

    def set_pinch(self, active: bool) -> None:
        self._pinch_target = 1.0 if active else 0.0

    def set_fist(self, active: bool) -> None:
        self._fist = active

    def change_fingers(self, delta: int) -> None:
        self._finger_count = min(max(self._finger_count + delta, 0), 5)

    def scroll(self, amount: float, now: float) -> None:
        """Mouse wheel or trackpad: one finger per ``wheel_step``, then a short cooldown."""
        if now < self._scroll_blocked_until:
            return
        self._scroll_accum += amount
        if abs(self._scroll_accum) >= self._cfg.wheel_step:
            self.change_fingers(1 if self._scroll_accum > 0 else -1)
            self._scroll_accum = 0.0
            self._scroll_blocked_until = now + self._cfg.wheel_cooldown_s

    def toggle_second_hand(self) -> None:
        self._second_hand = not self._second_hand

    def update(self, now: float, dt: float, image_aspect: float) -> None:
        """Advance the mock by ``dt`` and publish a new frame stamped ``now``."""
        cfg = self._cfg
        target = np.array(FINGER_PRESETS[0 if self._fist else self._finger_count])
        self._curls = damp(self._curls, target, cfg.curl_tau_s, dt)
        self._pinch = damp(self._pinch, self._pinch_target, cfg.pinch_tau_s, dt)

        observations: list[HandObservation] = []
        if self._pointer is not None:
            u, v = self._pointer
            observations.append(self._observe((u, v), RIGHT, image_aspect))
            if self._second_hand:
                observations.append(self._observe((1.0 - u, v), LEFT, image_aspect))

        hands = self._bank.update(observations, now)
        self._frame = TrackingFrame(self._frame.seq + 1, now, hands)
        self._meter.tick(now)

    def _observe(self, center: tuple[float, float], handedness: str, image_aspect: float) -> HandObservation:
        cfg = self._cfg
        lean = (center[0] - 0.5) * 2.0 * cfg.tilt_deg  # lean outwards towards the sides
        return synthetic_hand(
            center,
            self._curls,
            pinch=self._pinch,
            handedness=handedness,
            height=cfg.hand_height,
            image_aspect=image_aspect,
            roll_deg=-lean,
            noise_m=cfg.jitter * HAND_SPAN_M / cfg.hand_height,
            rng=self._rng,
        )
