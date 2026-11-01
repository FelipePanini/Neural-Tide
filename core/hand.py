"""Hand landmark layout, per-hand data containers and hand geometry.

Coordinates used across the project:

* image space: MediaPipe's normalised coordinates in the *displayed* (mirrored)
  frame, x to the right and y down, both in 0..1; z is relative depth with the
  wrist at 0 and negative values closer to the camera.
* world landmarks: metres around the hand centre, as returned by MediaPipe.
  Finger states and pinch use these because they do not depend on how far the
  hand is from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from config import HandGeometryConfig
from core.easing import smoothstep

if TYPE_CHECKING:
    from core.camera import CameraFrame

NUM_LANDMARKS = 21

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
PALM_POINTS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
# Wrist followed by the four joints of each long finger.
_FINGER_CHAINS = np.array(
    [
        (WRIST, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
        (WRIST, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
        (WRIST, RING_MCP, RING_PIP, RING_DIP, RING_TIP),
        (WRIST, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    ]
)
_THUMB_CHAIN = np.array((THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP))

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)  # fmt: skip

LEFT = "Left"
RIGHT = "Right"


def other_hand(handedness: str) -> str:
    return LEFT if handedness == RIGHT else RIGHT


@dataclass(frozen=True)
class HandObservation:
    """One raw detection, before filtering."""

    landmarks: np.ndarray  # (21, 3) image space
    world: np.ndarray  # (21, 3) metres
    handedness: str  # the user's real hand: LEFT or RIGHT
    score: float  # handedness confidence


@dataclass(frozen=True)
class TrackedHand:
    """A filtered hand with a stable slot id (ids are never reused)."""

    slot: int
    handedness: str
    landmarks: np.ndarray  # (21, 3) image space, One Euro filtered
    raw_landmarks: np.ndarray  # (21, 3) image space, straight from the detector
    world: np.ndarray  # (21, 3) metres, One Euro filtered
    fingers: tuple[bool, ...]  # extended state, thumb → pinky
    pinch: float  # 0 = open, 1 = thumb and index tips touching
    stale: bool  # missing from the latest detection, held for a short grace period

    @property
    def finger_count(self) -> int:
        return sum(self.fingers)


@dataclass(frozen=True)
class TrackingFrame:
    """Everything the hand source knows at one instant."""

    seq: int
    timestamp: float
    hands: tuple[TrackedHand, ...]


EMPTY_FRAME = TrackingFrame(seq=0, timestamp=0.0, hands=())


@dataclass(frozen=True)
class SourceStatus:
    """Numbers and messages the debug overlay shows about the input."""

    label: str
    tracking_fps: float = 0.0
    camera_fps: float = 0.0
    inference_ms: float = 0.0
    resolution: tuple[int, int] | None = None
    error: str | None = None


class HandSource(Protocol):
    """Anything that produces hands: the webcam tracker or the mouse mock."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def latest(self) -> TrackingFrame: ...

    def status(self) -> SourceStatus: ...

    def camera_frame(self) -> CameraFrame | None:
        """Newest camera image for the background, if the source has one."""
        ...

    def set_segmentation(self, enabled: bool) -> None:
        """Turn person segmentation on or off (it costs CPU, so only while needed)."""
        ...

    def segmentation(self) -> np.ndarray | None:
        """Latest person mask (H, W), 0..1, in the displayed image space."""
        ...


# --------------------------------------------------------------------------- geometry


def palm_center(points: np.ndarray) -> np.ndarray:
    """Mean of the wrist and the four finger bases (works for 2D or 3D points)."""
    return points[list(PALM_POINTS)].mean(axis=0)


def palm_size(world: np.ndarray) -> float:
    """Wrist to middle-finger base distance: a scale reference for ratios."""
    return float(np.linalg.norm(world[MIDDLE_MCP] - world[WRIST]))


def _angles_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between matching vectors of ``a`` and ``b`` (last axis)."""
    dot = np.sum(a * b, axis=-1)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    cos = np.clip(dot / np.maximum(norms, 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def finger_bends(world: np.ndarray) -> np.ndarray:
    """Total joint bend per finger in degrees, thumb → pinky (0 = perfectly straight)."""
    chains = world[_FINGER_CHAINS]  # (4, 5, 3)
    bones = np.diff(chains, axis=1)  # (4, 4, 3)
    long_fingers = _angles_between(bones[:, :-1], bones[:, 1:]).sum(axis=1)
    thumb_bones = np.diff(world[_THUMB_CHAIN], axis=0)  # (3, 3)
    thumb = _angles_between(thumb_bones[:-1], thumb_bones[1:]).sum()
    return np.concatenate(([thumb], long_fingers))


def fingers_extended(world: np.ndarray, cfg: HandGeometryConfig) -> tuple[bool, ...]:
    """Which fingers are raised, thumb → pinky.

    Long fingers are extended when their joints are nearly straight. The thumb
    also has to point away from the palm, because a straight thumb tucked
    across the palm (the usual "4 fingers" pose) should not count.
    """
    bends = finger_bends(world)
    palm_width = np.linalg.norm(world[INDEX_MCP] - world[PINKY_MCP])
    spread = np.linalg.norm(world[THUMB_TIP] - world[PINKY_MCP]) / max(palm_width, 1e-9)
    thumb = bool(bends[0] < cfg.thumb_max_bend_deg and spread > cfg.thumb_min_spread)
    return (thumb, *(bool(b < cfg.finger_max_bend_deg) for b in bends[1:]))


def pinch_strength(world: np.ndarray, cfg: HandGeometryConfig) -> float:
    """0 when thumb and index tips are apart, rising smoothly to 1 when they touch."""
    ratio = np.linalg.norm(world[THUMB_TIP] - world[INDEX_TIP]) / max(palm_size(world), 1e-9)
    return smoothstep(cfg.pinch_open_ratio, cfg.pinch_closed_ratio, float(ratio))
