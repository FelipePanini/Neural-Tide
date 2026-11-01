"""Gesture recognition: a per-hand state machine with debounce, holds and cooldown.

Every frame each hand gets a raw pose (fist, 1–5 fingers, horns, pinch). A
pose only becomes *stable* after ``debounce_frames`` identical frames. Stable
poses then drive actions:

* 1–4 fingers held still for ``count_hold_s`` → select that mode.
* Open palm (5 fingers) is a two-stage hold: released between
  ``count_hold_s`` and ``freeze_hold_s`` it selects mode 5; held the whole
  ``freeze_hold_s`` it toggles freeze instead. (Both gestures use the same
  pose, so the hold length tells them apart.)
* Horns (index + pinky) → next palette; fist → explode.
* Pinch → start / end events for the active mode.
* Open hand moving sideways fast → next / previous mode.

After any action the hand is disarmed until its pose changes, and a global
cooldown blocks every other action for ``cooldown_s``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from config import GestureConfig
from core.hand import FINGERTIPS, INDEX_TIP, MIDDLE_TIP, WRIST, palm_size
from core.hand_state import HandState


class Pose(Enum):
    NONE = "—"
    FIST = "punho"
    ONE = "1 dedo"
    TWO = "2 dedos"
    THREE = "3 dedos"
    FOUR = "4 dedos"
    OPEN = "palma aberta"
    HORNS = "chifrinho"
    PINCH = "pinça"


COUNT_POSES: dict[Pose, int] = {Pose.ONE: 1, Pose.TWO: 2, Pose.THREE: 3, Pose.FOUR: 4, Pose.OPEN: 5}
_POSE_BY_COUNT = {count: pose for pose, count in COUNT_POSES.items()}
_OPEN_HAND = (Pose.FOUR, Pose.OPEN)


class GestureKind(Enum):
    SELECT_MODE = "modo"
    NEXT_MODE = "próximo modo"
    PREVIOUS_MODE = "modo anterior"
    PINCH_START = "pinça"
    PINCH_END = "fim da pinça"
    EXPLODE = "explosão"
    CYCLE_PALETTE = "paleta"
    TOGGLE_FREEZE = "congelar"


@dataclass(frozen=True)
class GestureEvent:
    kind: GestureKind
    slot: int
    position: np.ndarray  # world space
    mode: int | None = None  # 0-based, for SELECT_MODE


@dataclass(frozen=True)
class HoldProgress:
    """A hold in progress, for the progress ring."""

    slot: int
    position: np.ndarray  # world space, at a fingertip
    progress: float  # 0..1 of the full hold
    labels: tuple[str, ...]
    marker: float | None = None  # stage boundary on the ring (open palm), as a fraction
    blocked: bool = False  # complete but waiting for the cooldown


@dataclass(frozen=True)
class GestureFrame:
    events: tuple[GestureEvent, ...]
    holds: tuple[HoldProgress, ...]
    poses: dict[int, Pose]
    intensity: float | None  # from the distance between two hands, if both are visible


def index_reach(hand: HandState) -> float:
    """Index fingertip to wrist distance relative to palm size (low when curled)."""
    world = hand.world
    return float(np.linalg.norm(world[INDEX_TIP] - world[WRIST])) / max(palm_size(world), 1e-9)


def classify(hand: HandState, cfg: GestureConfig, was_pinching: bool = False) -> Pose:
    """Raw pose of one hand this frame."""
    threshold = cfg.pinch_off if was_pinching else cfg.pinch_on
    if hand.pinch >= threshold and index_reach(hand) >= cfg.pinch_min_index_reach:
        return Pose.PINCH
    thumb, index, middle, ring, pinky = hand.fingers
    if not any(hand.fingers):
        return Pose.FIST
    if index and pinky and not middle and not ring:  # thumb is free: 🤘 and 🤟 both count
        return Pose.HORNS
    return _POSE_BY_COUNT[sum(hand.fingers)]


def _ring_anchor(hand: HandState, pose: Pose) -> np.ndarray:
    """Where the progress ring sits: a raised fingertip, or the palm for a fist."""
    if pose is Pose.FIST:
        return hand.palm
    if pose is Pose.OPEN:
        return hand.points[MIDDLE_TIP]
    if hand.fingers[1]:
        return hand.points[INDEX_TIP]
    raised = [tip for tip, up in zip(FINGERTIPS, hand.fingers, strict=True) if up]
    return hand.points[raised[0]] if raised else hand.palm


@dataclass
class _HandGestures:
    candidate: Pose = Pose.NONE
    candidate_frames: int = 0
    pose: Pose = Pose.NONE  # debounced
    held: float = 0.0
    armed: bool = True
    pending_mode: bool = False  # open palm went past the mode-5 stage
    last_palm: np.ndarray = field(default_factory=lambda: np.zeros(2))
    history: deque[tuple[float, float, float, bool]] = field(default_factory=deque)


class GestureEngine:
    """Turns eased hand states into discrete gesture events and hold progress."""

    def __init__(self, cfg: GestureConfig) -> None:
        self._cfg = cfg
        self._hands: dict[int, _HandGestures] = {}
        self._cooldown_until = 0.0
        self.last_event: GestureEvent | None = None

    def cooling_down(self, now: float) -> bool:
        return now < self._cooldown_until

    def update(
        self,
        hands: Sequence[HandState],
        dt: float,
        now: float,
        *,
        current_mode: int,
        mode_count: int,
        frozen: bool = False,
    ) -> GestureFrame:
        cfg = self._cfg
        events: list[GestureEvent] = []
        holds: list[HoldProgress] = []
        active = [h for h in hands if h.visible and h.presence >= cfg.presence_min]

        for hand in active:
            st = self._hands.setdefault(hand.slot, _HandGestures())
            st.last_palm = hand.palm
            previous = st.pose
            self._debounce(st, classify(hand, cfg, previous is Pose.PINCH))
            if st.pose is not previous:
                self._leave_pose(st, previous, hand.slot, now, events, mode_count)
                if st.pose is Pose.PINCH:
                    events.append(GestureEvent(GestureKind.PINCH_START, hand.slot, hand.pinch_point))

            swipe = self._detect_swipe(st, hand, now)
            if swipe is not None:
                events.append(swipe)
                self._fired(st, now)
                continue
            hold = self._update_hold(st, hand, dt, now, events, current_mode, mode_count, frozen)
            if hold is not None:
                holds.append(hold)

        visible_slots = {h.slot for h in active}
        for slot in [s for s in self._hands if s not in visible_slots]:
            st = self._hands.pop(slot)
            self._leave_pose(st, st.pose, slot, now, events, mode_count)

        if events:
            self.last_event = events[-1]
        return GestureFrame(
            events=tuple(events),
            holds=tuple(holds),
            poses={slot: st.pose for slot, st in self._hands.items()},
            intensity=self._two_hand_intensity(active),
        )

    # ------------------------------------------------------------------ helpers

    def _debounce(self, st: _HandGestures, raw: Pose) -> None:
        if raw is st.candidate:
            st.candidate_frames += 1
        else:
            st.candidate, st.candidate_frames = raw, 1
        if st.candidate_frames >= self._cfg.debounce_frames and st.candidate is not st.pose:
            st.pose = st.candidate

    def _leave_pose(
        self, st: _HandGestures, previous: Pose, slot: int, now: float, events: list[GestureEvent], mode_count: int
    ) -> None:
        """Pose changed (or hand left): finish what the old pose started and re-arm."""
        if previous is Pose.PINCH:
            events.append(GestureEvent(GestureKind.PINCH_END, slot, st.last_palm))
        if previous is Pose.OPEN and st.pending_mode and not self.cooling_down(now) and mode_count > 4:
            events.append(GestureEvent(GestureKind.SELECT_MODE, slot, st.last_palm, mode=4))
            self._fired(st, now)
        st.held = 0.0
        st.armed = True
        st.pending_mode = False

    def _fired(self, st: _HandGestures, now: float) -> None:
        st.armed = False
        st.pending_mode = False
        st.history.clear()
        self._cooldown_until = now + self._cfg.cooldown_s

    def _detect_swipe(self, st: _HandGestures, hand: HandState, now: float) -> GestureEvent | None:
        cfg = self._cfg
        palm = hand.palm
        st.history.append((now, float(palm[0]), float(palm[1]), st.pose in _OPEN_HAND))
        while st.history and now - st.history[0][0] > cfg.swipe_window_s:
            st.history.popleft()
        if len(st.history) < 3 or self.cooling_down(now) or st.pose is Pose.PINCH:
            return None
        _, x0, y0, _ = st.history[0]
        dx, dy = float(palm[0]) - x0, float(palm[1]) - y0
        open_share = sum(sample[3] for sample in st.history) / len(st.history)
        vx = float(hand.palm_velocity[0])
        if (
            open_share >= cfg.swipe_open_share
            and abs(dx) >= cfg.swipe_min_distance
            and abs(dx) >= cfg.swipe_axis_ratio * abs(dy)
            and abs(vx) >= cfg.swipe_min_speed
            and np.sign(vx) == np.sign(dx)
        ):
            moved_left = dx < 0
            kind = GestureKind.NEXT_MODE if moved_left == cfg.swipe_left_is_next else GestureKind.PREVIOUS_MODE
            return GestureEvent(kind, hand.slot, palm)
        return None

    def _update_hold(
        self,
        st: _HandGestures,
        hand: HandState,
        dt: float,
        now: float,
        events: list[GestureEvent],
        current_mode: int,
        mode_count: int,
        frozen: bool,
    ) -> HoldProgress | None:
        cfg = self._cfg
        still = float(np.linalg.norm(hand.palm_velocity)) < cfg.hold_max_speed
        st.held = st.held + dt if still else max(st.held - dt * cfg.hold_decay_rate, 0.0)
        if not st.armed or st.held <= 0.0:
            return None

        cooling = self.cooling_down(now)
        anchor = _ring_anchor(hand, st.pose)
        pose = st.pose
        if pose is Pose.OPEN:
            freeze_label = "descongelar" if frozen else "congelar"
            mode5 = mode_count > 4 and current_mode != 4
            stage = cfg.count_hold_s / cfg.freeze_hold_s if mode5 else None
            st.pending_mode = mode5 and st.held >= cfg.count_hold_s
            if st.held >= cfg.freeze_hold_s and not cooling:
                events.append(GestureEvent(GestureKind.TOGGLE_FREEZE, hand.slot, hand.palm))
                self._fired(st, now)
                return None
            if st.pending_mode:
                labels: tuple[str, ...] = ("solte: modo 5", f"segure: {freeze_label}")
            elif mode5:
                labels = ("modo 5", freeze_label)
            else:
                labels = (freeze_label,)
            progress = min(st.held / cfg.freeze_hold_s, 1.0)
            return HoldProgress(hand.slot, anchor, progress, labels, stage, blocked=progress >= 1.0 and cooling)

        if pose in COUNT_POSES:
            mode = COUNT_POSES[pose] - 1
            if mode >= mode_count or mode == current_mode:
                return None
            duration, kind, label = cfg.count_hold_s, GestureKind.SELECT_MODE, f"modo {mode + 1}"
        elif pose is Pose.HORNS:
            mode, duration, kind, label = None, cfg.horns_hold_s, GestureKind.CYCLE_PALETTE, "paleta"
        elif pose is Pose.FIST:
            mode, duration, kind, label = None, cfg.fist_hold_s, GestureKind.EXPLODE, "explodir"
        else:
            return None

        progress = min(st.held / duration, 1.0)
        if progress >= 1.0 and not cooling:
            position = hand.palm if kind is GestureKind.EXPLODE else anchor
            events.append(GestureEvent(kind, hand.slot, position, mode=mode))
            self._fired(st, now)
            return None
        return HoldProgress(hand.slot, anchor, progress, (label,), blocked=progress >= 1.0 and cooling)

    def _two_hand_intensity(self, hands: Sequence[HandState]) -> float | None:
        if len(hands) < 2:
            return None
        cfg = self._cfg
        distance = float(np.linalg.norm(hands[0].palm - hands[1].palm))
        t = min(max((distance - cfg.two_hand_near) / (cfg.two_hand_far - cfg.two_hand_near), 0.0), 1.0)
        return cfg.intensity_near + (cfg.intensity_far - cfg.intensity_near) * t
