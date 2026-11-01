"""Gesture state machine, driven by synthetic landmarks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from config import CONFIG
from core.gestures import GestureEngine, GestureEvent, GestureKind, HoldProgress, Pose, classify
from core.hand import INDEX_TIP, LEFT, RIGHT, fingers_extended, pinch_strength
from core.hand_state import HandState
from core.input_mock import FINGER_PRESETS, HORNS_PRESET, synthetic_hand
from core.viewport import Viewport

CFG = CONFIG.gestures
DT = 1 / 60
VIEWPORT = Viewport(1280, 720, 16 / 9)
HandsAt = Callable[[float], Sequence[HandState]]


def make_hand(
    curls: Sequence[float],
    center: tuple[float, float] = (0.5, 0.5),
    *,
    velocity: tuple[float, float] = (0.0, 0.0),
    pinch: float = 0.0,
    slot: int = 0,
    handedness: str = RIGHT,
) -> HandState:
    obs = synthetic_hand(center, curls, pinch=pinch, handedness=handedness, image_aspect=VIEWPORT.aspect)
    points = VIEWPORT.image_to_world(obs.landmarks)
    return HandState(
        slot=slot,
        handedness=handedness,
        points=points,
        raw_points=points,
        velocity=np.tile(np.array(velocity, dtype=np.float64), (21, 1)),
        depth=obs.landmarks[:, 2],
        world=obs.world,
        fingers=fingers_extended(obs.world, CONFIG.geometry),
        pinch=pinch_strength(obs.world, CONFIG.geometry),
        presence=1.0,
        visible=True,
    )


def run(
    engine: GestureEngine,
    hands_at: HandsAt,
    seconds: float,
    *,
    start: float = 0.0,
    current_mode: int = 0,
    mode_count: int = 5,
) -> tuple[list[tuple[float, GestureEvent]], list[HoldProgress]]:
    """Feed ``hands_at(t)`` for ``seconds``; returns timed events and the last frame's holds."""
    events: list[tuple[float, GestureEvent]] = []
    holds: tuple[HoldProgress, ...] = ()
    for i in range(round(seconds / DT)):
        t = start + i * DT
        frame = engine.update(hands_at(t), DT, t, current_mode=current_mode, mode_count=mode_count)
        events += [(t, e) for e in frame.events]
        holds = frame.holds
    return events, list(holds)


def kinds(events: list[tuple[float, GestureEvent]]) -> list[GestureKind]:
    return [e.kind for _, e in events]


def still(curls: Sequence[float], **kw: object) -> HandsAt:
    hand = make_hand(curls, **kw)  # type: ignore[arg-type]
    return lambda t: [hand]


def test_count_hold_selects_mode_once_after_half_a_second() -> None:
    events, _ = run(GestureEngine(CFG), still(FINGER_PRESETS[2]), 1.5)
    assert kinds(events) == [GestureKind.SELECT_MODE]
    t, event = events[0]
    assert event.mode == 1
    assert CFG.count_hold_s <= t <= CFG.count_hold_s + (CFG.debounce_frames + 2) * DT


def test_hold_needs_a_still_hand() -> None:
    events, holds = run(GestureEngine(CFG), still(FINGER_PRESETS[2], velocity=(1.0, 0.0)), 1.5)
    assert events == [] and holds == []


def test_debounce_ignores_flickering_poses() -> None:
    one, three = make_hand(FINGER_PRESETS[1]), make_hand(FINGER_PRESETS[3])
    events, _ = run(GestureEngine(CFG), lambda t: [one if round(t / DT) % 2 else three], 2.0)
    assert events == []


def test_hand_must_change_pose_before_firing_again() -> None:
    engine = GestureEngine(CFG)
    first, _ = run(engine, still(FINGER_PRESETS[2]), 2.0)
    gone, _ = run(engine, lambda t: [], 0.1, start=2.0)
    again, _ = run(engine, still(FINGER_PRESETS[2]), 1.0, start=2.1)
    assert len(first) == 1 and gone == [] and kinds(again) == [GestureKind.SELECT_MODE]


def test_current_or_missing_mode_shows_no_ring() -> None:
    for count, current, available in ((2, 1, 5), (4, 0, 3)):
        events, holds = run(GestureEngine(CFG), still(FINGER_PRESETS[count]), 1.0, current_mode=current, mode_count=available)
        assert events == [] and holds == []


def test_hold_progress_ring_sits_on_the_index_tip() -> None:
    hand = make_hand(FINGER_PRESETS[2])
    _, holds = run(GestureEngine(CFG), lambda t: [hand], 0.3)
    (hold,) = holds
    assert 0.0 < hold.progress < 1.0 and hold.labels == ("modo 2",)
    np.testing.assert_allclose(hold.position, hand.points[INDEX_TIP])


def test_open_palm_released_midway_selects_mode_5() -> None:
    engine = GestureEngine(CFG)
    held, holds = run(engine, still(FINGER_PRESETS[5]), 0.75)
    assert held == [] and holds[0].labels[0] == "solte: modo 5" and holds[0].marker == CFG.count_hold_s / CFG.freeze_hold_s
    released, _ = run(engine, lambda t: [], 0.1, start=0.75)
    assert [(e.kind, e.mode) for _, e in released] == [(GestureKind.SELECT_MODE, 4)]


def test_open_palm_held_long_toggles_freeze_instead() -> None:
    engine = GestureEngine(CFG)
    events, _ = run(engine, still(FINGER_PRESETS[5]), 1.5)
    released, _ = run(engine, lambda t: [], 0.1, start=1.5)
    assert kinds(events) == [GestureKind.TOGGLE_FREEZE] and released == []


def test_open_palm_without_a_mode_5_is_only_freeze() -> None:
    _, holds = run(GestureEngine(CFG), still(FINGER_PRESETS[5]), 0.3, mode_count=3)
    assert holds[0].labels == ("congelar",) and holds[0].marker is None


def test_horns_cycle_the_palette_instead_of_selecting_mode_2() -> None:
    events, _ = run(GestureEngine(CFG), still(HORNS_PRESET), 1.0)
    assert kinds(events) == [GestureKind.CYCLE_PALETTE]


def test_fist_explodes_at_the_palm() -> None:
    hand = make_hand(FINGER_PRESETS[0])
    events, _ = run(GestureEngine(CFG), lambda t: [hand], 1.0)
    assert kinds(events) == [GestureKind.EXPLODE]
    np.testing.assert_allclose(events[0][1].position, hand.palm)


def test_pinch_emits_start_and_end_and_never_counts_fingers() -> None:
    engine = GestureEngine(CFG)
    pinching, _ = run(engine, still(FINGER_PRESETS[5], pinch=1.0), 1.5)  # "OK" sign: 3 other fingers up
    released, _ = run(engine, still(FINGER_PRESETS[0]), 0.1, start=1.5)
    assert kinds(pinching) == [GestureKind.PINCH_START]
    assert GestureKind.PINCH_END in kinds(released)


def test_a_fist_is_never_a_pinch() -> None:
    fist = replace(make_hand(FINGER_PRESETS[0]), pinch=1.0)  # thumb resting near the index
    assert classify(fist, CFG) is Pose.FIST


def _sweep(vx_world: float, vy_world: float = 0.0) -> HandsAt:
    def hands_at(t: float) -> list[HandState]:
        center = (0.5 + vx_world * t / (2 * VIEWPORT.aspect), 0.5 - vy_world * t / 2)
        return [make_hand(FINGER_PRESETS[5], center, velocity=(vx_world, vy_world))]

    return hands_at


def test_swipe_left_is_next_and_right_is_previous() -> None:
    left, _ = run(GestureEngine(CFG), _sweep(-2.5), 0.4)
    right, _ = run(GestureEngine(CFG), _sweep(2.5), 0.4)
    assert kinds(left) == [GestureKind.NEXT_MODE]
    assert kinds(right) == [GestureKind.PREVIOUS_MODE]


def test_vertical_or_slow_motion_is_not_a_swipe() -> None:
    vertical, _ = run(GestureEngine(CFG), _sweep(0.3, 2.5), 0.4)
    slow, _ = run(GestureEngine(CFG), _sweep(-0.8), 1.0)
    assert vertical == [] and slow == []


def test_cooldown_delays_a_second_action() -> None:
    fist = make_hand(FINGER_PRESETS[0], (0.3, 0.5), slot=0)
    horns = make_hand(HORNS_PRESET, (0.7, 0.5), slot=1, handedness=LEFT)
    events, _ = run(GestureEngine(CFG), lambda t: [fist, horns], 2.0)
    assert kinds(events) == [GestureKind.EXPLODE, GestureKind.CYCLE_PALETTE]
    assert events[1][0] - events[0][0] >= CFG.cooldown_s - DT


def test_distance_between_two_hands_sets_intensity() -> None:
    engine = GestureEngine(CFG)
    values = []
    for spread in (0.02, 0.15, 0.3, 0.45):
        hands = [make_hand(FINGER_PRESETS[3], (0.5 - spread, 0.5), slot=0), make_hand(FINGER_PRESETS[3], (0.5 + spread, 0.5), slot=1)]
        values.append(engine.update(hands, DT, 0.0, current_mode=0, mode_count=5).intensity)
    assert values[0] == CFG.intensity_near and values[-1] == CFG.intensity_far
    assert values == sorted(values)
    assert engine.update([hands[0]], DT, 0.0, current_mode=0, mode_count=5).intensity is None
