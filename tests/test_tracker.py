"""Slot assignment, stale handling and handedness labels of the hand tracker."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from config import CONFIG
from core.hand import LEFT, RIGHT
from core.input_mock import FINGER_PRESETS, synthetic_hand
from core.tracker import HandFilterBank, needs_label_swap, observations_from_result

OPEN = FINGER_PRESETS[5]
DT = 1.0 / 30.0


def _bank() -> HandFilterBank:
    return HandFilterBank(CONFIG.tracker, CONFIG.geometry)


def test_slots_follow_hands_even_when_detection_order_flips() -> None:
    bank = _bank()
    left = synthetic_hand((0.3, 0.5), OPEN, handedness=LEFT)
    right = synthetic_hand((0.7, 0.5), OPEN, handedness=RIGHT)
    first = {h.handedness: h.slot for h in bank.update([left, right], 0.0)}
    for i in range(1, 10):
        x = 0.01 * i
        moved = [synthetic_hand((0.7 + x, 0.5), OPEN), synthetic_hand((0.3 - x, 0.5), OPEN, handedness=LEFT)]
        hands = bank.update(moved if i % 2 else moved[::-1], i * DT)
        assert {h.handedness: h.slot for h in hands} == first


def test_short_dropout_is_stale_then_expires() -> None:
    bank = _bank()
    bank.update([synthetic_hand((0.5, 0.5), OPEN)], 0.0)
    (held,) = bank.update([], DT)
    assert held.stale
    assert bank.update([], CONFIG.tracker.lost_timeout_s + DT) == ()


def test_hand_returning_after_timeout_gets_a_new_slot() -> None:
    bank = _bank()
    (first,) = bank.update([synthetic_hand((0.5, 0.5), OPEN)], 0.0)
    bank.update([], CONFIG.tracker.lost_timeout_s + DT)
    (second,) = bank.update([synthetic_hand((0.5, 0.5), OPEN)], 1.0)
    assert second.slot != first.slot


def test_distant_detection_starts_a_new_slot() -> None:
    bank = _bank()
    (a,) = bank.update([synthetic_hand((0.2, 0.5), OPEN)], 0.0)
    hands = bank.update([synthetic_hand((0.9, 0.5), OPEN)], DT)
    assert {h.slot for h in hands if not h.stale} != {a.slot}


def test_tracked_hand_reports_fingers_and_pinch() -> None:
    bank = _bank()
    (hand,) = bank.update([synthetic_hand((0.5, 0.5), FINGER_PRESETS[2])], 0.0)
    assert hand.fingers == (False, True, True, False, False)
    assert hand.pinch < 0.05


def _fake_result(label: str) -> SimpleNamespace:
    obs = synthetic_hand((0.5, 0.5), OPEN)
    point = lambda p: SimpleNamespace(x=p[0], y=p[1], z=p[2])  # noqa: E731
    return SimpleNamespace(
        hand_landmarks=[[point(p) for p in obs.landmarks]],
        hand_world_landmarks=[[point(p) for p in obs.world]],
        handedness=[[SimpleNamespace(category_name=label, score=0.9)]],
    )


def test_label_swap_rule() -> None:
    assert needs_label_swap(image_mirrored=True, model_expects_mirrored=False)
    assert not needs_label_swap(image_mirrored=False, model_expects_mirrored=False)
    assert not needs_label_swap(image_mirrored=True, model_expects_mirrored=True)


def test_observations_from_result_converts_and_swaps_labels() -> None:
    (obs,) = observations_from_result(_fake_result(RIGHT), swap_labels=True)
    assert obs.handedness == LEFT
    assert obs.landmarks.shape == (21, 3) and obs.world.shape == (21, 3)
    (kept,) = observations_from_result(_fake_result(RIGHT), swap_labels=False)
    assert kept.handedness == RIGHT
    np.testing.assert_allclose(kept.landmarks, obs.landmarks)
