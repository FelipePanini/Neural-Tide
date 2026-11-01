"""Finger counting and pinch detection on synthetic landmarks."""

from __future__ import annotations

import numpy as np
import pytest

from config import CONFIG
from core.hand import LEFT, RIGHT, fingers_extended, pinch_strength
from core.input_mock import FINGER_PRESETS, HORNS_PRESET, synthetic_hand

GEOMETRY = CONFIG.geometry


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _expected(curls: tuple[float, ...]) -> tuple[bool, ...]:
    return tuple(c < 0.5 for c in curls)


@pytest.mark.parametrize("count", range(6))
@pytest.mark.parametrize("handedness", [LEFT, RIGHT])
def test_counts_raised_fingers(count: int, handedness: str) -> None:
    hand = synthetic_hand((0.5, 0.5), FINGER_PRESETS[count], handedness=handedness)
    fingers = fingers_extended(hand.world, GEOMETRY)
    assert fingers == _expected(FINGER_PRESETS[count])
    assert sum(fingers) == count


@pytest.mark.parametrize("count", range(6))
def test_count_is_rotation_invariant_and_noise_tolerant(count: int) -> None:
    rng = np.random.default_rng(count)
    for _ in range(25):
        hand = synthetic_hand(
            (0.5, 0.5), FINGER_PRESETS[count], roll_deg=rng.uniform(-60, 60), noise_m=0.0015, rng=rng
        )
        world = hand.world @ _random_rotation(rng).T
        assert sum(fingers_extended(world, GEOMETRY)) == count


def test_horns_pattern() -> None:
    hand = synthetic_hand((0.5, 0.5), HORNS_PRESET)
    assert fingers_extended(hand.world, GEOMETRY) == (False, True, False, False, True)


def test_half_curled_finger_is_not_extended() -> None:
    curls = (0.0, 0.0, 0.0, 0.0, 0.8)
    hand = synthetic_hand((0.5, 0.5), curls)
    assert fingers_extended(hand.world, GEOMETRY)[4] is False


@pytest.mark.parametrize(("pinch", "low", "high"), [(0.0, 0.0, 0.05), (1.0, 0.95, 1.0)])
def test_pinch_strength_limits(pinch: float, low: float, high: float) -> None:
    hand = synthetic_hand((0.5, 0.5), FINGER_PRESETS[5], pinch=pinch)
    assert low <= pinch_strength(hand.world, GEOMETRY) <= high


def test_pinch_strength_grows_monotonically() -> None:
    values = [
        pinch_strength(synthetic_hand((0.5, 0.5), FINGER_PRESETS[5], pinch=p).world, GEOMETRY)
        for p in np.linspace(0.0, 1.0, 11)
    ]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))
