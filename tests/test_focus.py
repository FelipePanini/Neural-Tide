"""Hand spotlights that keep the effects around the hands."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import CONFIG
from core.input_mock import FINGER_PRESETS
from render.focus import FocusState, hand_spot
from tests.test_gestures import make_hand

CFG = CONFIG.focus
DT = 1 / 60
HALF_WIDTH = 16 / 9


def test_spot_sits_on_the_hand_and_follows_its_presence() -> None:
    hand = replace(make_hand(FINGER_PRESETS[5]), presence=0.6)
    spot = hand_spot(hand, CFG)
    np.testing.assert_allclose(spot.center, hand.points.mean(axis=0))
    assert spot.weight == 0.6
    assert CFG.min_radius <= spot.radius <= CFG.max_radius


def test_spot_radius_grows_with_the_mode_scale() -> None:
    hand = make_hand(FINGER_PRESETS[0])  # a small fist still gets at least min_radius
    assert hand_spot(hand, CFG, scale=1.6).radius == hand_spot(hand, CFG).radius * 1.6


def test_idle_spotlight_appears_without_hands_and_yields_to_them() -> None:
    focus = FocusState(CFG)
    (idle,) = focus.update([], DT, HALF_WIDTH)
    assert idle.weight == CFG.idle_brightness
    for _ in range(300):
        spots = focus.update([make_hand(FINGER_PRESETS[5])], DT, HALF_WIDTH)
    assert len(spots) == 1 and spots[0].weight == 1.0  # the idle spot faded away


def test_toggle_eases_instead_of_cutting() -> None:
    focus = FocusState(CFG)
    assert not focus.toggle()
    focus.update([], DT, HALF_WIDTH)
    assert 0.9 < focus.amount < 1.0
    for _ in range(300):
        focus.update([], DT, HALF_WIDTH)
    assert focus.amount < 0.01


def test_never_more_spots_than_the_shader_handles() -> None:
    hands = [replace(make_hand(FINGER_PRESETS[5], (0.2 * i, 0.5)), slot=i) for i in range(6)]
    assert len(FocusState(CFG).update(hands, DT, HALF_WIDTH)) == CFG.max_spots
