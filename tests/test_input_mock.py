"""Mouse-driven mock hand: finger controls and scrolling."""

from __future__ import annotations

from config import CONFIG
from core.input_mock import MockHandSource

STEP = CONFIG.mock.wheel_step
COOLDOWN = CONFIG.mock.wheel_cooldown_s


def _settle(mock: MockHandSource, seconds: float = 1.0, dt: float = 1 / 60) -> None:
    """Advance the mock with a clock that continues from its last frame."""
    t = mock.latest().timestamp
    end = t + seconds
    while t < end:
        t += dt
        mock.update(t, dt, image_aspect=16 / 9)


def test_fist_and_finger_count_reach_the_tracker() -> None:
    mock = MockHandSource(CONFIG)
    mock.set_pointer((0.5, 0.5))
    mock.change_fingers(-3)
    _settle(mock)
    (hand,) = mock.latest().hands
    assert hand.finger_count == 2
    mock.set_fist(True)
    _settle(mock)
    assert mock.latest().hands[0].finger_count == 0


def test_scroll_steps_once_per_notch_and_respects_cooldown() -> None:
    mock = MockHandSource(CONFIG)
    start = mock.finger_count
    mock.scroll(-STEP, now=0.0)
    assert mock.finger_count == start - 1
    mock.scroll(-STEP, now=COOLDOWN / 2)  # trackpad burst inside the cooldown is ignored
    assert mock.finger_count == start - 1
    mock.scroll(-STEP, now=COOLDOWN * 2)
    assert mock.finger_count == start - 2


def test_small_trackpad_deltas_accumulate() -> None:
    mock = MockHandSource(CONFIG)
    mock.change_fingers(-5)
    for i in range(4):
        mock.scroll(STEP / 4 + 1e-6, now=float(i) * 0.01)
    assert mock.finger_count == 1


def test_finger_count_is_clamped() -> None:
    mock = MockHandSource(CONFIG)
    mock.change_fingers(10)
    assert mock.finger_count == 5
    mock.change_fingers(-10)
    assert mock.finger_count == 0


def test_pointer_outside_window_removes_the_hand_after_timeout() -> None:
    mock = MockHandSource(CONFIG)
    mock.set_pointer((0.5, 0.5))
    _settle(mock, 0.2)
    mock.set_pointer(None)
    mock.update(mock.latest().timestamp + 10.0, 1 / 60, image_aspect=16 / 9)
    assert mock.latest().hands == ()
