"""Mode switching with cross-fades."""

from __future__ import annotations

from collections.abc import Sequence

from config import CONFIG
from core.hand_state import HandState
from effects.base import Effect, EffectEnv
from effects.manager import ModeManager

DT = 1 / 60


class _Recorder(Effect):
    name = "recorder"

    def __init__(self, env: EffectEnv) -> None:
        super().__init__(env)
        self.events: list[str] = []
        self.updates = 0

    def enter(self) -> None:
        self.events.append("enter")

    def exit(self) -> None:
        self.events.append("exit")

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        self.updates += 1

    def draw(self) -> None:
        pass


def _manager(count: int = 2) -> ModeManager:
    env = EffectEnv(None, None, CONFIG, None, None)  # type: ignore[arg-type]
    return ModeManager(env, [_Recorder] * count, CONFIG.modes)


def test_initial_mode_is_entered() -> None:
    modes = _manager()
    assert modes.active_index == 0
    assert modes.active.events == ["enter"]  # type: ignore[union-attr]


def test_select_rejects_invalid_or_current_index() -> None:
    modes = _manager()
    assert not modes.select(0)
    assert not modes.select(5)
    assert not modes.select(-1)
    assert modes.select(1)


def test_crossfade_eases_both_modes_and_stops_hidden_ones() -> None:
    modes = _manager()
    first, second = modes._effects  # noqa: SLF001
    for _ in range(300):
        modes.update(DT, [])
    assert first.opacity > 0.99
    modes.select(1)
    assert first.events == ["enter", "exit"] and second.events == ["enter"]
    modes.update(DT, [])
    assert 0.9 < first.opacity < 1.0 and 0.0 < second.opacity < 0.1  # no hard cut
    for _ in range(600):
        modes.update(DT, [])
    assert second.opacity > 0.99 and first.opacity < CONFIG.modes.hidden_opacity
    updates = first.updates
    modes.update(DT, [])
    assert first.updates == updates  # fully faded modes no longer simulate
