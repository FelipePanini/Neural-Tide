"""Owns the effect instances and cross-fades between them."""

from __future__ import annotations

from collections.abc import Sequence

from config import ModesConfig
from core.easing import damp
from core.hand_state import HandState
from effects.base import Effect, EffectEnv


class ModeManager:
    """Keeps every mode alive; switching eases opacities instead of cutting.

    While a mode fades out it keeps updating and drawing, so the outgoing and
    incoming effects overlap in the shared trail buffer.
    """

    def __init__(self, env: EffectEnv, effect_types: Sequence[type[Effect]], cfg: ModesConfig) -> None:
        self._cfg = cfg
        self._effects = [effect_type(env) for effect_type in effect_types]
        self._active = min(cfg.initial, len(self._effects) - 1)
        if self._effects:
            self._effects[self._active].enter()

    @property
    def count(self) -> int:
        return len(self._effects)

    @property
    def active_index(self) -> int:
        return self._active

    @property
    def active(self) -> Effect | None:
        return self._effects[self._active] if self._effects else None

    def select(self, index: int) -> bool:
        """Switch to mode ``index`` (0-based). Returns False if it is invalid or already active."""
        if not 0 <= index < len(self._effects) or index == self._active:
            return False
        self._effects[self._active].exit()
        self._active = index
        self._effects[index].enter()
        return True

    def _visible(self, index: int, effect: Effect) -> bool:
        return index == self._active or effect.opacity > self._cfg.hidden_opacity

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        for i, effect in enumerate(self._effects):
            target = 1.0 if i == self._active else 0.0
            effect.opacity = damp(effect.opacity, target, self._cfg.crossfade_tau_s, dt)
            if self._visible(i, effect):
                effect.update(dt, hands)

    def draw(self) -> None:
        for i, effect in enumerate(self._effects):
            if self._visible(i, effect):
                effect.draw()

    def release(self) -> None:
        for effect in self._effects:
            effect.release()
