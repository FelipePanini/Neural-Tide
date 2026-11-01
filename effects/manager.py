"""Owns the effect instances and cross-fades between them."""

from __future__ import annotations

from collections.abc import Sequence

from config import ModesConfig
from core.easing import damp
from core.gestures import GestureEvent
from core.hand_state import HandState
from effects.base import Effect, EffectEnv


class ModeManager:
    """Keeps every mode alive; switching eases opacities instead of cutting.

    While a mode fades out it keeps updating and drawing, so the outgoing and
    incoming effects overlap in the shared trail buffer. Freezing eases the
    simulation time scale to zero rather than stopping dead.
    """

    def __init__(self, env: EffectEnv, effect_types: Sequence[type[Effect]], cfg: ModesConfig) -> None:
        self._cfg = cfg
        self._effects = [effect_type(env) for effect_type in effect_types]
        self._active = min(cfg.initial, len(self._effects) - 1)
        self._frozen = False
        self.time_scale = 1.0
        self.intensity = cfg.neutral_intensity
        self._intensity_target: float | None = None
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

    @property
    def frozen(self) -> bool:
        return self._frozen

    def names(self) -> list[str]:
        return [effect.name for effect in self._effects]

    def toggle_freeze(self) -> bool:
        self._frozen = not self._frozen
        return self._frozen

    def set_intensity_target(self, value: float | None) -> None:
        """Intensity requested by the two-hand distance; ``None`` drifts back to neutral."""
        self._intensity_target = value

    def dispatch(self, event: GestureEvent) -> None:
        """Hand a gesture event to the active mode."""
        if self.active is not None:
            self.active.on_gesture(event)

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
        cfg = self._cfg
        self.time_scale = damp(self.time_scale, 0.0 if self._frozen else 1.0, cfg.freeze_tau_s, dt)
        if self._intensity_target is None:
            self.intensity = damp(self.intensity, cfg.neutral_intensity, cfg.intensity_release_tau_s, dt)
        else:
            self.intensity = damp(self.intensity, self._intensity_target, cfg.intensity_tau_s, dt)
        for i, effect in enumerate(self._effects):
            target = 1.0 if i == self._active else 0.0
            effect.opacity = damp(effect.opacity, target, cfg.crossfade_tau_s, dt)
            effect.intensity = self.intensity
            if self._visible(i, effect):
                effect.update(dt * self.time_scale, hands)

    def trail(self, default: tuple[float, float]) -> tuple[float, float]:
        """Trail settings for this frame, blended by the opacity of the visible modes."""
        weights = [(e.opacity, e.trail()) for i, e in enumerate(self._effects) if self._visible(i, e)]
        total = sum(w for w, _ in weights)
        if total <= 1e-6:
            return default
        tau = sum(w * t for w, (t, _) in weights) / total
        diffusion = sum(w * d for w, (_, d) in weights) / total
        return tau, diffusion

    def focus_scale(self) -> float:
        """Spotlight size for this frame, blended by the opacity of the visible modes."""
        weights = [(e.opacity, e.focus_scale()) for i, e in enumerate(self._effects) if self._visible(i, e)]
        total = sum(w for w, _ in weights)
        return sum(w * f for w, f in weights) / total if total > 1e-6 else 1.0

    def draw(self, frame_gain: float = 1.0) -> None:
        for i, effect in enumerate(self._effects):
            if self._visible(i, effect):
                effect.frame_gain = frame_gain
                effect.draw()

    def release(self) -> None:
        for effect in self._effects:
            effect.release()
