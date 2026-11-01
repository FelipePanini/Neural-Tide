"""Base class for the visual modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import moderngl

from config import Config
from core.gestures import GestureEvent
from core.hand_state import HandState
from core.viewport import Viewport
from render.palette import PaletteState
from render.shader_library import ShaderLibrary


@dataclass
class EffectEnv:
    """GPU handles and settings every effect receives."""

    ctx: moderngl.Context
    shaders: ShaderLibrary
    config: Config
    viewport: Viewport  # shared with the renderer, so it follows window resizes
    palette: PaletteState


class Effect(ABC):
    """One visual mode. Register subclasses in ``effects.EFFECTS``.

    Lifecycle: ``enter()`` when the mode is selected; ``update()`` and
    ``draw()`` every frame while it is visible, including cross-fades, during
    which ``opacity`` ramps between 0 and 1; ``exit()`` when it starts fading
    out; ``release()`` on shutdown. ``intensity`` (eased, 1 = neutral) follows
    the distance between two hands; ``on_gesture()`` receives pinch and
    explosion events while the mode is active.
    """

    name: ClassVar[str] = "efeito"

    def __init__(self, env: EffectEnv) -> None:
        self.env = env
        self.opacity = 0.0
        self.intensity = 1.0
        self.frame_gain = 1.0  # set by the mode manager every frame (see PostFX.begin)

    @property
    def visibility(self) -> float:
        """Brightness factor for drawing: cross-fade opacity × frame-rate compensation."""
        return self.opacity * self.frame_gain

    def trail(self) -> tuple[float, float]:
        """(trail time constant, diffusion per second) this mode wants for the effect layer."""
        cfg = self.env.config.postfx
        return cfg.trail_tau_s, cfg.trail_diffusion_per_s

    def enter(self) -> None:
        """Called when the mode becomes active."""

    def exit(self) -> None:
        """Called when the mode starts fading out."""

    @abstractmethod
    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        """Advance the simulation by ``dt`` seconds; hands may be empty (idle)."""

    @abstractmethod
    def draw(self) -> None:
        """Draw into the bound framebuffer (the renderer sets additive blending)."""

    def on_gesture(self, event: GestureEvent) -> None:
        """React to PINCH_START / PINCH_END / EXPLODE (default: ignore)."""

    def debug_lines(self) -> list[str]:
        """Extra lines for the debug overlay (particle counts, timings...)."""
        return []

    def release(self) -> None:
        """Free GPU resources."""
