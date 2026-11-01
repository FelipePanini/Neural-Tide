"""Base class for the visual modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import moderngl

from config import Config
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
    out; ``release()`` on shutdown.
    """

    name: ClassVar[str] = "efeito"

    def __init__(self, env: EffectEnv) -> None:
        self.env = env
        self.opacity = 0.0

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

    def debug_lines(self) -> list[str]:
        """Extra lines for the debug overlay (particle counts, timings...)."""
        return []

    def release(self) -> None:
        """Free GPU resources."""
