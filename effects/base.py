"""Base class for the visual modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import moderngl
import numpy as np

from config import Config
from core.gestures import GestureEvent
from core.hand_state import HandState
from core.viewport import Viewport
from render.focus import Spot
from render.palette import PaletteState
from render.shader_library import ShaderLibrary


def _no_segmentation() -> np.ndarray | None:
    return None


@dataclass
class EffectEnv:
    """GPU handles and settings every effect receives."""

    ctx: moderngl.Context
    shaders: ShaderLibrary
    config: Config
    viewport: Viewport  # shared with the renderer, so it follows window resizes
    palette: PaletteState
    # Latest person mask (H, W) in image space, 0..1, when a mode asked for it.
    segmentation: Callable[[], np.ndarray | None] = _no_segmentation


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
    needs_segmentation: ClassVar[bool] = False  # turns on person segmentation while visible

    def __init__(self, env: EffectEnv) -> None:
        self.env = env
        self.opacity = 0.0
        self.intensity = 1.0
        self.frame_gain = 1.0  # set by the mode manager every frame (see PostFX.begin)

    @property
    def visibility(self) -> float:
        """Brightness factor for drawing: cross-fade opacity × frame-rate compensation."""
        return self.opacity * self.frame_gain

    def focus_scale(self) -> float:
        """Size of the hand spotlight for this mode (1 = default)."""
        return 1.0

    def focus_strength(self) -> float:
        """How much the hand spotlight masks this mode (0 = full screen)."""
        return 1.0

    def focus_spots(self) -> list[Spot]:
        """Extra spotlights this mode wants lit (e.g. something between the hands)."""
        return []

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
