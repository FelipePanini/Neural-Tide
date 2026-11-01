"""Instanced particle drawing shared by the particle modes (soft velocity streaks)."""

from __future__ import annotations

import moderngl
import numpy as np

from config import ParticleLook
from core.viewport import Viewport
from render.palette import PaletteState
from render.shader_library import ShaderLibrary

_UNIT_QUAD = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
_FLOATS = 6  # pos(2) vel(2) alpha(1) colour seed(1)


class ParticleRenderer:
    """Draws up to ``capacity`` particles in one instanced call, additively."""

    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary, capacity: int) -> None:
        self._capacity = capacity
        self._prog = shaders.program("tide")
        self._quad = ctx.buffer(_UNIT_QUAD.tobytes())
        self._instances = ctx.buffer(reserve=capacity * _FLOATS * 4, dynamic=True)
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._quad, "2f", "in_corner"), (self._instances, "2f 2f 2f/i", "in_pos", "in_vel", "in_info")],
        )
        self._packed = np.empty((capacity, _FLOATS), dtype=np.float32)

    def draw(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        alpha: np.ndarray,
        hue: np.ndarray,
        look: ParticleLook,
        palette: PaletteState,
        viewport: Viewport,
        brightness: float,
    ) -> None:
        """Draw particles; ``brightness`` multiplies ``look.intensity`` (fade, frame gain...)."""
        n = min(len(pos), self._capacity)
        if n == 0:
            return
        packed = self._packed[:n]
        packed[:, 0:2] = pos[:n]
        packed[:, 2:4] = vel[:n]
        packed[:, 4] = alpha[:n]
        packed[:, 5] = hue[:n]
        self._instances.write(packed.tobytes())
        prog = self._prog
        prog["u_aspect"].value = viewport.aspect
        prog["u_px"].value = viewport.pixels_to_world(1.0)
        prog["u_size_px"].value = look.size_px
        prog["u_streak_s"].value = look.streak_s
        prog["u_max_streak_px"].value = look.max_streak_px
        prog["u_fast_speed"].value = look.fast_speed
        prog["u_intensity"].value = look.intensity * brightness
        prog["u_primary"].value = palette.primary
        prog["u_secondary"].value = palette.secondary
        prog["u_highlight"].value = palette.highlight
        prog["u_secondary_share"].value = look.secondary_share
        self._vao.render(moderngl.TRIANGLE_STRIP, instances=n)

    def release(self) -> None:
        for obj in (self._vao, self._instances, self._quad):
            obj.release()
