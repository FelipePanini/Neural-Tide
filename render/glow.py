"""Additive soft light blobs (instanced quads with a gaussian falloff)."""

from __future__ import annotations

import moderngl
import numpy as np

from core.viewport import Viewport
from render.shader_library import ShaderLibrary

_FLOATS = 6  # center(2) radius(1) color(3)
_UNIT_QUAD = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)


class GlowBatch:
    """Collect glows during a frame and draw them in one instanced call."""

    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary, capacity: int) -> None:
        self._capacity = capacity
        self._prog = shaders.program("glow")
        self._quad = ctx.buffer(_UNIT_QUAD.tobytes())
        self._instances = ctx.buffer(reserve=capacity * _FLOATS * 4, dynamic=True)
        self._vao = ctx.vertex_array(
            self._prog,
            [
                (self._quad, "2f", "in_corner"),
                (self._instances, "2f 1f 3f/i", "in_center", "in_radius", "in_color"),
            ],
        )
        self._items: list[np.ndarray] = []

    def add(self, centers: np.ndarray, radius: float | np.ndarray, color: np.ndarray | tuple[float, ...]) -> None:
        """Glows at ``centers`` (N, 2); ``color`` already includes the intensity."""
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        data = np.empty((len(centers), _FLOATS), dtype=np.float32)
        data[:, :2] = centers
        data[:, 2] = radius
        data[:, 3:] = color
        self._items.append(data)

    def flush(self, viewport: Viewport) -> None:
        """Draw with the current blend state (the effect layer uses additive)."""
        if not self._items:
            return
        data = np.concatenate(self._items)[: self._capacity]
        self._items.clear()
        self._instances.write(data.tobytes())
        self._prog["u_aspect"].value = viewport.aspect
        self._vao.render(moderngl.TRIANGLE_STRIP, instances=len(data))

    def release(self) -> None:
        for obj in (self._vao, self._instances, self._quad):
            obj.release()
