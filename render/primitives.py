"""Immediate-mode batch of soft points and anti-aliased thick lines in world space."""

from __future__ import annotations

from collections.abc import Sequence

import moderngl
import numpy as np

from config import RGBA, RenderConfig
from core.viewport import Viewport
from render.shader_library import ShaderLibrary

_POINT_FLOATS = 7  # pos(2) size(1) color(4)
_LINE_FLOATS = 8  # pos(2) color(4) edge(1) feather(1)
_VERTS_PER_SEGMENT = 6
_F32 = 4

Color = RGBA | Sequence[float] | np.ndarray


class PrimitiveBatch:
    """Collect points and lines during a frame, then draw them in two calls."""

    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary, cfg: RenderConfig) -> None:
        self._cfg = cfg
        self._points_prog = shaders.program("points")
        self._lines_prog = shaders.program("lines")
        self._point_buf = ctx.buffer(reserve=cfg.max_points * _POINT_FLOATS * _F32, dynamic=True)
        self._line_buf = ctx.buffer(
            reserve=cfg.max_line_segments * _VERTS_PER_SEGMENT * _LINE_FLOATS * _F32, dynamic=True
        )
        self._point_vao = ctx.vertex_array(
            self._points_prog, [(self._point_buf, "2f 1f 4f", "in_pos", "in_size", "in_color")]
        )
        self._line_vao = ctx.vertex_array(
            self._lines_prog, [(self._line_buf, "2f 4f 1f 1f", "in_pos", "in_color", "in_edge", "in_feather")]
        )
        self._points: list[np.ndarray] = []
        self._lines: list[np.ndarray] = []

    def add_points(self, xy: np.ndarray, size_px: float | np.ndarray, color: Color) -> None:
        """Points ``(N, 2)``; size and color may be shared or per point."""
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
        data = np.empty((len(xy), _POINT_FLOATS), dtype=np.float32)
        data[:, :2] = xy
        data[:, 2] = size_px
        data[:, 3:] = color
        self._points.append(data)

    def add_lines(
        self, starts: np.ndarray, ends: np.ndarray, width_px: float, color: Color, viewport: Viewport
    ) -> None:
        """Segments from ``starts[i]`` to ``ends[i]`` (both ``(N, 2)``), ``width_px`` wide."""
        a = np.asarray(starts, dtype=np.float32).reshape(-1, 2)
        b = np.asarray(ends, dtype=np.float32).reshape(-1, 2)
        total_px = width_px + self._cfg.line_feather_px
        half = viewport.pixels_to_world(total_px / 2.0)
        d = b - a
        length = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        normal = np.stack((-d[:, 1], d[:, 0]), axis=1) / length * half

        corners = np.stack((a + normal, a - normal, b + normal, b + normal, a - normal, b - normal), axis=1)
        data = np.empty((len(a), _VERTS_PER_SEGMENT, _LINE_FLOATS), dtype=np.float32)
        data[:, :, :2] = corners
        data[:, :, 2:6] = np.broadcast_to(np.asarray(color, dtype=np.float32), (len(a), 4))[:, None, :]
        data[:, :, 6] = (1.0, -1.0, 1.0, 1.0, -1.0, -1.0)
        data[:, :, 7] = self._cfg.line_feather_px / total_px
        self._lines.append(data.reshape(-1, _LINE_FLOATS))

    def flush(self, viewport: Viewport, premultiplied: bool = False) -> None:
        """Draw everything collected since the last flush (lines below points).

        ``premultiplied`` outputs colour × alpha, for additive blending in the effect layer.
        """
        for prog in (self._points_prog, self._lines_prog):
            prog["u_aspect"].value = viewport.aspect
            prog["u_premultiply"].value = int(premultiplied)
        if self._lines:
            verts = np.concatenate(self._lines)[: self._cfg.max_line_segments * _VERTS_PER_SEGMENT]
            self._line_buf.write(verts.tobytes())
            self._line_vao.render(moderngl.TRIANGLES, vertices=len(verts))
        if self._points:
            pts = np.concatenate(self._points)[: self._cfg.max_points]
            self._point_buf.write(pts.tobytes())
            self._point_vao.render(moderngl.POINTS, vertices=len(pts))
        self._points.clear()
        self._lines.clear()

    def release(self) -> None:
        for obj in (self._point_vao, self._line_vao, self._point_buf, self._line_buf):
            obj.release()
