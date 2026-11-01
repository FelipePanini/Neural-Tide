"""Text sprites: a fixed set of strings rendered once into one texture, drawn instanced."""

from __future__ import annotations

from collections.abc import Sequence

import moderngl
import numpy as np
import pygame

from core.viewport import Viewport
from render.shader_library import ShaderLibrary

_UNIT_QUAD = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
_FLOATS = 12  # center(2) half size(2) uv rect(4) colour(3) angle(1)


class GlyphAtlas:
    """White-on-transparent glyphs or words packed into shelves of one texture."""

    def __init__(
        self,
        ctx: moderngl.Context,
        texts: Sequence[str],
        font_names: str,
        size: int,
        width: int = 1024,
        padding: int = 2,
    ) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.SysFont(font_names, size)
        surfaces = [font.render(text, True, (255, 255, 255)) for text in texts]
        positions: list[tuple[int, int]] = []
        x = y = row = 0
        for surface in surfaces:
            w, h = surface.get_size()
            if x + w + padding > width:
                x, y, row = 0, y + row + padding, 0
            positions.append((x, y))
            x += w + padding
            row = max(row, h)
        height = 1 << max(int(np.ceil(np.log2(max(y + row, 1)))), 0)
        sheet = pygame.Surface((width, height), pygame.SRCALPHA)
        for surface, pos in zip(surfaces, positions, strict=True):
            sheet.blit(surface, pos)
        self.texture = ctx.texture((width, height), 4, pygame.image.tobytes(sheet, "RGBA", True))
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texts = list(texts)
        self.sizes_px = np.array([s.get_size() for s in surfaces], dtype=np.float32)
        rects = []
        for (px, py), (w, h) in zip(positions, self.sizes_px, strict=True):
            # The sheet was flipped on upload, so image row y sits at v = 1 - y / height.
            rects.append((px / width, 1.0 - (py + h) / height, (px + w) / width, 1.0 - py / height))
        self.uv_rects = np.array(rects, dtype=np.float32)

    def release(self) -> None:
        self.texture.release()


class SpriteBatch:
    """Instanced, tinted text quads drawn with the current (additive) blend state."""

    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary, atlas: GlyphAtlas, capacity: int) -> None:
        self._atlas = atlas
        self._capacity = capacity
        self._prog = shaders.program("sprite")
        self._quad = ctx.buffer(_UNIT_QUAD.tobytes())
        self._instances = ctx.buffer(reserve=capacity * _FLOATS * 4, dynamic=True)
        self._vao = ctx.vertex_array(
            self._prog,
            [
                (self._quad, "2f", "in_corner"),
                (self._instances, "2f 2f 4f 3f 1f/i", "in_center", "in_half_size", "in_uv", "in_color", "in_angle"),
            ],
        )
        self._items: list[np.ndarray] = []

    def add(
        self,
        centers: np.ndarray,
        indices: np.ndarray,
        world_per_px: float | np.ndarray,
        colors: np.ndarray,
        angles: float | np.ndarray = 0.0,
    ) -> None:
        """Sprites of ``atlas.texts[indices]`` at ``centers``; colours include the intensity."""
        indices = np.asarray(indices, dtype=np.intp)
        data = np.empty((len(indices), _FLOATS), dtype=np.float32)
        data[:, 0:2] = centers
        data[:, 2:4] = self._atlas.sizes_px[indices] * 0.5 * np.asarray(world_per_px, dtype=np.float32).reshape(-1, 1)
        data[:, 4:8] = self._atlas.uv_rects[indices]
        data[:, 8:11] = colors
        data[:, 11] = angles
        self._items.append(data)

    def flush(self, viewport: Viewport) -> None:
        if not self._items:
            return
        data = np.concatenate(self._items)[: self._capacity]
        self._items.clear()
        self._instances.write(data.tobytes())
        self._prog["u_aspect"].value = viewport.aspect
        self._atlas.texture.use(0)
        self._vao.render(moderngl.TRIANGLE_STRIP, instances=len(data))

    def release(self) -> None:
        for obj in (self._vao, self._instances, self._quad):
            obj.release()
