"""Text via pygame.font, cached as GPU textures and drawn as screen-space quads."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import moderngl
import numpy as np
import pygame

from config import RGBA8, DebugConfig
from render.shader_library import ShaderLibrary

TextLine = tuple[str, RGBA8]


class TextRenderer:
    """Renders blocks of text once and reuses the texture while the text is unchanged."""

    def __init__(self, ctx: moderngl.Context, shaders: ShaderLibrary, cfg: DebugConfig) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        self._ctx = ctx
        self._cfg = cfg
        self._prog = shaders.program("text")
        self._buf = ctx.buffer(reserve=4 * 4 * 4, dynamic=True)
        self._vao = ctx.vertex_array(self._prog, [(self._buf, "2f 2f", "in_pos", "in_uv")])
        self._fonts: dict[int, pygame.font.Font] = {}
        self._cache: OrderedDict[tuple, moderngl.Texture] = OrderedDict()

    def font(self, size: int) -> pygame.font.Font:
        if size not in self._fonts:
            self._fonts[size] = pygame.font.SysFont(self._cfg.font_names, size)
        return self._fonts[size]

    def texture(
        self, lines: Sequence[TextLine], size: int, *, background: RGBA8 | None = None, padding: int = 0
    ) -> moderngl.Texture:
        """Texture holding ``lines`` (text, color) stacked vertically."""
        key = (tuple(lines), size, background, padding)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        font = self.font(size)
        rendered = [font.render(text or " ", True, color) for text, color in lines]
        spacing = self._cfg.line_spacing_px
        width = max(s.get_width() for s in rendered) + 2 * padding
        height = sum(s.get_height() for s in rendered) + spacing * (len(rendered) - 1) + 2 * padding
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill(background or (0, 0, 0, 0))
        y = padding
        for line in rendered:
            surface.blit(line, (padding, y))
            y += line.get_height() + spacing

        tex = self._ctx.texture((width, height), 4, pygame.image.tobytes(surface, "RGBA", True))
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._cache[key] = tex
        while len(self._cache) > self._cfg.text_cache_size:
            self._cache.popitem(last=False)[1].release()
        return tex

    def draw(
        self,
        tex: moderngl.Texture,
        pos_px: tuple[float, float],
        screen: tuple[int, int],
        *,
        anchor: tuple[float, float] = (0.0, 0.0),
        opacity: float = 1.0,
    ) -> None:
        """Draw ``tex`` at ``pos_px``; ``anchor`` (0..1, 0..1) picks which point of the box sits there."""
        w, h = tex.size
        x0 = round(pos_px[0] - w * anchor[0])  # whole pixels keep NEAREST-filtered text crisp
        y0 = round(pos_px[1] - h * anchor[1])
        quad = np.array(
            [(x0, y0, 0, 1), (x0 + w, y0, 1, 1), (x0, y0 + h, 0, 0), (x0 + w, y0 + h, 1, 0)], dtype=np.float32
        )
        self._buf.write(quad.tobytes())
        self._prog["u_screen"].value = screen
        self._prog["u_opacity"].value = opacity
        tex.use(0)
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def release(self) -> None:
        for tex in self._cache.values():
            tex.release()
        self._cache.clear()
        self._vao.release()
        self._buf.release()
