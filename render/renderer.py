"""Frame orchestration and the GPU state shared by every layer."""

from __future__ import annotations

import moderngl
import numpy as np

from config import Config
from core.viewport import Viewport
from render.postfx import PostFX
from render.primitives import PrimitiveBatch
from render.shader_library import ShaderLibrary
from render.text import TextRenderer

_FULLSCREEN_QUAD = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)


class Renderer:
    """Owns the shaders, the camera background, the effect layer, primitives and text.

    Frame order: ``begin_frame`` → ``draw_background`` → ``begin_effects`` →
    effects draw → ``finish_effects`` → overlay. Renders into ``target`` (the
    window by default, or an offscreen framebuffer for tests and recordings).
    """

    def __init__(
        self,
        ctx: moderngl.Context,
        size: tuple[int, int],
        cfg: Config,
        target: moderngl.Framebuffer | None = None,
    ) -> None:
        self.ctx = ctx
        self.cfg = cfg
        self.target = target if target is not None else ctx.screen
        self.viewport = Viewport(size[0], size[1], image_aspect=size[0] / max(size[1], 1))
        self.shaders = ShaderLibrary(ctx)

        ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self._quad = ctx.buffer(_FULLSCREEN_QUAD.tobytes())
        self._background_prog = self.shaders.program("background", vertex="fullscreen")
        self._background_vao = ctx.vertex_array(self._background_prog, [(self._quad, "2f", "in_pos")])
        self._camera_tex: moderngl.Texture | None = None

        self.postfx = PostFX(ctx, self.shaders, size, cfg.postfx)
        self.primitives = PrimitiveBatch(ctx, self.shaders, cfg.render)
        self.text = TextRenderer(ctx, self.shaders, cfg.debug)

    @property
    def size(self) -> tuple[int, int]:
        return (self.viewport.width, self.viewport.height)

    @property
    def has_camera_image(self) -> bool:
        return self._camera_tex is not None

    def resize(self, size: tuple[int, int]) -> None:
        size = (max(size[0], 1), max(size[1], 1))
        if size != self.size:
            self.viewport.width, self.viewport.height = size
            self.postfx.resize(size)

    def set_camera_image(self, image: np.ndarray, upload: bool = True) -> None:
        """Track the camera aspect ratio and (optionally) upload the RGB frame."""
        height, width = image.shape[:2]
        self.viewport.image_aspect = width / height
        if not upload:
            return
        if self._camera_tex is None or self._camera_tex.size != (width, height):
            if self._camera_tex is not None:
                self._camera_tex.release()
            self._camera_tex = self.ctx.texture((width, height), 3, alignment=1)
            self._camera_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._camera_tex.write(np.ascontiguousarray(image))

    def begin_frame(self) -> None:
        self.target.viewport = (0, 0, *self.size)
        self.target.use()
        self.target.clear(0.0, 0.0, 0.0, 1.0)

    def draw_background(self, visibility: float) -> None:
        """Camera feed (dimmed, tinted) behind everything; ``visibility`` is eased 0..1."""
        if self._camera_tex is None or visibility <= 0.001:
            return
        bg = self.cfg.background
        prog = self._background_prog
        prog["u_uv_rect"].value = self.viewport.uv_rect
        prog["u_visibility"].value = visibility
        prog["u_dim"].value = bg.camera_dim
        prog["u_desaturate"].value = bg.camera_desaturate
        prog["u_tint"].value = bg.camera_tint
        prog["u_vignette"].value = bg.vignette
        self._camera_tex.use(0)
        self.ctx.disable(moderngl.BLEND)
        self._background_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.BLEND)

    def begin_effects(self, dt: float, trail_tau_s: float, diffusion_per_s: float) -> float:
        """Bind the HDR effect layer (last frame faded) with additive blending; returns the frame gain."""
        self.postfx.begin(dt, trail_tau_s, diffusion_per_s)
        return self.postfx.frame_gain

    def finish_effects(self) -> None:
        """Bloom, tone map and add the effect layer onto the target."""
        self.postfx.finish(self.target)

    def use_alpha_blending(self) -> None:
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def release(self) -> None:
        self.text.release()
        self.primitives.release()
        self.postfx.release()
        if self._camera_tex is not None:
            self._camera_tex.release()
        self._background_vao.release()
        self._quad.release()
        self.shaders.release()
