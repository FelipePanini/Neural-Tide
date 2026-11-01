"""HDR effect layer: trails by decay, bloom and tone-mapped compositing.

Per frame::

    begin(dt)   last frame's light × decay → new trail buffer, bound for additive drawing
    (effects draw)
    finish()    bright-pass + dual-filter bloom, then tone map and add onto the target

``ripple()`` starts a liquid shock wave through the layer (used on mode changes).
"""

from __future__ import annotations

import math
import random

import moderngl
import numpy as np

from config import RGB, FocusConfig, PostFXConfig
from render.focus import Spot
from render.shader_library import ShaderLibrary

_FULLSCREEN_QUAD = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
_REFERENCE_DT = 1.0 / 60.0  # effects are tuned at 60 fps; frame_gain keeps that look at any rate


class _Target:
    """A float16 colour texture with its framebuffer."""

    def __init__(self, ctx: moderngl.Context, size: tuple[int, int]) -> None:
        self.texture = ctx.texture(size, 4, dtype="f2")
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = self.texture.repeat_y = False
        self.fbo = ctx.framebuffer(color_attachments=[self.texture])
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)

    @property
    def texel(self) -> tuple[float, float]:
        w, h = self.texture.size
        return (1.0 / w, 1.0 / h)

    def release(self) -> None:
        self.fbo.release()
        self.texture.release()


class PostFX:
    """Owns the trail ping-pong buffers and the bloom mip chain."""

    def __init__(
        self,
        ctx: moderngl.Context,
        shaders: ShaderLibrary,
        size: tuple[int, int],
        cfg: PostFXConfig,
        focus: FocusConfig,
    ) -> None:
        self._ctx = ctx
        self._cfg = cfg
        self._focus_cfg = focus
        self._spots = np.zeros((focus.max_spots, 4), dtype=np.float32)
        self._spot_count = 0
        self._focus_amount = 0.0
        self._quad = ctx.buffer(_FULLSCREEN_QUAD.tobytes())
        self._programs = {
            name: shaders.program(name, vertex="fullscreen")
            for name in ("fade", "bloom_down", "bloom_up", "composite")
        }
        self._vaos = {
            name: ctx.vertex_array(prog, [(self._quad, "2f", "in_pos")]) for name, prog in self._programs.items()
        }
        self._trails: list[_Target] = []
        self._bloom: list[_Target] = []
        self._current = 0
        self._aspect = size[0] / max(size[1], 1)
        self.frame_gain = 1.0
        self._ripple_age = math.inf
        self._ripple_center = (0.5, 0.5)
        self._ripple_color: RGB = (0.0, 0.0, 0.0)
        self._allocate(size)

    def _allocate(self, size: tuple[int, int]) -> None:
        self._trails = [_Target(self._ctx, size), _Target(self._ctx, size)]
        self._bloom = []
        w, h = size
        for _ in range(self._cfg.bloom_levels):
            w, h = max(w // 2, 1), max(h // 2, 1)
            self._bloom.append(_Target(self._ctx, (w, h)))
        self._current = 0

    def resize(self, size: tuple[int, int]) -> None:
        for target in (*self._trails, *self._bloom):
            target.release()
        self._aspect = size[0] / max(size[1], 1)
        self._allocate(size)

    def ripple(self, center_world: np.ndarray, color: RGB) -> None:
        """Start a ripple at a world-space point (e.g. the hand that changed mode)."""
        self._ripple_center = (
            float(center_world[0]) / self._aspect * 0.5 + 0.5,
            float(center_world[1]) * 0.5 + 0.5,
        )
        self._ripple_color = color
        self._ripple_age = 0.0

    def set_focus(self, spots: list[Spot], amount: float) -> None:
        """Spotlights for this frame (world space) and how strongly they mask the layer."""
        self._spots[:] = 0.0
        for row, spot in zip(self._spots, spots, strict=False):
            row[:] = (
                float(spot.center[0]) / self._aspect * 0.5 + 0.5,
                float(spot.center[1]) * 0.5 + 0.5,
                spot.radius * 0.5,  # world units → screen heights
                spot.weight,
            )
        self._spot_count = min(len(spots), len(self._spots))
        self._focus_amount = amount

    def _ripple_uniforms(self) -> tuple[float, float, RGB]:
        """Current radius, displacement and rim colour (all zero when idle)."""
        cfg = self._cfg
        t = self._ripple_age / cfg.ripple_duration_s
        if t >= 1.0:
            return 0.0, 0.0, (0.0, 0.0, 0.0)
        radius = (1.0 - (1.0 - t) ** 3) * cfg.ripple_max_radius  # ease-out cubic
        fade = (1.0 - t) ** 2
        rim = cfg.ripple_rim * fade
        r, g, b = self._ripple_color
        return radius, cfg.ripple_strength * fade, (r * rim, g * rim, b * rim)

    @property
    def scene(self) -> moderngl.Texture:
        """The effect layer of the current frame (HDR)."""
        return self._trails[self._current].texture

    def begin(self, dt: float, trail_tau_s: float, diffusion_per_s: float) -> None:
        """Fade last frame into a fresh buffer and leave it bound with additive blending.

        Light accumulates in the trail up to ``1 / (1 - decay)`` times, which
        depends on the frame rate; ``frame_gain`` (for the effects to multiply
        their brightness by) cancels that so the picture looks the same at 30
        or 144 fps.
        """
        cfg = self._cfg
        self._ripple_age += dt
        decay = math.exp(-dt / trail_tau_s)
        self.frame_gain = (1.0 - decay) / (1.0 - math.exp(-_REFERENCE_DT / trail_tau_s))
        src = self._trails[self._current]
        self._current = 1 - self._current
        dst = self._trails[self._current]

        prog = self._programs["fade"]
        prog["u_texel"].value = src.texel
        prog["u_decay"].value = decay
        prog["u_floor"].value = cfg.trail_floor_per_s * dt
        prog["u_diffusion"].value = min(diffusion_per_s * dt, 1.0)
        dst.fbo.use()
        src.texture.use(0)
        self._ctx.disable(moderngl.BLEND)
        self._vaos["fade"].render(moderngl.TRIANGLE_STRIP)

        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.ONE, moderngl.ONE

    def finish(self, target: moderngl.Framebuffer) -> None:
        """Build the bloom from the current trail buffer and add the tone-mapped result to ``target``."""
        cfg = self._cfg
        ctx = self._ctx
        down, up = self._programs["bloom_down"], self._programs["bloom_up"]

        ctx.disable(moderngl.BLEND)
        source_tex, source_texel = self.scene, self._trails[self._current].texel
        down["u_threshold"].value = cfg.bloom_threshold
        down["u_knee"].value = cfg.bloom_knee
        for level, dst in enumerate(self._bloom):
            down["u_prefilter"].value = 1 if level == 0 else 0
            down["u_texel"].value = source_texel
            dst.fbo.use()
            source_tex.use(0)
            self._vaos["bloom_down"].render(moderngl.TRIANGLE_STRIP)
            source_tex, source_texel = dst.texture, dst.texel

        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        for level in range(len(self._bloom) - 2, -1, -1):
            src, dst = self._bloom[level + 1], self._bloom[level]
            up["u_texel"].value = src.texel
            dst.fbo.use()
            src.texture.use(0)
            self._vaos["bloom_up"].render(moderngl.TRIANGLE_STRIP)

        comp = self._programs["composite"]
        comp["u_scene"].value = 0
        comp["u_bloom"].value = 1
        comp["u_bloom_intensity"].value = cfg.bloom_intensity
        comp["u_exposure"].value = cfg.exposure
        comp["u_whiteout"].value = cfg.whiteout
        comp["u_dither"].value = cfg.dither
        comp["u_seed"].value = random.random() * 1000.0
        radius, strength, rim = self._ripple_uniforms()
        comp["u_aspect"].value = self._aspect
        comp["u_ripple_center"].value = self._ripple_center
        comp["u_ripple_radius"].value = radius
        comp["u_ripple_width"].value = cfg.ripple_width
        comp["u_ripple_strength"].value = strength
        comp["u_ripple_color"].value = rim
        comp["u_spots"].write(self._spots.tobytes())
        comp["u_spot_count"].value = self._spot_count
        comp["u_focus_amount"].value = self._focus_amount
        comp["u_focus_ambient"].value = self._focus_cfg.ambient
        comp["u_focus_falloff"].value = self._focus_cfg.falloff
        target.use()
        self.scene.use(0)
        self._bloom[0].texture.use(1)
        self._vaos["composite"].render(moderngl.TRIANGLE_STRIP)

    def release(self) -> None:
        for target in (*self._trails, *self._bloom):
            target.release()
        for vao in self._vaos.values():
            vao.release()
        self._quad.release()
