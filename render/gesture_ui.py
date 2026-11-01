"""On-screen gesture feedback: progress rings at the fingertips, toasts and badges."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import moderngl
import numpy as np

from config import RGBA, RecordingConfig, UIConfig
from core.easing import damp
from core.gestures import GestureFrame, HoldProgress
from core.hand_state import HandState
from render.palette import PaletteState
from render.renderer import Renderer

_FLOATS = 10  # center(2) radius progress marker alpha flash color(3)
_UNIT_QUAD = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)


@dataclass
class _Ring:
    position: np.ndarray
    progress: float = 0.0
    alpha: float = 0.0
    flash: float = 0.0
    labels: tuple[str, ...] = ()
    marker: float | None = None
    blocked: bool = False
    active: bool = True


@dataclass
class _Toast:
    text: str = ""
    age: float = float("inf")
    alpha: float = 0.0


@dataclass
class _Fades:
    frozen: float = 0.0
    link: float = 0.0
    link_ends: tuple[np.ndarray, np.ndarray] = field(default_factory=lambda: (np.zeros(2), np.zeros(2)))


class GestureUI:
    """Draws on top of the composited frame (after the effects, before the debug overlay)."""

    def __init__(self, renderer: Renderer, cfg: UIConfig, palette: PaletteState, capacity: int) -> None:
        self._renderer = renderer
        self._cfg = cfg
        self._palette = palette
        self._capacity = capacity
        ctx = renderer.ctx
        self._prog = renderer.shaders.program("ring")
        self._quad = ctx.buffer(_UNIT_QUAD.tobytes())
        self._instances = ctx.buffer(reserve=capacity * _FLOATS * 4, dynamic=True)
        self._vao = ctx.vertex_array(
            self._prog,
            [
                (self._quad, "2f", "in_corner"),
                (
                    self._instances,
                    "2f 1f 1f 1f 1f 1f 3f/i",
                    "in_center",
                    "in_radius",
                    "in_progress",
                    "in_marker",
                    "in_alpha",
                    "in_flash",
                    "in_color",
                ),
            ],
        )
        self._rings: dict[int, _Ring] = {}
        self._toast = _Toast()
        self._fades = _Fades()
        self._intensity: float | None = None
        self._started = time.perf_counter()

    def notify(self, text: str) -> None:
        """Show a short message at the top of the screen."""
        self._toast = _Toast(text=text, age=0.0, alpha=self._toast.alpha)

    def completed(self, slot: int, position: np.ndarray) -> None:
        """A hold finished: fill the ring and burst it."""
        ring = self._rings.setdefault(slot, _Ring(position=position))
        ring.position, ring.progress, ring.flash, ring.active = position, 1.0, 1.0, False

    def update(self, frame: GestureFrame, hands: Sequence[HandState], frozen: bool, dt: float) -> None:
        cfg = self._cfg
        live: dict[int, HoldProgress] = {hold.slot: hold for hold in frame.holds}
        for slot, hold in live.items():
            ring = self._rings.setdefault(slot, _Ring(position=hold.position))
            ring.position = hold.position
            ring.progress = damp(ring.progress, hold.progress, cfg.ring_progress_tau_s, dt) if ring.active else hold.progress
            ring.labels, ring.marker, ring.blocked, ring.active = hold.labels, hold.marker, hold.blocked, True
        for slot, ring in list(self._rings.items()):
            visible = slot in live
            ring.alpha = damp(ring.alpha, 1.0 if visible or ring.flash > 0.0 else 0.0, cfg.ring_fade_tau_s, dt)
            ring.flash = max(ring.flash - dt / cfg.flash_duration_s, 0.0)
            if not visible and ring.flash <= 0.0 and ring.alpha < 0.01:
                del self._rings[slot]

        self._toast.age += dt
        showing = self._toast.age < cfg.toast_duration_s
        self._toast.alpha = damp(self._toast.alpha, 1.0 if showing else 0.0, cfg.toast_fade_tau_s, dt)
        self._fades.frozen = damp(self._fades.frozen, 1.0 if frozen else 0.0, cfg.toast_fade_tau_s, dt)

        visible_hands = [h for h in hands if h.visible]
        self._intensity = frame.intensity
        if frame.intensity is not None and len(visible_hands) >= 2:
            self._fades.link_ends = (visible_hands[0].palm, visible_hands[1].palm)
            presence = min(visible_hands[0].presence, visible_hands[1].presence)
            self._fades.link = damp(self._fades.link, presence, cfg.link_fade_tau_s, dt)
        else:
            self._fades.link = damp(self._fades.link, 0.0, cfg.link_fade_tau_s, dt)

    def draw(self, intensity: float) -> None:
        self._draw_rings()
        self._renderer.use_alpha_blending()
        self._draw_link(intensity)
        self._draw_texts()

    def draw_recording(self, elapsed: float, cfg: RecordingConfig) -> None:
        """Blinking red dot and timer in the bottom-right corner (not part of the video)."""
        ui = self._cfg
        renderer = self._renderer
        renderer.use_alpha_blending()
        width, height = renderer.size
        label = f"REC {int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
        tex = renderer.text.texture([(label, ui.text_color)], ui.badge_font_size, background=ui.panel_color, padding=ui.padding_px)
        corner = (width - ui.toast_top_px / 2.0, height - ui.toast_top_px / 2.0)
        renderer.text.draw(tex, corner, renderer.size, anchor=(1.0, 1.0))
        dot = renderer.viewport.pixel_to_world_point(corner[0] - tex.size[0] - cfg.indicator_px * 1.5, corner[1] - tex.size[1] / 2.0)
        r, g, b, a = cfg.indicator_color
        blink = 0.55 + 0.45 * np.cos(elapsed * 2.0 * np.pi)
        renderer.primitives.add_points(dot, cfg.indicator_px * 2.0, (r, g, b, a * blink))
        renderer.primitives.flush(renderer.viewport)

    def _draw_rings(self) -> None:
        if not self._rings:
            return
        cfg = self._cfg
        viewport = self._renderer.viewport
        px = viewport.pixels_to_world(1.0)
        color = np.asarray(self._palette.highlight) * cfg.ring_intensity
        rings = list(self._rings.values())[: self._capacity]
        data = np.zeros((len(rings), _FLOATS), dtype=np.float32)
        for row, ring in zip(data, rings, strict=True):
            dim = 0.45 if ring.blocked else 1.0
            marker = -1.0 if ring.marker is None else ring.marker
            row[:] = (*ring.position, cfg.ring_radius_px * px, ring.progress, marker, ring.alpha * dim, ring.flash, *color)
        self._instances.write(data.tobytes())
        prog = self._prog
        prog["u_aspect"].value = viewport.aspect
        prog["u_quad_scale"].value = cfg.ring_quad_scale
        prog["u_half_width"].value = 0.5 * cfg.ring_width_px / cfg.ring_radius_px
        prog["u_time"].value = time.perf_counter() - self._started
        self._renderer.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._vao.render(moderngl.TRIANGLE_STRIP, instances=len(rings))

    def _draw_link(self, intensity: float) -> None:
        if self._fades.link < 0.01:
            return
        cfg = self._cfg
        a, b = self._fades.link_ends
        r, g, bl = self._palette.glow
        color: RGBA = (r, g, bl, cfg.link_alpha * self._fades.link)
        prims = self._renderer.primitives
        prims.add_lines(a, b, cfg.link_px, color, self._renderer.viewport)
        prims.flush(self._renderer.viewport)
        mid = self._renderer.viewport.world_to_pixels((a + b) * 0.5)
        self._label([f"intensidade {intensity:.2f}"], (float(mid[0]), float(mid[1]) - 14.0), (0.5, 1.0), self._fades.link)

    def _draw_texts(self) -> None:
        cfg = self._cfg
        viewport = self._renderer.viewport
        for ring in self._rings.values():
            if ring.labels and ring.alpha > 0.01:
                x, y = viewport.world_to_pixels(ring.position)
                self._label(list(ring.labels), (float(x) + cfg.label_offset_px, float(y)), (0.0, 0.5), ring.alpha)
        width = self._renderer.size[0]
        if self._toast.alpha > 0.01:
            self._label([self._toast.text], (width / 2.0, cfg.toast_top_px), (0.5, 0.0), self._toast.alpha, cfg.toast_font_size)
        if self._fades.frozen > 0.01:
            self._label(["CONGELADO"], (width - cfg.toast_top_px / 2.0, cfg.toast_top_px / 2.0), (1.0, 0.0), self._fades.frozen, cfg.badge_font_size)

    def _label(
        self,
        lines: list[str],
        pos: tuple[float, float],
        anchor: tuple[float, float],
        opacity: float,
        size: int | None = None,
    ) -> None:
        cfg = self._cfg
        text = self._renderer.text
        tex = text.texture(
            [(line, cfg.text_color) for line in lines],
            size or cfg.label_font_size,
            background=cfg.panel_color,
            padding=cfg.padding_px,
        )
        text.draw(tex, pos, self._renderer.size, anchor=anchor, opacity=opacity)

    def release(self) -> None:
        for obj in (self._vao, self._instances, self._quad):
            obj.release()
