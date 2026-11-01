"""Mode 5 — Fios de luz (light ribbons).

Each raised fingertip pulls a ribbon of light. A ribbon is a chain of points
where every point chases the one ahead of it, so the ribbon trails behind the
finger in smooth curves; the tail also sways in a curl-noise current, like
silk in water. Curling a finger fades its ribbon, pinching makes the thumb
and index ribbons flare, and a fist flings the ribbons outwards. With no
hands, a few phantom ribbons keep looping slowly.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import replace

import moderngl
import numpy as np

from config import FlowFieldConfig, RibbonsConfig
from core.easing import damp, damp_factor
from core.gestures import GestureEvent, GestureKind
from core.hand import FINGERTIPS
from core.hand_state import HandState
from effects.base import Effect, EffectEnv
from sim.flow_field import FlowField
from sim.forces import gaussian

_VERTEX_FLOATS = 7  # pos(2) colour(3) alpha(1) edge(1)
_IMPULSE_DECAY = 3.0  # 1/s: how fast a fist's fling slows down


class RibbonSimulation:
    """All ribbons as one ``(ribbons, segments, 2)`` array (NumPy only)."""

    def __init__(self, cfg: RibbonsConfig, flow: FlowFieldConfig) -> None:
        self._cfg = cfg
        self.field = FlowField(replace(flow, speed=1.0, seed=flow.seed + 202, drift_speed=0.0))
        self.keys: list[Hashable] = []
        self.points = np.zeros((0, cfg.segments, 2), dtype=np.float32)
        self.vel = np.zeros_like(self.points)
        self.alpha = np.zeros(0)
        self.flare = np.zeros(0)
        self.shade = np.zeros(0)  # 0..1 position in the palette gradient
        self.time = 0.0
        self._idle_weight = 1.0
        self._along = np.linspace(0.0, 1.0, cfg.segments, dtype=np.float32)

    @property
    def count(self) -> int:
        return len(self.keys)

    def _row(self, key: Hashable, head: np.ndarray, shade: float) -> int:
        if key in self.keys:
            return self.keys.index(key)
        if len(self.keys) >= self._cfg.max_ribbons:
            return -1
        self.keys.append(key)
        fresh = np.repeat(head[None, None, :].astype(np.float32), self._cfg.segments, axis=1)  # collapsed, grows out
        self.points = np.concatenate((self.points, fresh))
        self.vel = np.concatenate((self.vel, np.zeros_like(fresh)))
        self.alpha = np.append(self.alpha, 0.0)
        self.flare = np.append(self.flare, 0.0)
        self.shade = np.append(self.shade, shade)
        return len(self.keys) - 1

    def _heads(
        self, hands: Sequence[HandState], half_width: float, dt: float
    ) -> dict[Hashable, tuple[np.ndarray, float, float, float]]:
        """key → (head position, target alpha, target flare, colour shade)."""
        cfg = self._cfg
        heads: dict[Hashable, tuple[np.ndarray, float, float, float]] = {}
        for hand in hands:
            for finger, (tip, raised) in enumerate(zip(FINGERTIPS, hand.fingers, strict=True)):
                flare = hand.pinch if finger < 2 else 0.0
                heads[(hand.slot, finger)] = (hand.points[tip], hand.presence * float(raised), flare, finger / 4.0)
        presence = max((h.presence for h in hands), default=0.0)
        self._idle_weight = damp(self._idle_weight, 1.0 - presence, cfg.fade_tau_s * 4.0, dt)
        ax, ay = cfg.idle_size
        for i in range(cfg.idle_ribbons):
            phase = 2.0 * math.pi * (self.time / cfg.idle_period_s + i / cfg.idle_ribbons)
            head = np.array((ax * half_width * math.sin(phase * (1 + i % 2)), ay * math.sin(phase * 1.5 + i)))
            heads[("idle", i)] = (head, self._idle_weight, 0.0, i / max(cfg.idle_ribbons - 1, 1))
        return heads

    def step(self, dt: float, hands: Sequence[HandState], half_width: float) -> None:
        if dt <= 0.0:  # frozen
            return
        cfg = self._cfg
        self.time += dt
        self.field.update(dt, half_width)
        heads = self._heads(hands, half_width, dt)
        targets = np.zeros(len(self.keys))
        flares = np.zeros(len(self.keys))
        for key, (head, alpha, flare, shade) in heads.items():
            row = self._row(key, head, shade)
            if row < 0:
                continue
            if row >= len(targets):
                targets, flares = np.append(targets, 0.0), np.append(flares, 0.0)
            self.points[row, 0] = head
            targets[row], flares[row] = alpha, flare
        self.alpha = damp(self.alpha, targets, cfg.fade_tau_s, dt)
        self.flare = damp(self.flare, flares, cfg.fade_tau_s, dt)

        follow = damp_factor(dt, 1.0 / cfg.follow_rate)
        pts = self.points
        for i in range(1, cfg.segments):
            pts[:, i] += (pts[:, i - 1] - pts[:, i]) * follow
        sway = self.field.sample(pts.reshape(-1, 2)).reshape(pts.shape)
        pts[:, 1:] += sway[:, 1:] * (cfg.drift * dt) * self._along[None, 1:, None]
        pts[:, 1:] += self.vel[:, 1:] * dt
        self.vel *= math.exp(-_IMPULSE_DECAY * dt)

        keep = (self.alpha > 0.004) | np.array([key in heads for key in self.keys], dtype=bool)
        if not keep.all():
            self.keys = [k for k, kept in zip(self.keys, keep, strict=True) if kept]
            self.points, self.vel = self.points[keep], self.vel[keep]
            self.alpha, self.flare, self.shade = self.alpha[keep], self.flare[keep], self.shade[keep]

    def explode(self, center: np.ndarray, speed: float, radius: float = 0.8) -> None:
        rel = self.points - center
        r2 = np.einsum("rsk,rsk->rs", rel, rel)
        self.vel += rel * (speed * gaussian(r2, radius) / (np.sqrt(r2) + 1e-6))[..., None]

    def mesh(self, palette_colors: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Triangle vertices ``(n, 7)`` for every visible ribbon, tapering towards the tail."""
        cfg = self._cfg
        visible = self.alpha > 0.004
        if not visible.any():
            return np.zeros((0, _VERTEX_FLOATS), dtype=np.float32)
        pts = self.points[visible]
        tangent = np.gradient(pts, axis=1)
        normal = np.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
        width = cfg.head_width + (cfg.tail_width - cfg.head_width) * self._along**0.6
        left = pts + normal * width[None, :, None]
        right = pts - normal * width[None, :, None]
        fade = (1.0 - self._along) ** 1.3
        alpha = (self.alpha[visible] * (1.0 + cfg.pinch_flare * self.flare[visible]))[:, None] * fade[None, :]
        start, end = palette_colors
        color = start + (end - start) * self.shade[visible][:, None]  # (r, 3)

        def vertex(p: np.ndarray, a: np.ndarray, edge: float) -> np.ndarray:
            v = np.empty(p.shape[:2] + (_VERTEX_FLOATS,), dtype=np.float32)
            v[..., 0:2] = p
            v[..., 2:5] = color[:, None, :]
            v[..., 5] = a
            v[..., 6] = edge
            return v

        l0, r0 = vertex(left[:, :-1], alpha[:, :-1], 1.0), vertex(right[:, :-1], alpha[:, :-1], -1.0)
        l1, r1 = vertex(left[:, 1:], alpha[:, 1:], 1.0), vertex(right[:, 1:], alpha[:, 1:], -1.0)
        return np.stack((l0, r0, l1, l1, r0, r1), axis=2).reshape(-1, _VERTEX_FLOATS)


class LightRibbons(Effect):
    name = "Fios de luz"

    def __init__(self, env: EffectEnv) -> None:
        super().__init__(env)
        cfg = self._cfg = env.config.ribbons
        self.sim = RibbonSimulation(cfg, env.config.flow)
        capacity = cfg.max_ribbons * (cfg.segments - 1) * 6
        self._prog = env.shaders.program("ribbon")
        self._buf = env.ctx.buffer(reserve=capacity * _VERTEX_FLOATS * 4, dynamic=True)
        self._vao = env.ctx.vertex_array(
            self._prog, [(self._buf, "2f 3f 1f 1f", "in_pos", "in_color", "in_alpha", "in_edge")]
        )

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        self.sim.step(dt, hands, self.env.viewport.aspect)

    def on_gesture(self, event: GestureEvent) -> None:
        if event.kind is GestureKind.EXPLODE:
            self.sim.explode(event.position, self._cfg.explode_speed)

    def draw(self) -> None:
        cfg = self._cfg
        palette = self.env.palette
        verts = self.sim.mesh((np.asarray(palette.primary), np.asarray(palette.secondary)))
        if len(verts) == 0:
            return
        self._buf.write(verts.tobytes())
        brightness = self.visibility * (0.7 + 0.3 * self.intensity)
        self._prog["u_aspect"].value = self.env.viewport.aspect
        self._prog["u_highlight"].value = palette.highlight
        self._prog["u_intensity"].value = cfg.intensity * brightness
        self._prog["u_core"].value = cfg.core_intensity * brightness
        self._vao.render(moderngl.TRIANGLES, vertices=len(verts))

    def trail(self) -> tuple[float, float]:
        return self._cfg.trail_tau_s, self._cfg.trail_diffusion_per_s

    def debug_lines(self) -> list[str]:
        return [f"fios {int((self.sim.alpha > 0.01).sum())}  segmentos {self._cfg.segments}"]

    def release(self) -> None:
        self._vao.release()
        self._buf.release()
