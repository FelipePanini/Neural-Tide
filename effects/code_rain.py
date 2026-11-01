"""Mode 8 — Chuva de código líquida (liquid code rain).

Glyphs pour down from columns like a liquid. Each hand is a cylinder in an
ideal (potential) flow, so the rain parts in front of it and closes again
behind it, the way water flows around a stone; moving the hand also drags
the stream. Glyphs relax towards that flow instead of following it exactly,
which gives them a liquid wobble, and they keep flickering between symbols.
A pinch splashes them, a fist blasts a hole in the rain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from config import CodeRainConfig, FlowFieldConfig
from core.gestures import GestureEvent, GestureKind
from core.hand_state import HandState
from effects.base import Effect, EffectEnv
from render.atlas import GlyphAtlas, SpriteBatch
from sim.flow_field import FlowField
from sim.forces import gaussian


def flow_around_cylinder(rel: np.ndarray, fall_speed: np.ndarray, radius: float) -> np.ndarray:
    """Velocity of a downward stream of speed ``fall_speed`` around a cylinder at the origin.

    Complex potential ``w = U(z·e^{-iθ} + R²e^{iθ}/z)`` with θ = −90°, whose
    derivative gives ``u − iv = iU(1 + R²/z²)``. Inside the cylinder points are
    pushed straight out so nothing gets stuck in the hand.
    """
    z = rel[:, 0] + 1j * rel[:, 1]
    r = np.abs(z)
    safe = np.where(r < 1e-6, 1e-6, z)
    c = 1j * fall_speed * (1.0 + radius * radius / safe**2)
    around = np.stack((c.real, -c.imag), axis=1)
    out = rel / np.maximum(r, 1e-6)[:, None] * (fall_speed * 1.2)[:, None]
    s = np.clip((r - radius * 0.8) / (radius * 0.18), 0.0, 1.0)[:, None]  # blend only inside the hand
    s = s * s * (3.0 - 2.0 * s)
    return s * around + (1.0 - s) * out


class RainSimulation:
    """Glyph particles falling through the hands (NumPy only)."""

    def __init__(self, cfg: CodeRainConfig, flow: FlowFieldConfig, half_width: float, half_height: float = 1.0) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        n = cfg.count
        self.field = FlowField(replace(flow, speed=1.0, seed=flow.seed + 505, drift_speed=0.0))
        self.speed = self._rng.uniform(*cfg.fall_speed, n).astype(np.float32)
        self.pos = np.stack(
            (self._columns(n, half_width), self._rng.uniform(-half_height, half_height + 1.0, n)), axis=1
        ).astype(np.float32)
        self.vel = np.stack((np.zeros(n), -self.speed), axis=1).astype(np.float32)
        self.glyph = self._rng.integers(0, len(cfg.glyphs), n)
        self.time = 0.0

    @property
    def count(self) -> int:
        return len(self.pos)

    def _columns(self, n: int, half_width: float) -> np.ndarray:
        cols = self._rng.integers(0, self._cfg.columns, n)
        return ((cols + 0.5) / self._cfg.columns * 2.0 - 1.0) * half_width

    def step(self, dt: float, hands: Sequence[HandState], half_width: float, half_height: float = 1.0) -> None:
        if dt <= 0.0:  # frozen
            return
        cfg = self._cfg
        self.time += dt
        self.field.update(dt, half_width, half_height)
        free = np.stack((np.zeros(self.count), -self.speed), axis=1)
        target = free + self.field.sample(self.pos) * cfg.wobble
        for hand in hands:
            size = float(np.linalg.norm(hand.points - hand.points.mean(axis=0), axis=1).max())
            radius = max(size * cfg.obstacle_scale, cfg.obstacle_min_radius)
            rel = self.pos - hand.palm
            target += (flow_around_cylinder(rel, self.speed, radius) - free) * hand.presence
            r2 = np.einsum("ij,ij->i", rel, rel)
            target += np.outer(gaussian(r2, radius * 1.5), hand.palm_velocity * cfg.current_gain * hand.presence)
        self.vel += (target - self.vel) * (1.0 - math.exp(-dt / cfg.follow_tau_s))
        self.pos += self.vel * dt

        flicker = self._rng.random(self.count) < cfg.glyph_change_rate * dt
        self.glyph[flicker] = self._rng.integers(0, len(cfg.glyphs), int(flicker.sum()))
        gone = (self.pos[:, 1] < -half_height - cfg.margin) | (np.abs(self.pos[:, 0]) > half_width + cfg.margin)
        idx = np.flatnonzero(gone)
        if len(idx):  # re-enter from above the screen, so nobody sees the jump
            self.pos[idx, 0] = self._columns(len(idx), half_width)
            self.pos[idx, 1] = half_height + cfg.margin + self._rng.uniform(0.0, 0.4, len(idx))
            self.vel[idx] = np.stack((np.zeros(len(idx)), -self.speed[idx]), axis=1)

    def splash(self, center: np.ndarray, speed: float, radius: float) -> None:
        rel = self.pos - center
        r2 = np.einsum("ij,ij->i", rel, rel)
        self.vel += rel * (speed * gaussian(r2, radius) / (np.sqrt(r2) + 1e-6))[:, None]


class CodeRain(Effect):
    name = "Chuva de código"

    def __init__(self, env: EffectEnv) -> None:
        super().__init__(env)
        cfg = self._cfg = env.config.rain
        self.sim = RainSimulation(cfg, env.config.flow, env.viewport.aspect)
        self._atlas = GlyphAtlas(env.ctx, list(cfg.glyphs), env.config.debug.font_names, cfg.font_size)
        self._sprites = SpriteBatch(env.ctx, env.shaders, self._atlas, cfg.count)

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        self.sim.step(dt, hands, self.env.viewport.aspect)

    def on_gesture(self, event: GestureEvent) -> None:
        cfg = self._cfg
        if event.kind is GestureKind.PINCH_START:
            self.sim.splash(event.position, cfg.splash_speed, cfg.splash_radius)
        elif event.kind is GestureKind.EXPLODE:
            self.sim.splash(event.position, cfg.explode_speed, cfg.explode_radius)

    def draw(self) -> None:
        cfg, sim = self._cfg, self.sim
        palette, viewport = self.env.palette, self.env.viewport
        speed = np.linalg.norm(sim.vel, axis=1)
        fast = np.clip((speed - cfg.fall_speed[1]) / cfg.fall_speed[1], 0.0, 1.0)[:, None]
        base = np.asarray(palette.primary) * (0.5 + 0.5 * (sim.speed / cfg.fall_speed[1]))[:, None]
        colors = base + (np.asarray(palette.highlight) - base) * fast
        angles = np.arctan2(sim.vel[:, 0], -sim.vel[:, 1]) * 0.6  # lean into the flow
        brightness = cfg.intensity * self.visibility * (0.7 + 0.3 * self.intensity)
        self._sprites.add(sim.pos, sim.glyph, viewport.pixels_to_world(1.0) * cfg.text_scale, colors * brightness, angles)
        self._sprites.flush(viewport)

    def focus_scale(self) -> float:
        return self._cfg.focus_scale

    def trail(self) -> tuple[float, float]:
        return self._cfg.trail_tau_s, self._cfg.trail_diffusion_per_s

    def debug_lines(self) -> list[str]:
        return [f"glifos {self.sim.count}  colunas {self._cfg.columns}"]

    def release(self) -> None:
        self._sprites.release()
        self._atlas.release()
