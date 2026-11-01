"""Mode 7 — Corpo de partículas (particle body).

MediaPipe's selfie segmentation gives a mask of the person; dust particles
are emitted from it every frame (more along the outline, so the silhouette
reads clearly). Bands of a travelling wave sweep up the body: dust born in a
crest is released and drifts away on a curl-noise current, floating upwards,
while the rest lingers in place and fades. The body keeps dissolving in waves
and re-forming from the live mask. Hands stir the dust; a pinch releases the
whole body at once and a fist blasts it apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from config import FlowFieldConfig, ParticleBodyConfig
from core.gestures import GestureEvent, GestureKind
from core.hand import RIGHT
from core.hand_state import HandState
from core.viewport import Viewport
from effects.base import Effect, EffectEnv
from render.particles import ParticleRenderer
from sim.flow_field import FlowField
from sim.forces import gaussian, vortex


class BodySimulation:
    """Ring buffer of dust particles emitted from a person mask (NumPy only)."""

    def __init__(self, cfg: ParticleBodyConfig, flow: FlowFieldConfig) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        n = cfg.capacity
        self.field = FlowField(replace(flow, speed=1.0, seed=flow.seed + 404, drift_speed=0.0))
        self.pos = np.zeros((n, 2), dtype=np.float32)
        self.vel = np.zeros_like(self.pos)
        self.age = np.full(n, np.inf, dtype=np.float32)  # inf = free slot
        self.life = np.ones(n, dtype=np.float32)
        self.release = np.zeros(n, dtype=np.float32)
        self.hue = self._rng.random(n).astype(np.float32)
        self.time = 0.0
        self._next = 0
        self._carry = 0.0
        self._burst_until = -1.0
        self.coverage = 0.0  # share of the frame covered by the person

    @property
    def count(self) -> int:
        return len(self.pos)

    def alive(self) -> np.ndarray:
        return self.age < self.life

    def alpha(self) -> np.ndarray:
        cfg = self._cfg
        a = np.clip(self.age / cfg.fade_in_s, 0.0, 1.0) * np.clip(1.0 - self.age / self.life, 0.0, 1.0) ** 1.5
        return np.where(self.alive(), a, 0.0)

    def wave(self, y: np.ndarray) -> np.ndarray:
        """0..1 release factor of the dissolve bands travelling up the body."""
        cfg = self._cfg
        if self.time < self._burst_until:
            return np.ones_like(y)
        phase = 2.0 * math.pi * (y / cfg.wave_length - self.time * cfg.wave_speed / cfg.wave_length)
        return (0.5 - 0.5 * np.sin(phase)) ** cfg.wave_sharpness

    def emit(self, mask: np.ndarray, viewport: Viewport, dt: float) -> int:
        """Spawn dust from ``mask`` (H, W, 0..1, image space); returns how many."""
        cfg = self._cfg
        h, w = mask.shape
        body = mask > cfg.threshold
        self.coverage = float(body.mean())
        if not body.any():
            return 0
        gy, gx = np.gradient(mask)
        weight = (body + np.hypot(gx, gy) * cfg.edge_weight).ravel()
        self._carry += cfg.spawn_per_s * dt
        k = min(int(self._carry), self.count)
        self._carry -= k
        if k == 0:
            return 0
        idx = self._rng.choice(weight.size, size=k, p=weight / weight.sum())
        ys, xs = np.divmod(idx, w)
        uv = np.stack(((xs + self._rng.random(k)) / w, (ys + self._rng.random(k)) / h), axis=1)
        world = viewport.image_to_world(uv).astype(np.float32)
        slots = (self._next + np.arange(k)) % self.count
        self._next = int((self._next + k) % self.count)
        self.pos[slots] = world
        self.vel[slots] = 0.0
        self.age[slots] = 0.0
        self.life[slots] = self._rng.uniform(cfg.life_min_s, cfg.life_max_s, k)
        self.release[slots] = self.wave(world[:, 1])
        return k

    def step(self, dt: float, hands: Sequence[HandState], half_width: float) -> None:
        if dt <= 0.0:  # frozen
            return
        cfg = self._cfg
        self.time += dt
        self.field.update(dt, half_width)
        live = np.flatnonzero(self.alive())
        if len(live) == 0:
            return
        self.age[live] += dt
        pos, release = self.pos[live], self.release[live][:, None]
        target = self.field.sample(pos) * cfg.flow_drift * release
        target[:, 1] += cfg.rise * release[:, 0]
        target *= cfg.release_speed / max(cfg.flow_drift, 1e-6)
        for hand in hands:
            rel = pos - hand.palm
            r2 = np.einsum("ij,ij->i", rel, rel)
            spin = 1.0 if hand.handedness == RIGHT else -1.0
            target += vortex(rel, r2, cfg.hand_circulation * spin * hand.presence, cfg.hand_core, cfg.hand_reach)
            target += np.outer(gaussian(r2, cfg.hand_reach * 0.6), hand.palm_velocity * 0.6 * hand.presence)
        vel = self.vel[live]
        vel += (target - vel) * (1.0 - math.exp(-dt / cfg.follow_tau_s))
        self.vel[live] = vel
        self.pos[live] = pos + vel * dt

    def burst(self) -> None:
        """Release everything for a moment (pinch)."""
        self._burst_until = self.time + self._cfg.burst_duration_s
        self.release[self.alive()] = 1.0

    def explode(self, center: np.ndarray, speed: float, radius: float) -> None:
        rel = self.pos - center
        r2 = np.einsum("ij,ij->i", rel, rel)
        self.vel += rel * (speed * gaussian(r2, radius) / (np.sqrt(r2) + 1e-6))[:, None]
        self.release[self.alive()] = 1.0


class ParticleBody(Effect):
    name = "Corpo de partículas"
    needs_segmentation = True

    def __init__(self, env: EffectEnv) -> None:
        super().__init__(env)
        cfg = self._cfg = env.config.body
        self.sim = BodySimulation(cfg, env.config.flow)
        self._particles = ParticleRenderer(env.ctx, env.shaders, cfg.capacity)
        self._has_mask = False

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        if dt <= 0.0:
            return
        mask = self.env.segmentation()
        self._has_mask = mask is not None
        if mask is not None:
            self.sim.emit(mask, self.env.viewport, dt)
        self.sim.step(dt, hands, self.env.viewport.aspect)

    def on_gesture(self, event: GestureEvent) -> None:
        if event.kind is GestureKind.PINCH_START:
            self.sim.burst()
        elif event.kind is GestureKind.EXPLODE:
            self.sim.explode(event.position, self._cfg.explode_speed, self._cfg.explode_radius)

    def draw(self) -> None:
        sim = self.sim
        brightness = self.visibility * (0.7 + 0.3 * self.intensity)
        self._particles.draw(sim.pos, sim.vel, sim.alpha(), sim.hue, self._cfg.look, self.env.palette, self.env.viewport, brightness)

    def focus_strength(self) -> float:
        return 0.0  # the silhouette already is the shape: no hand spotlight

    def trail(self) -> tuple[float, float]:
        return self._cfg.trail_tau_s, self._cfg.trail_diffusion_per_s

    def debug_lines(self) -> list[str]:
        if not self._has_mask:
            return ["silhueta: aguardando a segmentação"]
        return [f"poeira {int(self.sim.alive().sum())}  silhueta {self.sim.coverage * 100:4.1f}% da imagem"]

    def release(self) -> None:
        self._particles.release()
