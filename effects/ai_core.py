"""Mode 4 — Núcleo de IA (AI core).

A sphere of particles, wrapped in orbit rings, that breathes. Held between
two hands, the distance between the palms sets its size, the wrist roll spins
it and tilting the hands like a steering wheel banks it. With one hand it
floats above the palm; with none it drifts at the centre. Every particle
chases its place on the sphere with a critically damped spring, so the core
lags and wobbles like something alive instead of moving rigidly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from config import AICoreConfig
from core.easing import damp
from core.gestures import GestureEvent, GestureKind
from core.hand_state import HandState
from effects.base import Effect, EffectEnv
from render.focus import Spot
from render.glow import GlowBatch
from render.particles import ParticleRenderer
from sim.noise import SimplexNoise

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_RING_RADIUS = 1.3  # rings orbit outside the shell (in sphere radii)


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` evenly spread unit vectors (x right, y up, z towards the viewer)."""
    i = np.arange(n) + 0.5
    y = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - y * y)
    theta = _GOLDEN_ANGLE * i
    return np.stack((np.cos(theta) * r, y, np.sin(theta) * r), axis=1)


def rotation(yaw: float, pitch: float, bank: float) -> np.ndarray:
    """Rotation matrix: yaw around y, then pitch around x, then bank around the view axis."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cb, sb = math.cos(bank), math.sin(bank)
    ry = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp)))
    rz = np.array(((cb, -sb, 0.0), (sb, cb, 0.0), (0.0, 0.0, 1.0)))
    return rz @ rx @ ry


class AICoreSimulation:
    """Control (centre, size, rotation) from the hands plus spring-driven particles (NumPy only)."""

    def __init__(self, cfg: AICoreConfig) -> None:
        self._cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self._noise = SimplexNoise(cfg.seed)
        self._shell = fibonacci_sphere(cfg.shell_particles)
        per_ring = cfg.ring_particles // max(cfg.rings, 1)
        self._ring_index = np.repeat(np.arange(cfg.rings), per_ring)
        self._ring_angle = np.tile(np.linspace(0.0, 2.0 * math.pi, per_ring, endpoint=False), cfg.rings)
        self._ring_angle += rng.uniform(0.0, 0.05, len(self._ring_angle))
        tilts = np.linspace(-0.6, 0.6, cfg.rings) + 0.3
        self._ring_tilt = tilts[self._ring_index]
        self.hue = np.concatenate((rng.uniform(0.0, 0.7, len(self._shell)), rng.uniform(0.85, 1.0, len(self._ring_angle))))
        self.time = 0.0
        self.center = np.zeros(2)
        self.radius = cfg.base_radius
        self.yaw = self.bank = 0.0
        self._spin = 0.0
        self._stiffness = 1.0  # drops after an explosion so the core really comes apart
        self._pulse_age = math.inf
        self.pos, self.depth = self.targets()
        self.vel = np.zeros_like(self.pos)

    @property
    def count(self) -> int:
        return len(self.pos)

    def _breath(self) -> float:
        cfg = self._cfg
        beat = 0.0
        if math.isfinite(self._pulse_age):  # damped heartbeat after a pinch
            beat = cfg.pulse_amount * self.pulse_energy() * math.sin(cfg.pulse_frequency * self._pulse_age)
        return 1.0 + cfg.breath_amount * math.sin(2.0 * math.pi * self.time / cfg.breath_period_s) + beat

    def targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Where each particle wants to be (world space) and its depth (−1 back … 1 front)."""
        cfg = self._cfg
        shell = self._shell
        t = self.time * cfg.wobble_speed
        wobble = self._noise.noise3(shell[:, 0] * cfg.wobble_scale, shell[:, 1] * cfg.wobble_scale, shell[:, 2] * cfg.wobble_scale + t)
        shell_pts = shell * (1.0 + cfg.wobble * wobble)[:, None]
        a, tilt = self._ring_angle, self._ring_tilt
        ring_pts = np.stack((np.cos(a), np.sin(a) * np.sin(tilt), np.sin(a) * np.cos(tilt)), axis=1) * _RING_RADIUS
        local = np.concatenate((shell_pts, ring_pts)) * (self.radius * self._breath())
        # einsum instead of @: macOS Accelerate can raise spurious FP warnings in small matmuls.
        world = np.einsum("ij,kj->ik", local, rotation(self.yaw, 0.35, self.bank))
        distance = cfg.camera_distance * self.radius
        scale = distance / np.maximum(distance - world[:, 2], 1e-3)
        pos = self.center + world[:, :2] * scale[:, None]
        return pos.astype(np.float32), np.clip(world[:, 2] / (self.radius * _RING_RADIUS), -1.0, 1.0)

    def _control(self, hands: Sequence[HandState], half_width: float) -> tuple[np.ndarray, float, float, float]:
        cfg = self._cfg
        held = sorted((h for h in hands if h.visible and h.presence > 0.3), key=lambda h: -h.presence)[:2]
        if len(held) == 2:
            left, right = sorted(held, key=lambda h: h.palm[0])
            span = right.palm - left.palm
            radius = float(np.clip(np.linalg.norm(span) * cfg.hand_distance_scale, cfg.min_radius, cfg.max_radius))
            bank = float(np.clip(math.atan2(span[1], span[0]) * cfg.tilt_gain, -1.2, 1.2))
            return (left.palm + right.palm) * 0.5, radius, 0.5 * (left.roll + right.roll), bank
        if len(held) == 1:
            hand = held[0]
            return hand.palm + np.array(cfg.one_hand_offset) * cfg.base_radius / 0.32, cfg.base_radius, hand.roll, 0.0
        drift = np.array((0.25 * half_width * math.sin(self.time * 0.13), 0.15 * math.sin(self.time * 0.21)))
        return drift, cfg.base_radius * 0.9, 0.0, 0.0

    def step(self, dt: float, hands: Sequence[HandState], half_width: float) -> None:
        if dt <= 0.0:  # frozen
            return
        cfg = self._cfg
        self.time += dt
        self._pulse_age += dt
        self._spin += cfg.auto_spin * dt
        center, radius, roll, bank = self._control(hands, half_width)
        self.center = damp(self.center, center, cfg.center_tau_s, dt)
        self.radius = damp(self.radius, radius, cfg.radius_tau_s, dt)
        self.yaw = damp(self.yaw, self._spin + roll * cfg.roll_gain, cfg.rotation_tau_s, dt)
        self.bank = damp(self.bank, bank, cfg.rotation_tau_s, dt)
        direction = np.where(self._ring_index % 2 == 0, 1.0, -1.0)
        self._ring_angle += cfg.ring_speed * dt * direction * (1.0 + 0.3 * self._ring_index)

        target, self.depth = self.targets()
        self._stiffness = damp(self._stiffness, 1.0, cfg.reform_tau_s, dt)
        omega = 2.0 * math.pi / cfg.spring_period_s * math.sqrt(self._stiffness)
        self.vel += (omega * omega * (target - self.pos) - 2.0 * omega * self.vel) * dt
        self.pos += self.vel * dt

    def pulse(self) -> None:
        self._pulse_age = 0.0

    def pulse_energy(self) -> float:
        return math.exp(-self._pulse_age / self._cfg.pulse_decay_s)

    def explode(self, speed: float) -> None:
        rel = self.pos - self.center
        dist = np.linalg.norm(rel, axis=1, keepdims=True) + 1e-6
        jitter = np.random.default_rng().uniform(0.5, 1.5, (self.count, 1))
        self.vel += rel / dist * speed * jitter
        self._stiffness = self._cfg.explode_looseness


class AICore(Effect):
    name = "Núcleo de IA"

    def __init__(self, env: EffectEnv) -> None:
        super().__init__(env)
        cfg = self._cfg = env.config.core
        self.sim = AICoreSimulation(cfg)
        self._particles = ParticleRenderer(env.ctx, env.shaders, self.sim.count)
        self._glows = GlowBatch(env.ctx, env.shaders, 4)

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        self.sim.step(dt, hands, self.env.viewport.aspect)

    def on_gesture(self, event: GestureEvent) -> None:
        if event.kind is GestureKind.PINCH_START:
            self.sim.pulse()
        elif event.kind is GestureKind.EXPLODE:
            self.sim.explode(self._cfg.explode_speed)

    def draw(self) -> None:
        cfg, sim = self._cfg, self.sim
        palette = self.env.palette
        alpha = 0.25 + 0.75 * (sim.depth * 0.5 + 0.5) ** 1.5  # the far side of the sphere is dimmer
        brightness = self.visibility * (0.7 + 0.3 * self.intensity)
        self._particles.draw(sim.pos, sim.vel, alpha, sim.hue, cfg.look, palette, self.env.viewport, brightness)
        energy = cfg.glow_intensity * self.visibility * (1.0 + 2.0 * sim.pulse_energy())
        self._glows.add(sim.center, sim.radius * cfg.glow_radius, np.asarray(palette.glow) * energy)
        self._glows.flush(self.env.viewport)

    def focus_spots(self) -> list[Spot]:
        return [Spot(self.sim.center.copy(), self.sim.radius * self._cfg.focus_margin, 1.0)]

    def trail(self) -> tuple[float, float]:
        return self._cfg.trail_tau_s, self._cfg.trail_diffusion_per_s

    def debug_lines(self) -> list[str]:
        sim = self.sim
        return [f"partículas {sim.count}  raio {sim.radius:.2f}  giro {math.degrees(sim.yaw) % 360:3.0f}°"]

    def release(self) -> None:
        self._particles.release()
        self._glows.release()
