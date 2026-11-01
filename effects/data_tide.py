"""Mode 1 — Maré de dados (data tide).

Thousands of particles ride a curl-noise current. Each hand is a smooth
vortex plus a current along its motion, so the particles swirl around it and
get dragged behind it like water behind a paddle; pinching turns the vortex
into a drain. Particles never follow the hand directly: they relax towards
the local flow velocity (a damped spring on velocity), which keeps every
motion soft and wave-like.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import moderngl
import numpy as np

from config import DataTideConfig, FlowFieldConfig
from core.easing import damp
from core.hand import RIGHT
from core.hand_state import HandState
from effects.base import Effect, EffectEnv
from render.glow import GlowBatch
from sim.flow_field import FlowField
from sim.forces import Stirrer, gaussian, suction, vortex

_UNIT_QUAD = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
_INSTANCE_FLOATS = 6  # pos(2) vel(2) alpha(1) colour seed(1)
_TIMING_EMA = 0.1


def stirrers_from_hands(hands: Sequence[HandState], cfg: DataTideConfig) -> list[Stirrer]:
    """Each hand stirs at its palm; right and left hands spin in opposite directions."""
    return [
        Stirrer(
            center=hand.palm,
            velocity=hand.palm_velocity,
            weight=hand.presence,
            spin=cfg.right_hand_spin if hand.handedness == RIGHT else -cfg.right_hand_spin,
            pinch=hand.pinch,
        )
        for hand in hands
    ]


class TideSimulation:
    """The particle system itself: NumPy only, no GPU, so it can be tested directly."""

    def __init__(self, cfg: DataTideConfig, flow: FlowFieldConfig, half_width: float, half_height: float = 1.0) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        n = cfg.particle_count
        self.field = FlowField(flow)
        self.field.update(0.0, half_width, half_height)
        self.pos = self._random_positions(n, half_width, half_height)
        self.vel = self.field.sample(self.pos)
        self.life = self._rng.uniform(cfg.life_min_s, cfg.life_max_s, n).astype(np.float32)
        self.age = (self._rng.random(n) * self.life).astype(np.float32)  # staggered so recycling is spread out
        self.hue = self._rng.random(n).astype(np.float32)
        variation = self._rng.uniform(-cfg.follow_variation, cfg.follow_variation, n)
        self._inv_follow = (1.0 / (cfg.follow_tau_s * (1.0 + variation))).astype(np.float32)
        self._idle_weight = 1.0
        self.step_ms = 0.0

    @property
    def count(self) -> int:
        return len(self.pos)

    @property
    def idle_weight(self) -> float:
        return self._idle_weight

    def _random_positions(self, n: int, half_width: float, half_height: float) -> np.ndarray:
        return self._rng.uniform((-half_width, -half_height), (half_width, half_height), (n, 2)).astype(np.float32)

    def alpha(self) -> np.ndarray:
        """Brightness envelope over each particle's life: fades in after spawning, out before dying."""
        cfg = self._cfg
        a = np.clip(self.age / cfg.fade_in_s, 0.0, 1.0) * np.clip((self.life - self.age) / cfg.fade_out_s, 0.0, 1.0)
        return a * a * (3.0 - 2.0 * a)

    def idle_stirrer(self, half_width: float, half_height: float) -> Stirrer:
        """A faint phantom vortex drifting on a figure-of-eight while no hand is around."""
        cfg = self._cfg
        omega = 2.0 * math.pi / cfg.idle_period_s
        phase = omega * self.field.time
        ax, ay = cfg.idle_path[0] * half_width, cfg.idle_path[1] * half_height
        center = np.array((ax * math.sin(phase), ay * math.sin(2.0 * phase + 0.7)), dtype=np.float32)
        velocity = np.array(
            (ax * omega * math.cos(phase), 2.0 * ay * omega * math.cos(2.0 * phase + 0.7)), dtype=np.float32
        )
        strength = cfg.idle_circulation / cfg.vortex_circulation
        return Stirrer(center=center, velocity=velocity, weight=self._idle_weight * strength, spin=1.0)

    def step(self, dt: float, stirrers: Sequence[Stirrer], half_width: float, half_height: float = 1.0) -> None:
        """Advance every particle by ``dt`` seconds."""
        started = time.perf_counter()
        cfg = self._cfg
        self.field.update(dt, half_width, half_height)
        hand_weight = max((s.weight for s in stirrers), default=0.0)
        self._idle_weight = damp(self._idle_weight, 1.0 - hand_weight, cfg.idle_fade_tau_s, dt)

        target = self.field.sample(self.pos)
        drain = np.zeros(self.count, dtype=np.float32)
        for s in (*stirrers, self.idle_stirrer(half_width, half_height)):
            if s.weight < 1e-3:
                continue
            rel = self.pos - s.center
            r2 = np.einsum("ij,ij->i", rel, rel)
            core = cfg.vortex_core * (1.0 - (1.0 - cfg.pinch_core_scale) * s.pinch)
            circulation = cfg.vortex_circulation * s.spin * s.weight * (1.0 + cfg.pinch_spin_boost * s.pinch)
            target += vortex(rel, r2, circulation, core, cfg.vortex_reach)
            target += np.outer(gaussian(r2, cfg.current_reach), s.velocity * (cfg.current_gain * s.weight))
            if s.pinch > 1e-3:
                pull = cfg.pinch_suction * s.pinch * s.weight
                target += suction(rel, r2, pull, core, cfg.vortex_reach)
                drain += cfg.pinch_drain * s.pinch * s.weight * gaussian(r2, core * 1.5)

        # Damped spring on velocity: relax towards the local flow, never jump.
        blend = 1.0 - np.exp(-dt * self._inv_follow)
        self.vel += (target - self.vel) * blend[:, None]
        speed2 = np.einsum("ij,ij->i", self.vel, self.vel)
        too_fast = speed2 > cfg.max_speed * cfg.max_speed
        if np.any(too_fast):
            self.vel[too_fast] *= (cfg.max_speed / np.sqrt(speed2[too_fast]))[:, None]
        self.pos += self.vel * dt
        self.age += dt * (1.0 + drain)
        self._recycle(half_width, half_height)
        self.step_ms += _TIMING_EMA * ((time.perf_counter() - started) * 1000.0 - self.step_ms)

    def _recycle(self, half_width: float, half_height: float) -> None:
        """Respawn dead or off-screen particles at random spots; they fade in, so nothing pops."""
        cfg = self._cfg
        limit_x, limit_y = half_width + cfg.margin, half_height + cfg.margin
        dead = (self.age >= self.life) | (np.abs(self.pos[:, 0]) > limit_x) | (np.abs(self.pos[:, 1]) > limit_y)
        idx = np.flatnonzero(dead)
        if len(idx) == 0:
            return
        self.pos[idx] = self._random_positions(len(idx), half_width, half_height)
        self.vel[idx] = self.field.sample(self.pos[idx])
        self.age[idx] = 0.0
        self.life[idx] = self._rng.uniform(cfg.life_min_s, cfg.life_max_s, len(idx))


class DataTide(Effect):
    """GPU side of mode 1: particles as velocity-stretched streaks plus a glow at each palm."""

    name = "Maré de dados"

    def __init__(self, env: EffectEnv) -> None:
        super().__init__(env)
        self._cfg = env.config.data_tide
        self.sim = TideSimulation(self._cfg, env.config.flow, env.viewport.aspect)
        ctx = env.ctx
        self._prog = env.shaders.program("tide")
        self._quad = ctx.buffer(_UNIT_QUAD.tobytes())
        self._instances = ctx.buffer(reserve=self.sim.count * _INSTANCE_FLOATS * 4, dynamic=True)
        self._vao = ctx.vertex_array(
            self._prog,
            [
                (self._quad, "2f", "in_corner"),
                (self._instances, "2f 2f 2f/i", "in_pos", "in_vel", "in_info"),
            ],
        )
        self._packed = np.empty((self.sim.count, _INSTANCE_FLOATS), dtype=np.float32)
        self._glows = GlowBatch(ctx, env.shaders, env.config.render.max_glows)
        self._hands: Sequence[HandState] = ()

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        self._hands = hands
        self.sim.step(dt, stirrers_from_hands(hands, self._cfg), self.env.viewport.aspect)

    def draw(self) -> None:
        cfg = self._cfg
        viewport = self.env.viewport
        palette = self.env.palette
        sim = self.sim
        self._packed[:, 0:2] = sim.pos
        self._packed[:, 2:4] = sim.vel
        self._packed[:, 4] = sim.alpha()
        self._packed[:, 5] = sim.hue
        self._instances.write(self._packed.tobytes())

        prog = self._prog
        prog["u_aspect"].value = viewport.aspect
        prog["u_px"].value = viewport.pixels_to_world(1.0)
        prog["u_size_px"].value = cfg.particle_px
        prog["u_streak_s"].value = cfg.streak_s
        prog["u_max_streak_px"].value = cfg.max_streak_px
        prog["u_fast_speed"].value = cfg.fast_speed
        prog["u_intensity"].value = cfg.intensity * self.opacity
        prog["u_primary"].value = palette.primary
        prog["u_secondary"].value = palette.secondary
        prog["u_highlight"].value = palette.highlight
        prog["u_secondary_share"].value = cfg.secondary_share
        self._vao.render(moderngl.TRIANGLE_STRIP, instances=sim.count)

        glow = np.asarray(palette.glow, dtype=np.float32)
        for hand in self._hands:
            energy = cfg.glow_intensity * hand.presence * self.opacity * (1.0 + cfg.glow_pinch_boost * hand.pinch)
            self._glows.add(hand.palm, cfg.glow_radius, glow * energy)
        self._glows.flush(viewport)

    def debug_lines(self) -> list[str]:
        return [f"partículas {self.sim.count}  simulação {self.sim.step_ms:4.1f} ms"]

    def release(self) -> None:
        self._glows.release()
        for obj in (self._vao, self._instances, self._quad):
            obj.release()
