"""Mode 2 — Rede neural viva (living neural network).

Nodes float on a slow current and link up when close (a plexus). A pinch
fires a pulse that hops from node to node along the links, losing a bit of
energy on every hop, so it spreads through the net like a wave and fades.
Fingertips plug into nearby nodes, touching a node sparks a small pulse, and
spontaneous pulses keep the network alive when nobody is around.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from config import FlowFieldConfig, NeuralNetConfig
from core.gestures import GestureEvent, GestureKind
from core.hand import FINGERTIPS
from core.hand_state import HandState
from effects.base import Effect, EffectEnv
from render.glow import GlowBatch
from render.primitives import PrimitiveBatch
from sim.flow_field import FlowField
from sim.forces import gaussian

_NEVER = -np.inf


class NeuralSimulation:
    """Node motion, links and pulse propagation (NumPy only)."""

    def __init__(self, cfg: NeuralNetConfig, flow: FlowFieldConfig, half_width: float, half_height: float = 1.0) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        n = cfg.node_count
        self.field = FlowField(replace(flow, speed=cfg.drift_speed, seed=flow.seed + 101, drift_speed=0.0))
        self.field.update(0.0, half_width, half_height)
        inner = (half_width - cfg.inset, half_height - cfg.inset)
        self.pos = self._rng.uniform((-inner[0], -inner[1]), inner, (n, 2)).astype(np.float32)
        self.vel = np.zeros_like(self.pos)
        self.time = 0.0
        self.fire_time = np.full(n, _NEVER)
        self.energy = np.zeros(n)
        self.arrival = np.full(n, np.inf)  # when a travelling pulse reaches each node
        self.arrival_energy = np.zeros(n)
        self.source = np.full(n, -1)  # node the travelling pulse came from
        self.dist = np.zeros((n, n), dtype=np.float32)
        self.adjacent = np.zeros((n, n), dtype=bool)
        self.link_i = np.zeros(0, dtype=np.intp)
        self.link_j = np.zeros(0, dtype=np.intp)
        self.link_distance = cfg.link_distance
        self._update_links(1.0)

    @property
    def count(self) -> int:
        return len(self.pos)

    def activity(self) -> np.ndarray:
        """0..1 glow of each node after it fires."""
        since = self.time - self.fire_time
        return self.energy * np.exp(-np.maximum(since, 0.0) / self._cfg.activity_decay_s)

    def sparks(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions and energies of pulses travelling along links."""
        travelling = np.flatnonzero(np.isfinite(self.arrival) & (self.source >= 0))
        if len(travelling) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros(0)
        src = self.source[travelling]
        start = self.fire_time[src]
        progress = np.clip((self.time - start) / np.maximum(self.arrival[travelling] - start, 1e-6), 0.0, 1.0)
        pos = self.pos[src] + (self.pos[travelling] - self.pos[src]) * progress[:, None]
        return pos, self.arrival_energy[travelling]

    def fire(self, node: int, energy: float) -> bool:
        """Fire ``node`` now unless it is still refractory."""
        if self.time - self.fire_time[node] < self._cfg.refractory_s:
            return False
        self.fire_time[node] = self.time
        self.energy[node] = energy
        self.arrival[node] = np.inf
        self.source[node] = -1
        self._schedule_from(np.array([node]))
        return True

    def fire_near(self, point: np.ndarray, energy: float, radius: float = np.inf) -> bool:
        d = np.linalg.norm(self.pos - point, axis=1)
        node = int(np.argmin(d))
        return d[node] <= radius and self.fire(node, energy)

    def explode(self, center: np.ndarray, speed: float, radius: float) -> None:
        rel = self.pos - center
        r2 = np.einsum("ij,ij->i", rel, rel)
        self.vel += rel * (speed * gaussian(r2, radius) / (np.sqrt(r2) + 1e-6))[:, None]
        for node in np.flatnonzero(r2 < radius * radius):
            self.fire(int(node), self._cfg.pulse_energy)

    def step(self, dt: float, hands: Sequence[HandState], half_width: float, half_height: float = 1.0, intensity: float = 1.0) -> None:
        if dt <= 0.0:  # frozen: keep everything where it is
            return
        cfg = self._cfg
        self.time += dt
        self.field.update(dt, half_width, half_height)
        target = self.field.sample(self.pos)
        target += self._spacing_push()
        target += self._walls(half_width, half_height)
        for hand in hands:
            rel = self.pos - hand.palm
            r2 = np.einsum("ij,ij->i", rel, rel)
            push = cfg.hand_push * hand.presence * gaussian(r2, cfg.hand_push_reach) / (np.sqrt(r2) + 1e-6)
            target += rel * push[:, None]
            target += np.outer(gaussian(r2, cfg.hand_current_reach), hand.palm_velocity * cfg.hand_current_gain * hand.presence)

        self.vel += (target - self.vel) * (1.0 - np.exp(-dt / cfg.follow_tau_s))
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel *= np.minimum(1.0, cfg.max_speed / np.maximum(speed, 1e-9))
        self.pos += self.vel * dt

        self._update_links(intensity)
        self._propagate()
        self._touches(hands)
        if self._rng.random() < cfg.idle_pulse_rate * dt:
            self.fire(int(self._rng.integers(self.count)), cfg.idle_pulse_energy)

    # ------------------------------------------------------------------ internals

    def _spacing_push(self) -> np.ndarray:
        cfg = self._cfg
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)) + np.eye(self.count, dtype=np.float32)
        overlap = np.clip((cfg.spacing - d) / cfg.spacing, 0.0, 1.0)
        np.fill_diagonal(overlap, 0.0)
        return np.einsum("ij,ijk->ik", overlap / d, diff) * cfg.spacing_strength

    def _walls(self, half_width: float, half_height: float) -> np.ndarray:
        cfg = self._cfg
        limit = np.array((half_width - cfg.inset, half_height - cfg.inset), dtype=np.float32)
        excess = np.abs(self.pos) - limit
        return -np.sign(self.pos) * np.maximum(excess, 0.0) * cfg.wall_stiffness

    def _update_links(self, intensity: float) -> None:
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        self.dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        self.link_distance = self._cfg.link_distance * (0.75 + 0.25 * intensity)
        self.adjacent = self.dist < self.link_distance
        np.fill_diagonal(self.adjacent, False)
        self.link_i, self.link_j = np.nonzero(np.triu(self.adjacent, k=1))

    def _propagate(self) -> None:
        due = np.flatnonzero(self.arrival <= self.time)
        if len(due) == 0:
            return
        ready = due[self.time - self.fire_time[due] >= self._cfg.refractory_s]
        self.fire_time[ready] = self.arrival[ready]
        self.energy[ready] = self.arrival_energy[ready]
        self.arrival[due] = np.inf
        self.source[due] = -1
        self._schedule_from(ready)

    def _schedule_from(self, nodes: np.ndarray) -> None:
        """Send pulses from freshly fired ``nodes`` to their linked neighbours."""
        cfg = self._cfg
        if len(nodes) == 0:
            return
        next_energy = self.energy[nodes] * cfg.hop_decay
        rested = self.time - self.fire_time >= cfg.refractory_s
        valid = self.adjacent[nodes] & rested[None, :] & (next_energy[:, None] >= cfg.min_energy)
        arrive = np.where(valid, self.fire_time[nodes, None] + self.dist[nodes] / cfg.pulse_speed, np.inf)
        best = np.argmin(arrive, axis=0)
        cols = np.arange(self.count)
        soonest = arrive[best, cols]
        better = soonest < self.arrival
        self.arrival[better] = soonest[better]
        self.arrival_energy[better] = next_energy[best[better]]
        self.source[better] = nodes[best[better]]

    def _touches(self, hands: Sequence[HandState]) -> None:
        cfg = self._cfg
        for hand in hands:
            if not hand.visible:
                continue
            for tip, raised in zip(hand.tips, hand.fingers, strict=True):
                if raised:
                    self.fire_near(tip, cfg.touch_energy, cfg.touch_radius)


class NeuralNet(Effect):
    """GPU side of mode 2: links as additive lines, nodes and pulses as glows."""

    name = "Rede neural viva"

    def __init__(self, env: EffectEnv) -> None:
        super().__init__(env)
        cfg = self._cfg = env.config.neural
        self.sim = NeuralSimulation(cfg, env.config.flow, env.viewport.aspect)
        render_cfg = replace(env.config.render, max_line_segments=cfg.max_links)
        self._lines = PrimitiveBatch(env.ctx, env.shaders, render_cfg)
        self._glows = GlowBatch(env.ctx, env.shaders, cfg.node_count * 3 + 64)
        self._hands: Sequence[HandState] = ()

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        self._hands = hands
        self.sim.step(dt, hands, self.env.viewport.aspect, intensity=self.intensity)

    def on_gesture(self, event: GestureEvent) -> None:
        cfg = self._cfg
        if event.kind is GestureKind.PINCH_START:
            self.sim.fire_near(event.position, cfg.pulse_energy)
        elif event.kind is GestureKind.EXPLODE:
            self.sim.explode(event.position, cfg.explode_speed, cfg.explode_radius)

    def draw(self) -> None:
        cfg = self._cfg
        sim = self.sim
        palette = self.env.palette
        viewport = self.env.viewport
        fade = self.visibility * (0.7 + 0.3 * self.intensity)
        primary, secondary, highlight = (np.asarray(c) for c in (palette.primary, palette.secondary, palette.highlight))
        activity = sim.activity()

        if len(sim.link_i):
            length = sim.dist[sim.link_i, sim.link_j]
            closeness = np.clip(1.0 - length / sim.link_distance, 0.0, 1.0) ** 1.5
            active = np.clip(np.maximum(activity[sim.link_i], activity[sim.link_j]), 0.0, 1.0)
            colors = np.empty((len(length), 4))
            colors[:, :3] = primary + (highlight - primary) * active[:, None]
            colors[:, 3] = closeness * cfg.line_intensity * (1.0 + cfg.active_line_boost * active) * fade
            self._lines.add_lines(sim.pos[sim.link_i], sim.pos[sim.link_j], cfg.line_px, colors, viewport)

        for hand in self._hands:
            tips = hand.tips[np.array(hand.fingers)]
            for tip in tips:
                d = np.linalg.norm(sim.pos - tip, axis=1)
                near = np.flatnonzero(d < cfg.finger_link_distance)
                if len(near):
                    alpha = (1.0 - d[near] / cfg.finger_link_distance) * cfg.finger_line_intensity * hand.presence * fade
                    colors = np.column_stack((np.tile(secondary, (len(near), 1)), alpha))
                    self._lines.add_lines(np.tile(tip, (len(near), 1)), sim.pos[near], cfg.line_px, colors, viewport)
            if len(tips):
                self._glows.add(tips, cfg.fingertip_radius, secondary * cfg.fingertip_intensity * hand.presence * fade)
        self._lines.flush(viewport, premultiplied=True)

        node_color = secondary + (highlight - secondary) * np.clip(activity, 0.0, 1.0)[:, None]
        glow = (cfg.node_intensity + cfg.active_intensity * activity)[:, None] * node_color * fade
        self._glows.add(sim.pos, cfg.node_radius * (1.0 + activity), glow)
        self._glows.add(sim.pos, cfg.core_radius, node_color * (cfg.core_intensity * fade))
        spark_pos, spark_energy = sim.sparks()
        if len(spark_pos):
            self._glows.add(spark_pos, cfg.spark_radius, highlight * (cfg.spark_intensity * fade) * spark_energy[:, None])
        self._glows.flush(viewport)

    def trail(self) -> tuple[float, float]:
        return self._cfg.trail_tau_s, self._cfg.trail_diffusion_per_s

    def debug_lines(self) -> list[str]:
        sim = self.sim
        pulses = int(np.isfinite(sim.arrival).sum())
        return [f"nós {sim.count}  ligações {len(sim.link_i)}  pulsos {pulses}"]

    def release(self) -> None:
        self._lines.release()
        self._glows.release()
