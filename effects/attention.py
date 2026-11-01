"""Mode 6 — Atenção (LLM attention).

Words from the world of language models float around. Each hand is a query:
the closest tokens receive attention weights from a softmax over their
distance, drawn as glowing arcs whose thickness follows the weight, plus
fainter self-attention arcs between the attended tokens. Pinching lowers the
softmax temperature (sharper focus) and sends sparks along the arcs into the
hand; a fist scatters the tokens.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from config import AttentionConfig, FlowFieldConfig
from core.easing import damp
from core.gestures import GestureEvent, GestureKind
from core.hand import INDEX_TIP
from core.hand_state import HandState
from effects.base import Effect, EffectEnv
from render.atlas import GlyphAtlas, SpriteBatch
from render.glow import GlowBatch
from render.primitives import PrimitiveBatch
from sim.flow_field import FlowField
from sim.forces import gaussian

_MIN_WEIGHT = 0.02
_MAX_PAIRS = 6


@dataclass(frozen=True)
class Query:
    key: Hashable
    point: np.ndarray  # world space
    tokens: np.ndarray  # indices of the attended tokens
    weights: np.ndarray  # softmax weights (sum to 1)
    strength: float  # hand presence, or the idle phantom's weight


def softmax_attention(pos: np.ndarray, query: np.ndarray, reach: float, top_k: int, temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices and softmax weights of the ``top_k`` tokens closest to ``query`` within ``reach``."""
    d2 = np.einsum("ij,ij->i", pos - query, pos - query)
    near = np.flatnonzero(d2 < reach * reach)
    if len(near) == 0:
        return near, np.zeros(0)
    near = near[np.argsort(d2[near])[:top_k]]
    logits = -d2[near] / temperature
    w = np.exp(logits - logits.max())
    return near, w / w.sum()


def arc(a: np.ndarray, b: np.ndarray, bend: float, segments: int) -> np.ndarray:
    """Points of a quadratic Bézier from ``a`` to ``b`` bowed sideways by ``bend`` × length."""
    mid = 0.5 * (a + b)
    d = b - a
    control = mid + np.array((-d[1], d[0])) * bend
    t = np.linspace(0.0, 1.0, segments + 1)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * control + t**2 * b


class AttentionSimulation:
    """Token motion and attention weights (NumPy only)."""

    def __init__(self, cfg: AttentionConfig, flow: FlowFieldConfig, half_width: float, half_height: float = 1.0) -> None:
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)
        n = len(cfg.tokens)
        self.field = FlowField(replace(flow, speed=cfg.drift_speed, seed=flow.seed + 303, drift_speed=0.0))
        self.field.update(0.0, half_width, half_height)
        inner = (half_width - cfg.inset, half_height - cfg.inset)
        self.pos = self._rng.uniform((-inner[0], -inner[1]), inner, (n, 2)).astype(np.float32)
        self.vel = np.zeros_like(self.pos)
        self.time = 0.0
        self.queries: list[Query] = []
        self.sparks: list[tuple[Hashable, int, float]] = []  # (query key, token, start time)
        self._temperature: dict[Hashable, float] = {}
        self._idle_weight = 1.0

    @property
    def count(self) -> int:
        return len(self.pos)

    def attention(self) -> np.ndarray:
        """Strongest attention each token receives (0..1)."""
        total = np.zeros(self.count)
        for q in self.queries:
            np.maximum.at(total, q.tokens, q.weights * q.strength)
        return total

    def _query_points(self, hands: Sequence[HandState], half_width: float, dt: float) -> list[tuple[Hashable, np.ndarray, float, float]]:
        cfg = self._cfg
        out: list[tuple[Hashable, np.ndarray, float, float]] = []
        for hand in hands:
            point = hand.points[INDEX_TIP] if hand.fingers[1] else hand.palm
            out.append((hand.slot, point, hand.presence, hand.pinch))
        presence = max((h.presence for h in hands), default=0.0)
        self._idle_weight = damp(self._idle_weight, 1.0 - presence, 0.8, dt)
        phase = 2.0 * math.pi * self.time / cfg.idle_period_s
        idle = np.array((0.6 * half_width * math.sin(phase), 0.45 * math.sin(2.0 * phase + 0.4)))
        out.append(("idle", idle, self._idle_weight * cfg.idle_strength, 0.0))
        return out

    def step(self, dt: float, hands: Sequence[HandState], half_width: float, half_height: float = 1.0) -> None:
        if dt <= 0.0:  # frozen
            return
        cfg = self._cfg
        self.time += dt
        self.field.update(dt, half_width, half_height)
        target = self.field.sample(self.pos) + self._spacing() + self._walls(half_width, half_height)

        queries = []
        for key, point, strength, pinch in self._query_points(hands, half_width, dt):
            wanted = cfg.temperature + (cfg.pinch_temperature - cfg.temperature) * pinch
            temperature = damp(self._temperature.get(key, cfg.temperature), wanted, 0.15, dt)
            self._temperature[key] = temperature
            tokens, weights = softmax_attention(self.pos, point, cfg.reach, cfg.top_k, temperature)
            queries.append(Query(key, point, tokens, weights, strength))
            if len(tokens):
                pull = (point - self.pos[tokens]) * (cfg.hand_pull * strength * weights)[:, None]
                target[tokens] += pull
        self.queries = queries

        self.vel += (target - self.vel) * (1.0 - math.exp(-dt / cfg.follow_tau_s))
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel *= np.minimum(1.0, cfg.max_speed / np.maximum(speed, 1e-9))
        self.pos += self.vel * dt
        self.sparks = [s for s in self.sparks if (self.time - s[2]) * cfg.spark_speed < 1.0]

    def _spacing(self) -> np.ndarray:
        cfg = self._cfg
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)) + np.eye(self.count, dtype=np.float32)
        overlap = np.clip((cfg.spacing - d) / cfg.spacing, 0.0, 1.0)
        np.fill_diagonal(overlap, 0.0)
        return np.einsum("ij,ijk->ik", overlap / d, diff) * cfg.spacing_strength

    def _walls(self, half_width: float, half_height: float) -> np.ndarray:
        cfg = self._cfg
        limit = np.array((half_width - cfg.inset, half_height - cfg.inset), dtype=np.float32)
        return -np.sign(self.pos) * np.maximum(np.abs(self.pos) - limit, 0.0) * cfg.wall_stiffness

    def fire_sparks(self, key: Hashable) -> None:
        for q in self.queries:
            if q.key == key:
                self.sparks += [(key, int(t), self.time) for t, w in zip(q.tokens, q.weights, strict=True) if w > _MIN_WEIGHT]

    def explode(self, center: np.ndarray, speed: float, radius: float) -> None:
        rel = self.pos - center
        r2 = np.einsum("ij,ij->i", rel, rel)
        self.vel += rel * (speed * gaussian(r2, radius) / (np.sqrt(r2) + 1e-6))[:, None]


class Attention(Effect):
    name = "Atenção (LLM)"

    def __init__(self, env: EffectEnv) -> None:
        super().__init__(env)
        cfg = self._cfg = env.config.attention
        self.sim = AttentionSimulation(cfg, env.config.flow, env.viewport.aspect)
        self._atlas = GlyphAtlas(env.ctx, cfg.tokens, env.config.debug.font_names, cfg.font_size)
        self._sprites = SpriteBatch(env.ctx, env.shaders, self._atlas, len(cfg.tokens))
        max_queries = 3
        segments = max_queries * (cfg.top_k + _MAX_PAIRS) * cfg.arc_segments
        self._lines = PrimitiveBatch(env.ctx, env.shaders, replace(env.config.render, max_line_segments=segments))
        self._glows = GlowBatch(env.ctx, env.shaders, 64)

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        self.sim.step(dt, hands, self.env.viewport.aspect)

    def on_gesture(self, event: GestureEvent) -> None:
        if event.kind is GestureKind.PINCH_START:
            self.sim.fire_sparks(event.slot)
        elif event.kind is GestureKind.EXPLODE:
            self.sim.explode(event.position, self._cfg.explode_speed, self._cfg.explode_radius)

    def draw(self) -> None:
        cfg, sim = self._cfg, self.sim
        palette, viewport = self.env.palette, self.env.viewport
        primary, secondary, highlight = (np.asarray(c) for c in (palette.primary, palette.secondary, palette.highlight))
        fade = self.visibility * (0.7 + 0.3 * self.intensity)

        for q in sim.queries:
            if q.strength < 0.01 or len(q.tokens) == 0:
                continue
            for rank, (token, weight) in enumerate(zip(q.tokens, q.weights, strict=True)):
                if weight < _MIN_WEIGHT:
                    continue
                bend = cfg.arc_bend * (1 if rank % 2 else -1)
                pts = arc(sim.pos[token], q.point, bend, cfg.arc_segments)
                color = (*(secondary + (highlight - secondary) * weight), weight * cfg.line_intensity * q.strength * fade)
                self._lines.add_lines(pts[:-1], pts[1:], cfg.line_px * (0.6 + 0.8 * weight), color, viewport)
            order = np.argsort(-q.weights)
            pairs = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))][:_MAX_PAIRS]
            for i, j in pairs:
                w = math.sqrt(q.weights[i] * q.weights[j])
                pts = arc(sim.pos[q.tokens[i]], sim.pos[q.tokens[j]], 0.35, cfg.arc_segments)
                color = (*primary, w * cfg.self_attention_intensity * q.strength * fade)
                self._lines.add_lines(pts[:-1], pts[1:], cfg.line_px * 0.7, color, viewport)
            self._glows.add(q.point, 0.06, highlight * 0.25 * q.strength * fade)
        self._lines.flush(viewport, premultiplied=True)

        attention = sim.attention()
        colors = primary * cfg.base_intensity + (highlight * cfg.attended_intensity - primary * cfg.base_intensity) * attention[:, None]
        scale = viewport.pixels_to_world(1.0) * cfg.text_scale * (1.0 + 0.3 * attention)
        self._sprites.add(sim.pos, np.arange(sim.count), scale, colors * fade)
        self._sprites.flush(viewport)

        by_key = {q.key: q for q in sim.queries}
        for key, token, start in sim.sparks:
            q = by_key.get(key)
            if q is None:
                continue
            progress = min((sim.time - start) * cfg.spark_speed, 1.0)
            rank = int(np.flatnonzero(q.tokens == token)[0]) if token in q.tokens else 0
            pts = arc(sim.pos[token], q.point, cfg.arc_bend * (1 if rank % 2 else -1), cfg.arc_segments)
            point = pts[min(int(progress * cfg.arc_segments), cfg.arc_segments)]
            self._glows.add(point, cfg.spark_radius, highlight * cfg.spark_intensity * fade)
        self._glows.flush(viewport)

    def focus_scale(self) -> float:
        return self._cfg.focus_scale

    def trail(self) -> tuple[float, float]:
        return self._cfg.trail_tau_s, self._cfg.trail_diffusion_per_s

    def debug_lines(self) -> list[str]:
        attended = int((self.sim.attention() > 0.05).sum())
        return [f"tokens {self.sim.count}  atendidos {attended}  faíscas {len(self.sim.sparks)}"]

    def release(self) -> None:
        self._sprites.release()
        self._atlas.release()
        self._lines.release()
        self._glows.release()
