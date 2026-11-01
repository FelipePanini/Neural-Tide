"""Simulations of modes 4–8, without the GPU."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from config import CONFIG
from core.hand import LEFT
from core.input_mock import FINGER_PRESETS, mock_silhouette
from core.viewport import Viewport
from effects.ai_core import AICoreSimulation
from effects.attention import AttentionSimulation, arc, softmax_attention
from effects.code_rain import RainSimulation, flow_around_cylinder
from effects.light_ribbons import RibbonSimulation
from effects.particle_body import BodySimulation
from tests.test_gestures import make_hand

DT = 1 / 60
HALF_WIDTH = 16 / 9
VIEWPORT = Viewport(1280, 720, 16 / 9)


def _steps(step, seconds: float) -> None:  # type: ignore[no-untyped-def]
    for _ in range(round(seconds / DT)):
        step()


# ---------------------------------------------------------------- mode 4: AI core


def test_core_sits_between_two_hands_and_scales_with_their_distance() -> None:
    cfg = CONFIG.core
    sizes = []
    for spread in (0.12, 0.25):
        sim = AICoreSimulation(cfg)
        hands = [make_hand(FINGER_PRESETS[5], (0.5 - spread, 0.5)), make_hand(FINGER_PRESETS[5], (0.5 + spread, 0.5), slot=1, handedness=LEFT)]
        _steps(lambda: sim.step(DT, hands, HALF_WIDTH), 2.0)
        np.testing.assert_allclose(sim.center, (hands[0].palm + hands[1].palm) / 2, atol=0.02)
        sizes.append(sim.radius)
    assert sizes[1] > sizes[0] * 1.5


def test_wrist_roll_turns_the_core() -> None:
    cfg = replace(CONFIG.core, auto_spin=0.0)
    yaws = []
    for roll in (-30.0, 30.0):
        sim = AICoreSimulation(cfg)
        hand = make_hand(FINGER_PRESETS[5])
        hand = replace(hand, points=_rolled(hand.points, roll))
        _steps(lambda: sim.step(DT, [hand], HALF_WIDTH), 2.0)
        yaws.append(sim.yaw)
    assert yaws[0] * yaws[1] < 0  # opposite rolls spin it opposite ways


def _rolled(points: np.ndarray, degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    rot = np.array(((math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a))))
    center = points.mean(axis=0)
    return (points - center) @ rot.T + center


def _lag(sim: AICoreSimulation) -> float:
    target, _ = sim.targets()
    return float(np.linalg.norm(sim.pos - target, axis=1).mean())


def test_core_reforms_after_an_explosion() -> None:
    sim = AICoreSimulation(CONFIG.core)
    _steps(lambda: sim.step(DT, [], HALF_WIDTH), 2.0)
    resting = _lag(sim)  # the springs always trail the turning sphere a little
    sim.explode(3.0)
    _steps(lambda: sim.step(DT, [], HALF_WIDTH), 0.3)
    assert _lag(sim) > 5 * resting  # blown apart...
    _steps(lambda: sim.step(DT, [], HALF_WIDTH), 4.0)
    assert _lag(sim) < 2 * resting  # ...and back together


# ---------------------------------------------------------------- mode 5: ribbons


def test_ribbon_trails_behind_a_moving_finger_and_fades_when_curled() -> None:
    sim = RibbonSimulation(CONFIG.ribbons, CONFIG.flow)
    for i in range(60):
        sim.step(DT, [make_hand(FINGER_PRESETS[5], (0.3 + i * 0.005, 0.5))], HALF_WIDTH)
    row = sim.keys.index((0, 1))  # index finger
    head, tail = sim.points[row, 0], sim.points[row, -1]
    assert tail[0] < head[0] - 0.05  # the tail lags behind the motion
    _steps(lambda: sim.step(DT, [make_hand(FINGER_PRESETS[0], (0.6, 0.5))], HALF_WIDTH), 2.0)
    assert sim.alpha[sim.keys.index((0, 1))] < 0.05


def test_ribbons_of_a_gone_hand_are_removed() -> None:
    sim = RibbonSimulation(CONFIG.ribbons, CONFIG.flow)
    _steps(lambda: sim.step(DT, [make_hand(FINGER_PRESETS[5])], HALF_WIDTH), 0.5)
    _steps(lambda: sim.step(DT, [], HALF_WIDTH), 4.0)
    assert all(key[0] == "idle" for key in sim.keys)


# ---------------------------------------------------------------- mode 6: attention


def test_softmax_attention_prefers_the_closest_tokens() -> None:
    pos = np.array([(0.1, 0.0), (0.3, 0.0), (0.6, 0.0), (5.0, 0.0)], dtype=np.float32)
    tokens, weights = softmax_attention(pos, np.zeros(2), reach=1.0, top_k=3, temperature=0.05)
    assert list(tokens) == [0, 1, 2]  # the far token is out of reach
    assert math.isclose(weights.sum(), 1.0) and weights[0] > weights[1] > weights[2]
    _, sharp = softmax_attention(pos, np.zeros(2), reach=1.0, top_k=3, temperature=0.01)
    assert sharp[0] > weights[0]  # lower temperature = sharper focus


def test_attention_arcs_connect_token_and_query() -> None:
    a, b = np.array((0.0, 0.0)), np.array((1.0, 0.5))
    pts = arc(a, b, 0.3, 10)
    np.testing.assert_allclose(pts[0], a)
    np.testing.assert_allclose(pts[-1], b)


def test_hand_queries_attend_and_pinch_sends_sparks() -> None:
    sim = AttentionSimulation(CONFIG.attention, CONFIG.flow, HALF_WIDTH)
    hand = make_hand(FINGER_PRESETS[1])
    _steps(lambda: sim.step(DT, [hand], HALF_WIDTH), 1.0)
    query = next(q for q in sim.queries if q.key == hand.slot)
    assert len(query.tokens) > 0
    sim.fire_sparks(hand.slot)
    assert len(sim.sparks) == int((query.weights > 0.02).sum())


# ---------------------------------------------------------------- mode 7: particle body


def test_dust_is_born_on_the_silhouette_and_released_dust_floats_up() -> None:
    sim = BodySimulation(CONFIG.body, CONFIG.flow)
    mask = mock_silhouette(CONFIG.segmentation.mock_size, [])
    assert sim.emit(mask, VIEWPORT, 0.1) > 0
    alive = sim.alive()
    # Map the new particles back into the mask: they must sit on the person.
    u0, v0, du, dv = VIEWPORT.uv_rect
    uv = np.stack(((sim.pos[alive, 0] / VIEWPORT.aspect + 1) / 2 * du + u0, (1 - sim.pos[alive, 1]) / 2 * dv + v0), axis=1)
    h, w = mask.shape
    inside = mask[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]
    assert (inside > 0.2).mean() > 0.85  # the outline is favoured, so a few land just on the edge
    sim.burst()
    start = sim.pos[alive, 1].mean()
    _steps(lambda: sim.step(DT, [], HALF_WIDTH), 0.5)
    assert sim.pos[alive, 1].mean() > start


def test_no_mask_no_dust() -> None:
    sim = BodySimulation(CONFIG.body, CONFIG.flow)
    assert sim.emit(np.zeros((90, 160), dtype=np.float32), VIEWPORT, 0.1) == 0


# ---------------------------------------------------------------- mode 8: code rain


def test_flow_goes_around_the_cylinder() -> None:
    speed = np.array([1.0])
    far = flow_around_cylinder(np.array([[0.0, 50.0]]), speed, 0.2)
    np.testing.assert_allclose(far, [[0.0, -1.0]], atol=1e-3)  # undisturbed far away
    side = np.array([[0.2 * math.cos(0.7), 0.2 * math.sin(0.7)]])
    v = flow_around_cylinder(side * 1.01, speed, 0.2)
    radial = float((v @ side.T)[0, 0]) / 0.2
    assert abs(radial) < 0.05 * float(np.linalg.norm(v))  # slides along the surface
    inside = flow_around_cylinder(side * 0.3, speed, 0.2)
    assert float((inside @ side.T)[0, 0]) > 0  # pushed out of the hand


def test_rain_keeps_out_of_the_hand() -> None:
    sim = RainSimulation(CONFIG.rain, CONFIG.flow, HALF_WIDTH)
    hand = make_hand(FINGER_PRESETS[5], (0.5, 0.55))
    _steps(lambda: sim.step(DT, [hand], HALF_WIDTH), 4.0)
    radius = max(float(np.linalg.norm(hand.points - hand.points.mean(axis=0), axis=1).max()) * CONFIG.rain.obstacle_scale, CONFIG.rain.obstacle_min_radius)
    inside = np.linalg.norm(sim.pos - hand.palm, axis=1) < radius * 0.8
    assert inside.mean() < 0.01
    assert np.isfinite(sim.pos).all()
