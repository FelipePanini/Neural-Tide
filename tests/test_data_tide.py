"""Particle simulation of mode 1, without the GPU."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import CONFIG
from effects.data_tide import TideSimulation
from sim.forces import Stirrer

HALF_WIDTH = 16 / 9
DT = 1 / 60
CFG = replace(CONFIG.data_tide, particle_count=6000)


def _sim() -> TideSimulation:
    return TideSimulation(CFG, CONFIG.flow, HALF_WIDTH)


def _hand(center=(0.0, 0.0), velocity=(0.0, 0.0), spin=1.0, pinch=0.0) -> Stirrer:  # type: ignore[no-untyped-def]
    return Stirrer(np.array(center, np.float32), np.array(velocity, np.float32), weight=1.0, spin=spin, pinch=pinch)


def _run(sim: TideSimulation, stirrers: list[Stirrer], seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        sim.step(DT, stirrers, HALF_WIDTH)


def _near(sim: TideSimulation, center: tuple[float, float], radius: float) -> np.ndarray:
    return np.linalg.norm(sim.pos - np.array(center), axis=1) < radius


def _angular_momentum(sim: TideSimulation, center: tuple[float, float], radius: float) -> float:
    mask = _near(sim, center, radius)
    rel = sim.pos[mask] - np.array(center)
    return float(np.mean(rel[:, 0] * sim.vel[mask, 1] - rel[:, 1] * sim.vel[mask, 0]))


def test_right_hand_spins_counter_clockwise_and_left_hand_clockwise() -> None:
    right, left = _sim(), _sim()
    _run(right, [_hand(spin=1.0)], 1.0)
    _run(left, [_hand(spin=-1.0)], 1.0)
    assert _angular_momentum(right, (0.0, 0.0), 0.4) > 0.02
    assert _angular_momentum(left, (0.0, 0.0), 0.4) < -0.02


def test_moving_hand_drags_particles_along() -> None:
    sim = _sim()
    _run(sim, [_hand(velocity=(1.5, 0.0))], 0.8)
    mean_vx = sim.vel[_near(sim, (0.0, 0.0), 0.25), 0].mean()
    assert mean_vx > 0.5


def test_pinch_gathers_particles_around_the_hand() -> None:
    # Particles carry inertia, so a fast vortex flings them outwards (an empty
    # eye); the pinch suction wins against that and packs them into a ring.
    relaxed, pinched = _sim(), _sim()
    _run(relaxed, [_hand()], 1.5)
    _run(pinched, [_hand(pinch=1.0)], 1.5)
    assert _near(pinched, (0.0, 0.0), 0.5).sum() > 2 * _near(relaxed, (0.0, 0.0), 0.5).sum()


def test_particles_respond_gradually_not_instantly() -> None:
    sim = _sim()
    _run(sim, [], 0.5)
    before = sim.vel.copy()
    sim.step(DT, [_hand(velocity=(3.0, 0.0))], HALF_WIDTH)
    alive = sim.age > 0.0  # respawned particles restart invisible, with the local flow velocity
    jump = np.linalg.norm(sim.vel - before, axis=1)[alive].max()
    assert jump < 0.5  # one frame only moves part of the way: no teleporting velocities


def test_state_stays_finite_and_bounded() -> None:
    sim = _sim()
    _run(sim, [_hand(velocity=(4.0, -2.0), pinch=1.0), _hand((0.8, 0.3), spin=-1.0)], 2.0)
    assert np.isfinite(sim.pos).all() and np.isfinite(sim.vel).all()
    assert np.linalg.norm(sim.vel, axis=1).max() <= CFG.max_speed + 1e-4
    margin = CFG.margin + CFG.max_speed * DT
    assert (np.abs(sim.pos[:, 0]) <= HALF_WIDTH + margin).all()
    assert (np.abs(sim.pos[:, 1]) <= 1.0 + margin).all()


def test_respawned_particles_fade_in() -> None:
    sim = _sim()
    sim.age[:100] = sim.life[:100]  # force these to die this frame
    sim.step(DT, [], HALF_WIDTH)
    assert sim.alpha()[:100].max() < 0.05


def test_idle_tide_keeps_moving_and_phantom_yields_to_hands() -> None:
    sim = _sim()
    _run(sim, [], 1.0)
    assert np.linalg.norm(sim.vel, axis=1).mean() > 0.05
    assert sim.idle_weight > 0.99
    _run(sim, [_hand()], 3.0)
    assert sim.idle_weight < 0.1
