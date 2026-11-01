"""Simulations of modes 2 and 3, without the GPU."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from config import CONFIG
from effects.neural_net import NeuralSimulation
from render.projection import look_at, ndc_to_plane, perspective
from sim.waves import WaveField

HALF_WIDTH = 16 / 9
DT = 1 / 60
QUIET = replace(CONFIG.neural, idle_pulse_rate=0.0)  # no spontaneous pulses: deterministic tests


def _net() -> NeuralSimulation:
    return NeuralSimulation(QUIET, CONFIG.flow, HALF_WIDTH)


def _run(sim: NeuralSimulation, seconds: float) -> None:
    for _ in range(round(seconds / DT)):
        sim.step(DT, [], HALF_WIDTH)


def test_pulse_spreads_hop_by_hop_and_fades() -> None:
    sim = _net()
    assert sim.fire_near(np.zeros(2), 1.0)
    _run(sim, 6.0)  # ~20 hops at most before the energy runs out
    fired = np.isfinite(sim.fire_time)
    assert fired.sum() > 10  # reached well beyond the first node
    assert sim.energy[fired].min() < 0.9 * sim.energy[fired].max()  # energy drops along the way
    assert not np.isfinite(sim.arrival).any()  # and the wave died out


def test_refractory_node_does_not_fire_twice() -> None:
    sim = _net()
    assert sim.fire(0, 1.0)
    assert not sim.fire(0, 1.0)


def test_nodes_stay_on_screen() -> None:
    sim = _net()
    sim.explode(np.zeros(2), speed=4.0, radius=1.0)
    _run(sim, 5.0)
    assert (np.abs(sim.pos[:, 0]) < HALF_WIDTH + 0.1).all() and (np.abs(sim.pos[:, 1]) < 1.1).all()


def test_frozen_network_does_not_move() -> None:
    sim = _net()
    before = sim.pos.copy()
    sim.step(0.0, [], HALF_WIDTH)
    np.testing.assert_array_equal(sim.pos, before)


def _waves() -> WaveField:
    cfg = CONFIG.holo
    return WaveField(97, 129, (0.125, 0.145), cfg.wave_speed, cfg.damping, cfg.edge_cells, cfg.edge_damping, cfg.max_substep_s)


def test_ripple_travels_at_the_wave_speed() -> None:
    waves = _waves()
    waves.push(64, 48, 5.0, (1.5, 1.5))
    for _ in range(30):  # 0.5 s
        waves.step(DT)
    row = np.abs(waves.h[48, 64:])
    crest = int(np.argmax(row)) * 0.125  # world units from the drop
    expected = CONFIG.holo.wave_speed * 0.5
    assert abs(crest - expected) < 0.35 * expected


def test_waves_stay_stable_at_low_frame_rates_and_settle() -> None:
    waves = _waves()
    waves.push(64, 48, 5.0, (1.5, 1.5))
    for _ in range(300):  # 10 s at 30 fps
        waves.step(1 / 30)
    assert np.isfinite(waves.h).all()
    assert np.abs(waves.h).max() < 0.05


def test_screen_ray_hits_the_ground_below_the_horizon() -> None:
    view = look_at(np.array((0.0, 1.6, 0.0)), np.array((0.0, 0.0, -6.0)))
    inverse = np.linalg.inv(perspective(55.0, 16 / 9, 0.1, 60.0) @ view)
    near = ndc_to_plane(np.array((0.0, -0.9)), inverse)
    far = ndc_to_plane(np.array((0.0, 0.3)), inverse)
    assert near is not None and far is not None
    assert -far[1] > -near[1] > 0.0  # higher on screen = further away
    assert ndc_to_plane(np.array((0.0, 0.95)), inverse) is None  # sky
