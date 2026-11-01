"""Simplex noise and the curl-noise flow field."""

from __future__ import annotations

import numpy as np

from config import CONFIG
from sim.flow_field import FlowField
from sim.noise import SimplexNoise

RNG = np.random.default_rng(0)
POINTS = RNG.uniform(-40, 40, (3, 20000)).astype(np.float32)


def test_noise_is_deterministic_per_seed() -> None:
    np.testing.assert_array_equal(SimplexNoise(5).noise3(*POINTS), SimplexNoise(5).noise3(*POINTS))
    assert not np.allclose(SimplexNoise(5).noise3(*POINTS), SimplexNoise(6).noise3(*POINTS))


def test_noise_range_and_spread() -> None:
    values = SimplexNoise(1).noise3(*POINTS)
    assert values.min() >= -1.05 and values.max() <= 1.05
    assert 0.25 < values.std() < 0.6
    assert abs(values.mean()) < 0.02


def test_noise_is_continuous() -> None:
    noise = SimplexNoise(2)
    step = noise.noise3(POINTS[0] + 1e-3, POINTS[1], POINTS[2]) - noise.noise3(*POINTS)
    assert np.abs(step).max() < 0.02


def test_fbm_stays_normalised() -> None:
    values = SimplexNoise(3).fbm(POINTS[0], POINTS[1], POINTS[2], octaves=4, lacunarity=2.0, gain=0.5)
    assert np.abs(values).max() <= 1.05


def _field() -> FlowField:
    field = FlowField(CONFIG.flow)
    field.update(0.5, half_width=16 / 9)
    return field


def test_flow_is_nearly_divergence_free() -> None:
    grid = _field().grid
    div = np.gradient(grid[..., 0], axis=1) + np.gradient(grid[..., 1], axis=0)
    shear = np.abs(np.gradient(grid[..., 0], axis=1)).mean()
    assert np.abs(div).mean() < 0.05 * shear


def test_flow_speed_matches_config() -> None:
    field = _field()
    swirl = field.grid - field.drift()
    rms = np.sqrt((swirl**2).sum(-1).mean())
    np.testing.assert_allclose(rms, CONFIG.flow.speed * field.swell(), rtol=1e-3)


def test_sampling_at_grid_nodes_returns_grid_values() -> None:
    field = FlowField(CONFIG.flow)
    field.update(0.0, half_width=1.0, half_height=1.0)
    rows, cols, _ = field.grid.shape
    hw = hh = 1.0 + CONFIG.flow.margin
    xs, ys = np.linspace(-hw, hw, cols), np.linspace(-hh, hh, rows)
    nodes = np.array([(xs[3], ys[5]), (xs[10], ys[20])], dtype=np.float32)
    np.testing.assert_allclose(field.sample(nodes), field.grid[[5, 20], [3, 10]], atol=1e-5)
