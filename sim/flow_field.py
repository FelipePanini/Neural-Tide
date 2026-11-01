"""Divergence-free 2D current: the curl of an evolving noise potential.

The potential is evaluated on a coarse grid each frame and the velocity is
sampled bilinearly per particle, so the cost of the noise does not grow with
the particle count. Because the velocity is a curl, the flow has no sources
or sinks: particles swirl and stream like water instead of clumping.
"""

from __future__ import annotations

import math

import numpy as np

from config import FlowFieldConfig
from sim.noise import SimplexNoise


class FlowField:
    """Curl-noise velocity field covering the screen plus a margin."""

    def __init__(self, cfg: FlowFieldConfig) -> None:
        self._cfg = cfg
        self._noise = SimplexNoise(cfg.seed)
        self._shape: tuple[int, int] = (0, 0)
        self._grid_x = np.zeros((0, 0), dtype=np.float32)
        self._grid_y = np.zeros((0, 0), dtype=np.float32)
        self._uv = np.zeros((0, 2), dtype=np.float32)
        self._origin = (0.0, 0.0)
        self._cell = (1.0, 1.0)
        self._inv_cell = (1.0, 1.0)
        self.time = 0.0

    @property
    def grid(self) -> np.ndarray:
        """Velocity on the grid, shape ``(rows, cols, 2)``."""
        return self._uv.reshape(self._shape[0], self._shape[1], 2)

    def swell(self) -> float:
        """Slow global multiplier on the flow speed (the tide breathing)."""
        cfg = self._cfg
        return 1.0 + cfg.swell_amplitude * math.sin(2.0 * math.pi * self.time / cfg.swell_period_s)

    def drift(self) -> np.ndarray:
        """Uniform tidal current whose direction wanders slowly."""
        cfg = self._cfg
        angle = 2.0 * math.pi * self.time / cfg.drift_turn_period_s
        angle += 1.5 * float(self._noise.noise3(np.float32(self.time * 0.02), np.float32(7.3), np.float32(1.1)))
        return np.array((math.cos(angle), math.sin(angle)), dtype=np.float32) * cfg.drift_speed

    def update(self, dt: float, half_width: float, half_height: float = 1.0) -> None:
        """Advance time by ``dt`` and rebuild the velocity grid for the given screen size."""
        cfg = self._cfg
        self.time += dt
        hw = half_width + cfg.margin
        hh = half_height + cfg.margin
        rows = cfg.grid_rows
        cols = max(2, round(rows * hw / hh))
        if (rows, cols) != self._shape or self._origin != (-hw, -hh):
            xs = np.linspace(-hw, hw, cols, dtype=np.float32)
            ys = np.linspace(-hh, hh, rows, dtype=np.float32)
            self._grid_x, self._grid_y = np.meshgrid(xs, ys)
            self._shape = (rows, cols)
            self._origin = (-hw, -hh)
            self._cell = (float(xs[1] - xs[0]), float(ys[1] - ys[0]))
            self._inv_cell = (1.0 / self._cell[0], 1.0 / self._cell[1])

        potential = self._noise.fbm(
            self._grid_x * cfg.scale,
            self._grid_y * cfg.scale,
            self.time * cfg.evolution,
            cfg.octaves,
            cfg.lacunarity,
            cfg.gain,
        )
        d_dy, d_dx = np.gradient(potential, self._cell[1], self._cell[0])
        uv = np.stack((d_dy, -d_dx), axis=-1)  # curl of a scalar potential
        rms = float(np.sqrt(np.mean(np.sum(uv * uv, axis=-1)))) or 1.0
        self._uv = (uv * (cfg.speed * self.swell() / rms)).reshape(-1, 2).astype(np.float32)
        self._uv += self.drift()

    def sample(self, pos: np.ndarray) -> np.ndarray:
        """Bilinearly interpolated velocity at world positions ``(N, 2)``."""
        rows, cols = self._shape
        gx = (pos[:, 0] - self._origin[0]) * self._inv_cell[0]
        gy = (pos[:, 1] - self._origin[1]) * self._inv_cell[1]
        np.clip(gx, 0.0, cols - 1.001, out=gx)
        np.clip(gy, 0.0, rows - 1.001, out=gy)
        ix = gx.astype(np.intp)
        iy = gy.astype(np.intp)
        fx = (gx - ix)[:, None]
        fy = (gy - iy)[:, None]
        idx = iy * cols + ix
        uv = self._uv
        top = uv[idx] + (uv[idx + 1] - uv[idx]) * fx
        bottom = uv[idx + cols] + (uv[idx + cols + 1] - uv[idx + cols]) * fx
        return top + (bottom - top) * fy
