"""2D wave equation on a grid (heights + velocities), solved explicitly in NumPy."""

from __future__ import annotations

import math

import numpy as np


class WaveField:
    """Height field ``h(z, x)`` obeying ``h_tt = c² ∇²h − damping · h_t``.

    Semi-implicit Euler with as many substeps as the CFL limit requires. A
    sponge layer of extra damping along the borders absorbs outgoing waves so
    they do not bounce back from the edges.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        cell: tuple[float, float],
        wave_speed: float,
        damping: float,
        edge_cells: int,
        edge_damping: float,
        max_substep_s: float,
    ) -> None:
        self.h = np.zeros((rows, cols), dtype=np.float32)
        self.v = np.zeros_like(self.h)
        self._dx, self._dz = cell
        self._c2 = wave_speed * wave_speed
        self._max_substep = max_substep_s
        zz, xx = np.mgrid[0:rows, 0:cols]
        to_edge = np.minimum.reduce([xx, cols - 1 - xx, zz, rows - 1 - zz]).astype(np.float32)
        sponge = np.clip(1.0 - to_edge / max(edge_cells, 1), 0.0, 1.0) ** 2
        self._damping = (damping + edge_damping * sponge).astype(np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        return self.h.shape  # type: ignore[return-value]

    def step(self, dt: float) -> None:
        if dt <= 0.0:
            return
        substeps = max(1, math.ceil(dt / self._max_substep))
        h_dt = dt / substeps
        inv_dx2, inv_dz2 = 1.0 / (self._dx * self._dx), 1.0 / (self._dz * self._dz)
        decay = np.exp(-self._damping * h_dt)
        lap = np.zeros_like(self.h)
        for _ in range(substeps):
            h = self.h
            lap[1:-1, 1:-1] = (h[1:-1, 2:] + h[1:-1, :-2] - 2.0 * h[1:-1, 1:-1]) * inv_dx2 + (
                h[2:, 1:-1] + h[:-2, 1:-1] - 2.0 * h[1:-1, 1:-1]
            ) * inv_dz2
            self.v += self._c2 * lap * h_dt
            self.v *= decay
            self.h += self.v * h_dt

    def push(self, col: float, row: float, amount: float, radius_cells: tuple[float, float]) -> None:
        """Add ``-amount × gaussian`` to the vertical velocity around grid point (col, row).

        Only a local window is touched, so many pushes per frame stay cheap.
        """
        rows, cols = self.shape
        rx, rz = radius_cells
        x0, x1 = max(int(col - 3 * rx), 0), min(int(col + 3 * rx) + 2, cols)
        z0, z1 = max(int(row - 3 * rz), 0), min(int(row + 3 * rz) + 2, rows)
        if x0 >= x1 or z0 >= z1:
            return
        xs = (np.arange(x0, x1) - col) / rx
        zs = (np.arange(z0, z1) - row) / rz
        bump = np.exp(-(zs[:, None] ** 2) - xs[None, :] ** 2).astype(np.float32)
        self.v[z0:z1, x0:x1] -= amount * bump
