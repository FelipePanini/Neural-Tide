"""Vectorised 3D simplex noise (Gustavson's reference algorithm) in NumPy."""

from __future__ import annotations

import numpy as np

_F3 = 1.0 / 3.0
_G3 = 1.0 / 6.0
_GRAD3 = np.array(
    [
        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    ],
    dtype=np.float32,
)  # fmt: skip
_RADIUS_SQ = 0.6  # kernel radius² of each simplex corner
_SCALE = 32.0  # brings the sum of corner contributions to roughly [-1, 1]


class SimplexNoise:
    """Smooth pseudo-random scalar field; same seed, same field."""

    def __init__(self, seed: int = 0) -> None:
        perm = np.random.default_rng(seed).permutation(256)
        self._perm = np.concatenate((perm, perm)).astype(np.intp)
        self._grad_index = self._perm % len(_GRAD3)

    def _corner(self, gi: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        t = _RADIUS_SQ - x * x - y * y - z * z
        g = _GRAD3[gi]
        dot = g[..., 0] * x + g[..., 1] * y + g[..., 2] * z
        t = np.maximum(t, 0.0)
        t2 = t * t
        return t2 * t2 * dot

    def noise3(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | float) -> np.ndarray:
        """Noise at the given coordinates (broadcast together), values in about [-1, 1]."""
        x, y, z = np.broadcast_arrays(
            np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), np.asarray(z, dtype=np.float32)
        )
        s = (x + y + z) * _F3
        i = np.floor(x + s)
        j = np.floor(y + s)
        k = np.floor(z + s)
        t = (i + j + k) * _G3
        x0 = x - (i - t)
        y0 = y - (j - t)
        z0 = z - (k - t)

        # Which simplex we are in: step first along the largest offset, then
        # along the two largest (i.e. everything but the smallest).
        i1 = (x0 >= y0) & (x0 >= z0)
        j1 = (y0 > x0) & (y0 >= z0)
        k1 = ~(i1 | j1)
        k_small = (z0 <= x0) & (z0 <= y0)
        j_small = (y0 < z0) & (y0 <= x0)
        i2 = j_small | k_small
        j2 = ~j_small
        k2 = ~k_small

        x1 = x0 - i1 + _G3
        y1 = y0 - j1 + _G3
        z1 = z0 - k1 + _G3
        x2 = x0 - i2 + 2.0 * _G3
        y2 = y0 - j2 + 2.0 * _G3
        z2 = z0 - k2 + 2.0 * _G3
        x3 = x0 - 1.0 + 3.0 * _G3
        y3 = y0 - 1.0 + 3.0 * _G3
        z3 = z0 - 1.0 + 3.0 * _G3

        p, gidx = self._perm, self._grad_index
        ii = i.astype(np.intp) & 255
        jj = j.astype(np.intp) & 255
        kk = k.astype(np.intp) & 255
        i1, j1, k1 = i1.astype(np.intp), j1.astype(np.intp), k1.astype(np.intp)
        i2, j2, k2 = i2.astype(np.intp), j2.astype(np.intp), k2.astype(np.intp)
        g0 = gidx[ii + p[jj + p[kk]]]
        g1 = gidx[ii + i1 + p[jj + j1 + p[kk + k1]]]
        g2 = gidx[ii + i2 + p[jj + j2 + p[kk + k2]]]
        g3 = gidx[ii + 1 + p[jj + 1 + p[kk + 1]]]

        total = (
            self._corner(g0, x0, y0, z0)
            + self._corner(g1, x1, y1, z1)
            + self._corner(g2, x2, y2, z2)
            + self._corner(g3, x3, y3, z3)
        )
        return (_SCALE * total).astype(np.float32)

    def fbm(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray | float, octaves: int, lacunarity: float, gain: float
    ) -> np.ndarray:
        """Fractal sum of ``octaves`` noise layers, normalised back to about [-1, 1]."""
        total = np.zeros(np.broadcast(x, y, z).shape, dtype=np.float32)
        amplitude, frequency, norm = 1.0, 1.0, 0.0
        for octave in range(octaves):
            # Offset every octave in z so layers do not line up at the origin.
            total += amplitude * self.noise3(x * frequency, y * frequency, z * frequency + octave * 17.17)
            norm += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return total / norm
