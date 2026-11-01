"""Mapping between camera-image, screen-pixel and world coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Viewport:
    """Window geometry plus how the camera image is fitted into it.

    World space is aspect-correct: y runs from -1 (bottom) to 1 (top) and x from
    -aspect to +aspect, so circles stay round. The camera image is fitted with
    "cover" semantics: it fills the window and the excess is cropped, which is
    also why landmarks near the cropped edges can fall slightly off-screen.
    """

    width: int
    height: int
    image_aspect: float

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def uv_rect(self) -> tuple[float, float, float, float]:
        """Visible part of the camera image as ``(u0, v0, du, dv)`` in image space."""
        ratio = self.image_aspect / self.aspect
        if ratio > 1.0:  # image wider than the window: crop left and right
            du = 1.0 / ratio
            return (0.5 - du / 2.0, 0.0, du, 1.0)
        dv = ratio  # image taller than the window: crop top and bottom
        return (0.0, 0.5 - dv / 2.0, 1.0, dv)

    def image_to_world(self, uv: np.ndarray) -> np.ndarray:
        """Image-space points ``(..., >=2)`` to world space ``(..., 2)``."""
        u0, v0, du, dv = self.uv_rect
        sx = (uv[..., 0] - u0) / du
        sy = (uv[..., 1] - v0) / dv
        return np.stack(((2.0 * sx - 1.0) * self.aspect, 1.0 - 2.0 * sy), axis=-1)

    def screen_to_image(self, px: float, py: float) -> tuple[float, float]:
        """Window pixel (origin top-left) to image space."""
        u0, v0, du, dv = self.uv_rect
        return (u0 + px / self.width * du, v0 + py / self.height * dv)

    def world_to_pixels(self, xy: np.ndarray) -> np.ndarray:
        """World-space points ``(..., 2)`` to window pixels (origin top-left)."""
        px = (xy[..., 0] / self.aspect + 1.0) * 0.5 * self.width
        py = (1.0 - xy[..., 1]) * 0.5 * self.height
        return np.stack((px, py), axis=-1)

    def pixel_to_world_point(self, px: float, py: float) -> np.ndarray:
        """Window pixel (origin top-left) to a world-space point."""
        return np.array(((px / self.width * 2.0 - 1.0) * self.aspect, 1.0 - py / self.height * 2.0))

    def pixels_to_world(self, length_px: float) -> float:
        """Convert a length in pixels to world units."""
        return 2.0 * length_px / max(self.height, 1)
