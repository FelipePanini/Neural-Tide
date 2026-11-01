"""Small 3D camera helpers for effects that draw in perspective."""

from __future__ import annotations

import math

import numpy as np


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """OpenGL-style perspective matrix (row-major numpy, column vectors)."""
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array((0.0, 1.0, 0.0))) -> np.ndarray:
    """View matrix for a camera at ``eye`` looking at ``target``."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    true_up = np.cross(side, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = side, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def gl_matrix(matrix: np.ndarray) -> bytes:
    """Bytes for a ``mat4`` uniform (GLSL expects column-major)."""
    return np.ascontiguousarray(matrix.T, dtype=np.float32).tobytes()


def ndc_to_plane(ndc: np.ndarray, inverse_view_proj: np.ndarray) -> np.ndarray | None:
    """Where the camera ray through ``ndc`` (x, y) meets the ground plane y = 0.

    Returns ``(x, z)`` or ``None`` when the ray points at or above the horizon.
    """
    near = inverse_view_proj @ np.array((ndc[0], ndc[1], -1.0, 1.0))
    far = inverse_view_proj @ np.array((ndc[0], ndc[1], 1.0, 1.0))
    near, far = near[:3] / near[3], far[:3] / far[3]
    direction = far - near
    if direction[1] >= -1e-9:
        return None
    t = -near[1] / direction[1]
    hit = near + direction * t
    return np.array((hit[0], hit[2]))
