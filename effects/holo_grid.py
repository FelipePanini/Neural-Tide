"""Mode 3 — Grade holográfica (holographic grid).

A wireframe surface stretches to the horizon in perspective. It is a real
2D wave equation: every raised fingertip presses the surface right below it
(the camera ray through the fingertip meets the plane there), so a still
finger makes a dimple and a moving finger leaves a wake. A pinch drops a
ripple, a fist slams a big one, and idle raindrops plus a slow swell keep the
surface moving on its own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import moderngl
import numpy as np

from core.gestures import GestureEvent, GestureKind
from core.hand import FINGERTIPS
from core.hand_state import HandState
from effects.base import Effect, EffectEnv
from render.glow import GlowBatch
from render.projection import gl_matrix, look_at, ndc_to_plane, perspective
from sim.waves import WaveField

_CLIP_NEAR, _CLIP_FAR = 0.1, 60.0
_RAIN_SCREEN = ((-0.85, 0.85), (-0.9, 0.35))  # NDC box where idle drops land (lower part of the screen)


class HoloGrid(Effect):
    name = "Grade holográfica"

    def __init__(self, env: EffectEnv) -> None:
        super().__init__(env)
        cfg = self._cfg = env.config.holo
        self._rows, self._cols = cfg.rows + 1, cfg.cols + 1  # vertices
        self._cell = (2.0 * cfg.half_width / cfg.cols, (cfg.far - cfg.near) / cfg.rows)
        self.waves = WaveField(
            self._rows,
            self._cols,
            self._cell,
            cfg.wave_speed,
            cfg.damping,
            cfg.edge_cells,
            cfg.edge_damping,
            cfg.max_substep_s,
        )
        self._rng = np.random.default_rng(31)
        self._time = 0.0
        xs = -cfg.half_width + np.arange(self._cols) * self._cell[0]
        distances = cfg.near + np.arange(self._rows) * self._cell[1]
        self._swell_x, self._swell_d = np.meshgrid(xs, distances)
        self._eye = np.array((0.0, cfg.eye_height, 0.0))
        self._aspect = 0.0
        self._view_proj = np.eye(4)
        self._inverse = np.eye(4)
        self._hands: Sequence[HandState] = ()

        ctx = env.ctx
        cols_idx, rows_idx = np.meshgrid(np.arange(self._cols), np.arange(self._rows))
        grid = np.stack((cols_idx, rows_idx), axis=-1).reshape(-1, 2).astype(np.float32)
        v0 = (rows_idx[:-1, :-1] * self._cols + cols_idx[:-1, :-1]).ravel()
        quads = np.stack((v0, v0 + 1, v0 + self._cols, v0 + 1, v0 + self._cols + 1, v0 + self._cols), axis=1)
        self._prog = env.shaders.program("holo")
        self._grid_vbo = ctx.buffer(grid.tobytes())
        self._height_vbo = ctx.buffer(reserve=self._rows * self._cols * 4, dynamic=True)
        self._ibo = ctx.buffer(quads.astype(np.uint32).tobytes())
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._grid_vbo, "2f", "in_grid"), (self._height_vbo, "1f", "in_height")],
            index_buffer=self._ibo,
            index_element_size=4,
        )
        self._glows = GlowBatch(ctx, env.shaders, 32 + cfg.horizon_glow_count)

    # ------------------------------------------------------------------ geometry

    def _update_camera(self) -> None:
        aspect = self.env.viewport.aspect
        if aspect == self._aspect:
            return
        cfg = self._cfg
        view = look_at(self._eye, np.array((0.0, 0.0, -cfg.look_distance)))
        self._view_proj = perspective(cfg.fov_y_deg, aspect, _CLIP_NEAR, _CLIP_FAR) @ view
        self._inverse = np.linalg.inv(self._view_proj)
        self._aspect = aspect

    def _horizon_y(self) -> float:
        """Screen height (world units of the 2D layer) of the far edge of the plane."""
        far_point = self._view_proj @ np.array((0.0, 0.0, -self._cfg.far, 1.0))
        return float(far_point[1] / far_point[3])

    def _ndc_to_grid(self, ndc: np.ndarray) -> tuple[float, float] | None:
        hit = ndc_to_plane(ndc, self._inverse)
        if hit is None:
            return None
        cfg = self._cfg
        col = (hit[0] + cfg.half_width) / self._cell[0]
        row = (-hit[1] - cfg.near) / self._cell[1]
        if not (0.0 <= col < self._cols and 0.0 <= row < self._rows):
            return None
        return col, row

    def screen_to_grid(self, world_xy: np.ndarray) -> tuple[float, float] | None:
        """Grid point under a screen position (world space of the 2D layer)."""
        self._update_camera()
        return self._ndc_to_grid(np.array((world_xy[0] / self._aspect, world_xy[1])))

    def _radius_cells(self, radius: float) -> tuple[float, float]:
        return (radius / self._cell[0], radius / self._cell[1])

    def _impulse(self, world_xy: np.ndarray, amount: float, radius: float) -> None:
        grid = self.screen_to_grid(world_xy)
        if grid is not None:
            self.waves.push(grid[0], grid[1], amount, self._radius_cells(radius))

    # ------------------------------------------------------------------ effect API

    def update(self, dt: float, hands: Sequence[HandState]) -> None:
        self._hands = hands
        if dt <= 0.0:  # frozen
            return
        cfg = self._cfg
        self._time += dt
        self._update_camera()
        radius = self._radius_cells(cfg.finger_radius)
        for hand in hands:
            for tip_index, raised in zip(FINGERTIPS, hand.fingers, strict=True):
                if not raised:
                    continue
                grid = self.screen_to_grid(hand.points[tip_index])
                if grid is None:
                    continue
                speed = float(np.linalg.norm(hand.velocity[tip_index]))
                amount = (cfg.finger_force + speed * cfg.finger_speed_gain) * hand.presence * self.intensity * dt
                self.waves.push(grid[0], grid[1], amount, radius)
        if self._rng.random() < cfg.rain_rate * dt:
            (x0, x1), (y0, y1) = _RAIN_SCREEN
            drop = self._ndc_to_grid(np.array((self._rng.uniform(x0, x1), self._rng.uniform(y0, y1))))
            if drop is not None:
                self.waves.push(drop[0], drop[1], cfg.rain_impulse, self._radius_cells(cfg.rain_radius))
        self.waves.step(dt)

    def on_gesture(self, event: GestureEvent) -> None:
        cfg = self._cfg
        if event.kind is GestureKind.PINCH_START:
            self._impulse(event.position, cfg.pinch_impulse * self.intensity, cfg.pinch_radius)
        elif event.kind is GestureKind.EXPLODE:
            self._impulse(event.position, cfg.explode_impulse * self.intensity, cfg.explode_radius)

    def _swell(self) -> np.ndarray:
        cfg = self._cfg
        k = 2.0 * math.pi / cfg.swell_wavelength
        along = np.sin(k * self._swell_d - k * cfg.swell_speed * self._time)
        across = 0.7 + 0.3 * np.sin(0.37 * k * self._swell_x + 0.5 * self._time)
        return (cfg.swell_amplitude * along * across).astype(np.float32)

    def draw(self) -> None:
        cfg = self._cfg
        self._update_camera()
        palette = self.env.palette
        heights = self.waves.h + self._swell()
        self._height_vbo.write(np.ascontiguousarray(heights, dtype=np.float32).tobytes())

        prog = self._prog
        prog["u_view_proj"].write(gl_matrix(self._view_proj))
        prog["u_origin"].value = (-cfg.half_width, -cfg.near)
        prog["u_cell"].value = self._cell
        prog["u_height_scale"].value = cfg.height_scale * (0.6 + 0.4 * self.intensity)
        prog["u_eye"].value = tuple(self._eye)
        prog["u_primary"].value = palette.primary
        prog["u_secondary"].value = palette.secondary
        prog["u_highlight"].value = palette.highlight
        prog["u_line_px"].value = cfg.line_px
        prog["u_line_intensity"].value = cfg.line_intensity
        prog["u_crest_intensity"].value = cfg.crest_intensity
        prog["u_fill_intensity"].value = cfg.fill_intensity
        prog["u_fog_density"].value = cfg.fog_density
        prog["u_cols"].value = float(cfg.cols)
        prog["u_edge_fade"].value = cfg.edge_fade
        prog["u_intensity"].value = self.visibility * (0.7 + 0.3 * self.intensity)
        self._vao.render(moderngl.TRIANGLES)

        aspect = self.env.viewport.aspect
        xs = np.linspace(-aspect, aspect, cfg.horizon_glow_count)
        horizon = np.column_stack((xs, np.full_like(xs, self._horizon_y())))
        self._glows.add(horizon, cfg.horizon_glow_radius, np.asarray(palette.secondary) * cfg.horizon_glow_intensity * self.visibility)
        glow = np.asarray(palette.highlight) * cfg.contact_intensity * self.visibility
        for hand in self._hands:
            tips = hand.tips[np.array(hand.fingers)]
            if len(tips):
                self._glows.add(tips, cfg.contact_radius, glow * hand.presence)
        self._glows.flush(self.env.viewport)

    def trail(self) -> tuple[float, float]:
        return self._cfg.trail_tau_s, self._cfg.trail_diffusion_per_s

    def focus_scale(self) -> float:
        return self._cfg.focus_scale

    def debug_lines(self) -> list[str]:
        h = self.waves.h
        return [f"grade {self._cols}x{self._rows}  onda máx {np.abs(h).max():.2f}"]

    def release(self) -> None:
        self._glows.release()
        for obj in (self._vao, self._ibo, self._height_vbo, self._grid_vbo):
            obj.release()
