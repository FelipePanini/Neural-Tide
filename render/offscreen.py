"""The full pipeline rendering into an offscreen framebuffer (tests, demo GIFs).

Hands come in as synthetic observations and go through the same filtering,
easing, modes, spotlights, post-processing and gesture UI as in the app.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import moderngl
import numpy as np

from config import Config
from core.gestures import GestureEvent, GestureFrame
from core.hand import HandObservation, SourceStatus, TrackingFrame, palm_center
from core.hand_state import HandState, HandStateSmoother
from core.input_mock import mock_silhouette
from core.tracker import HandFilterBank
from effects import EFFECTS, EffectEnv
from effects.base import Effect
from effects.manager import ModeManager
from render.focus import FocusState
from render.gesture_ui import GestureUI
from render.overlay import DebugInfo, DebugOverlay
from render.palette import PaletteState
from render.renderer import Renderer


class OffscreenStage:
    def __init__(
        self,
        ctx: moderngl.Context,
        size: tuple[int, int],
        cfg: Config,
        *,
        mode: int = 0,
        focus: bool = True,
        overlay: bool = False,
        effects: Sequence[type[Effect]] = EFFECTS,
    ) -> None:
        self.cfg = cfg = replace(cfg, debug=replace(cfg.debug, start_visible=overlay), modes=replace(cfg.modes, initial=mode))
        self.size = size
        self.fbo = ctx.simple_framebuffer(size)
        self.renderer = Renderer(ctx, size, cfg, target=self.fbo)
        self.palette = PaletteState(cfg.palette)
        self._palms: list[np.ndarray] = []
        env = EffectEnv(ctx, self.renderer.shaders, cfg, self.renderer.viewport, self.palette, self._silhouette)
        self.modes = ModeManager(env, effects, cfg.modes)
        self.focus = FocusState(replace(cfg.focus, start_enabled=focus))
        self.ui = GestureUI(self.renderer, cfg.ui, self.palette, cfg.render.max_rings)
        self.overlay = DebugOverlay(self.renderer, cfg.debug)
        self._bank = HandFilterBank(cfg.tracker, cfg.geometry)
        self._smoother = HandStateSmoother(cfg.hand_state)
        self.time = 0.0
        self._seq = 0
        self.camera_visibility = 0.0
        self.hands: list[HandState] = []

    def _silhouette(self) -> np.ndarray:
        return mock_silhouette(self.cfg.segmentation.mock_size, self._palms)

    def set_camera_image(self, image: np.ndarray, visibility: float = 1.0) -> None:
        self.renderer.set_camera_image(image)
        self.camera_visibility = visibility

    def select_mode(self, index: int, origin: np.ndarray | None = None) -> None:
        """Switch mode the way the app does: ripple from ``origin`` plus a toast."""
        if self.modes.select(index):
            self.renderer.postfx.ripple(np.zeros(2) if origin is None else origin, self.palette.glow)
            self.ui.notify(f"modo {index + 1} · {self.modes.names()[index]}")

    def step(self, observations: Sequence[HandObservation], dt: float, events: Sequence[GestureEvent] = ()) -> None:
        """Advance and draw one frame."""
        self.time += dt
        self._seq += 1
        frame = TrackingFrame(self._seq, self.time, self._bank.update(observations, self.time))
        self.hands = self._smoother.update(frame, self.renderer.viewport, dt)
        self._palms = [palm_center(obs.landmarks[:, :2]) for obs in observations]
        for event in events:
            self.modes.dispatch(event)
        self.palette.update(dt)
        self.modes.update(dt, self.hands)
        spots = self.focus.update(self.hands, dt, self.renderer.viewport.aspect, self.modes.focus_scale())
        spots = sorted(spots + self.modes.focus_spots(), key=lambda spot: -spot.weight)
        self.renderer.postfx.set_focus(spots, self.focus.amount * self.modes.focus_strength())
        self.ui.update(GestureFrame((), (), {}, None), self.hands, self.modes.frozen, dt)

        r = self.renderer
        r.begin_frame()
        r.draw_background(self.camera_visibility)
        defaults = (self.cfg.postfx.trail_tau_s, self.cfg.postfx.trail_diffusion_per_s)
        self.modes.draw(r.begin_effects(dt, *self.modes.trail(defaults)))
        r.finish_effects()
        self.ui.draw(self.modes.intensity)
        info = DebugInfo(1.0 / dt, SourceStatus("mock (mouse)", 1.0 / dt), "—", "—", "câmera")
        self.overlay.draw(self.hands, info, dt)

    def read(self) -> np.ndarray:
        """The current frame as an (H, W, 3) uint8 array, top row first."""
        w, h = self.size
        return np.frombuffer(self.fbo.read(components=3), dtype=np.uint8).reshape(h, w, 3)[::-1].copy()

    def release(self) -> None:
        self.ui.release()
        self.modes.release()
        self.renderer.release()
        self.fbo.release()
