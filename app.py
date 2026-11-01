"""Main loop: hand source → eased hand state → layers → window."""

from __future__ import annotations

import time

import numpy as np
import pygame

from config import Config
from core.easing import damp
from core.gestures import GestureEngine, GestureEvent, GestureFrame, GestureKind
from core.hand import HandSource
from core.hand_state import HandStateSmoother
from core.input_mock import MockHandSource
from core.stats import RateMeter
from core.tracker import HandTracker
from effects import EFFECTS, EffectEnv
from effects.manager import ModeManager
from render.gesture_ui import GestureUI
from render.overlay import DebugInfo, DebugOverlay
from render.palette import PaletteState
from render.renderer import Renderer
from render.window import create_window

KEY_HINTS = "[1-8] modo  [P] paleta  [F1 ou D] debug  [C] fundo  [ESC] sair"
MOCK_HINTS = (
    "mock: roda ou setas = dedos · clique = pinça",
    "      botão direito ou F = punho · G = chifrinho · H = 2ª mão",
)
# Gestures that finish a hold: their ring bursts when they fire.
_HOLD_ACTIONS = (GestureKind.SELECT_MODE, GestureKind.CYCLE_PALETTE, GestureKind.TOGGLE_FREEZE, GestureKind.EXPLODE)
# F1 needs Fn on most Mac keyboards, so D toggles the overlay too.
DEBUG_KEYS = (pygame.K_F1, pygame.K_d)
_NUMBER_KEYS = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8)
MODE_KEYS = {key: index for index, key in enumerate(_NUMBER_KEYS)}


class App:
    """Owns the window, the hand source and the per-frame loop."""

    def __init__(self, cfg: Config, mock: bool = False) -> None:
        self.cfg = cfg
        self.mock = MockHandSource(cfg) if mock else None
        self.source: HandSource = self.mock if self.mock is not None else HandTracker(cfg)
        self._camera_target = cfg.background.start_with_camera
        self._camera_mix = 1.0 if self._camera_target else 0.0
        self._fps = RateMeter(cfg.debug.rate_window_s)
        self._palette = PaletteState(cfg.palette)
        self._gestures = GestureEngine(cfg.gestures)
        self._last_gestures: GestureFrame | None = None
        self._modes: ModeManager | None = None
        self._renderer: Renderer | None = None
        self._ui: GestureUI | None = None
        self._running = True

    def run(self) -> None:
        # Start the source before the window: on macOS the first camera open
        # may block on the permission prompt.
        self.source.start()
        ctx, size = create_window(self.cfg.window)
        renderer = Renderer(ctx, size, self.cfg)
        overlay = DebugOverlay(renderer, self.cfg.debug)
        smoother = HandStateSmoother(self.cfg.hand_state)
        env = EffectEnv(ctx, renderer.shaders, self.cfg, renderer.viewport, self._palette)
        modes = self._modes = ModeManager(env, EFFECTS, self.cfg.modes)
        ui = self._ui = GestureUI(renderer, self.cfg.ui, self._palette, self.cfg.render.max_rings)
        self._renderer = renderer
        if self.mock is not None and pygame.mouse.get_focused():
            self._move_pointer(renderer, pygame.mouse.get_pos())

        clock = pygame.time.Clock()
        last_camera_seq = -1
        last = time.perf_counter()
        try:
            while self._running:
                now = time.perf_counter()
                dt = min(now - last, self.cfg.window.max_dt)
                last = now
                for event in pygame.event.get():
                    self._handle_event(event, renderer, overlay)

                if self.mock is not None:
                    self.mock.update(now, dt, renderer.viewport.image_aspect)
                camera = self.source.camera_frame()
                if camera is not None and camera.seq != last_camera_seq:
                    show = self._camera_target or self._camera_mix > 0.001
                    renderer.set_camera_image(camera.image, upload=show)
                    last_camera_seq = camera.seq
                hands = smoother.update(self.source.latest(), renderer.viewport, dt)
                gestures = self._gestures.update(
                    hands, dt, now, current_mode=modes.active_index, mode_count=modes.count, frozen=modes.frozen
                )
                self._last_gestures = gestures
                for event in gestures.events:
                    self._on_gesture(event)
                modes.set_intensity_target(gestures.intensity)
                ui.update(gestures, hands, modes.frozen, dt)
                self._camera_mix = damp(
                    self._camera_mix, float(self._camera_target), self.cfg.background.fade_tau_s, dt
                )
                self._palette.update(dt)
                modes.update(dt, hands)

                renderer.begin_frame()
                renderer.draw_background(self._camera_mix)
                gain = renderer.begin_effects(dt, *modes.trail(self._default_trail()))
                modes.draw(gain)
                renderer.finish_effects()
                ui.draw(modes.intensity)
                overlay.draw(hands, self._debug_info(), dt)
                pygame.display.flip()

                self._fps.tick(now)
                if self.cfg.window.max_fps:
                    clock.tick(self.cfg.window.max_fps)
        finally:
            self.source.stop()
            ui.release()
            modes.release()
            renderer.release()
            pygame.quit()

    def _select_mode(self, index: int, origin: np.ndarray) -> None:
        """Switch mode with a ripple from ``origin`` and a toast naming the new mode."""
        if self._modes is None or not self._modes.select(index):
            return
        if self._renderer is not None:
            self._renderer.postfx.ripple(origin, self._palette.glow)
        if self._ui is not None:
            self._ui.notify(f"modo {index + 1} · {self._modes.names()[index]}")

    def _cycle_palette(self) -> None:
        self._palette.cycle()
        if self._ui is not None:
            self._ui.notify(f"paleta · {self._palette.name}")

    def _on_gesture(self, event: GestureEvent) -> None:
        modes, ui = self._modes, self._ui
        if modes is None or ui is None:
            return
        if event.kind in _HOLD_ACTIONS:
            ui.completed(event.slot, event.position)
        if event.kind is GestureKind.SELECT_MODE and event.mode is not None:
            self._select_mode(event.mode, event.position)
        elif event.kind in (GestureKind.NEXT_MODE, GestureKind.PREVIOUS_MODE):
            step = 1 if event.kind is GestureKind.NEXT_MODE else -1
            self._select_mode((modes.active_index + step) % modes.count, event.position)
        elif event.kind is GestureKind.CYCLE_PALETTE:
            self._cycle_palette()
        elif event.kind is GestureKind.TOGGLE_FREEZE:
            ui.notify("congelado" if modes.toggle_freeze() else "descongelado")
        else:  # pinch and explosion belong to the active mode
            modes.dispatch(event)

    def _gesture_status(self) -> str:
        """Debug line: current pose of each hand plus the last gesture that fired."""
        frame = self._last_gestures
        poses = " · ".join(pose.value for pose in frame.poses.values()) if frame and frame.poses else "—"
        last = self._gestures.last_event
        return f"{poses}   último: {last.kind.value}" if last else poses

    def _default_trail(self) -> tuple[float, float]:
        return self.cfg.postfx.trail_tau_s, self.cfg.postfx.trail_diffusion_per_s

    def _debug_info(self) -> DebugInfo:
        if self.mock is not None:
            background = "preto (mock não usa câmera)"
            hints = (*MOCK_HINTS, f"      dedos do mock: {self.mock.finger_count}", KEY_HINTS)
        else:
            background = "câmera" if self._camera_target else "preto"
            hints = (KEY_HINTS,)
        effect = self._modes.active if self._modes is not None else None
        return DebugInfo(
            render_fps=self._fps.rate,
            source=self.source.status(),
            mode=f"{self._modes.active_index + 1} · {effect.name}" if self._modes and effect else "—",
            gesture=self._gesture_status(),
            background=background,
            details=(*self._mode_details(), *(effect.debug_lines() if effect else ())),
            hints=hints,
        )

    def _mode_details(self) -> tuple[str, ...]:
        if self._modes is None:
            return ()
        frozen = "  CONGELADO" if self._modes.frozen else ""
        return (f"intensidade {self._modes.intensity:.2f}  paleta {self._palette.name}{frozen}",)

    def _move_pointer(self, renderer: Renderer, pos: tuple[int, int]) -> None:
        if self.mock is not None:
            self.mock.set_pointer(renderer.viewport.screen_to_image(*pos))

    def _handle_event(self, event: pygame.event.Event, renderer: Renderer, overlay: DebugOverlay) -> None:
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.key in DEBUG_KEYS:
                overlay.toggle()
            elif event.key == pygame.K_c:
                self._camera_target = not self._camera_target
            elif event.key in MODE_KEYS:
                self._select_mode(MODE_KEYS[event.key], np.zeros(2))
            elif event.key == pygame.K_p:
                self._cycle_palette()
            elif self.mock is not None:
                self._handle_mock_event(event, renderer)
        elif event.type == pygame.VIDEORESIZE:
            renderer.resize(pygame.display.get_window_size())
        elif self.mock is not None:
            self._handle_mock_event(event, renderer)

    def _handle_mock_event(self, event: pygame.event.Event, renderer: Renderer) -> None:
        assert self.mock is not None
        if event.type == pygame.MOUSEMOTION:
            self._move_pointer(renderer, event.pos)
        elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            pressed = event.type == pygame.MOUSEBUTTONDOWN
            if event.button == pygame.BUTTON_LEFT:
                self.mock.set_pinch(pressed)
            elif event.button == pygame.BUTTON_RIGHT:
                self.mock.set_fist(pressed)
        elif event.type == pygame.MOUSEWHEEL:
            # "Natural" scrolling (macOS default) reports flipped values.
            self.mock.scroll(-event.precise_y if event.flipped else event.precise_y, time.perf_counter())
        elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
            pressed = event.type == pygame.KEYDOWN
            if event.key == pygame.K_f:
                self.mock.set_fist(pressed)
            elif event.key == pygame.K_g:
                self.mock.set_horns(pressed)
            elif pressed and event.key == pygame.K_UP:
                self.mock.change_fingers(1)
            elif pressed and event.key == pygame.K_DOWN:
                self.mock.change_fingers(-1)
            elif pressed and event.key == pygame.K_h:
                self.mock.toggle_second_hand()
        elif event.type == pygame.WINDOWLEAVE:
            self.mock.set_pointer(None)
        elif event.type == pygame.WINDOWENTER:
            self._move_pointer(renderer, pygame.mouse.get_pos())
