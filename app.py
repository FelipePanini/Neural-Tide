"""Main loop: hand source → eased hand state → layers → window."""

from __future__ import annotations

import time

import pygame

from config import Config
from core.easing import damp
from core.hand import HandSource
from core.hand_state import HandStateSmoother
from core.input_mock import MockHandSource
from core.stats import RateMeter
from core.tracker import HandTracker
from render.overlay import DebugInfo, DebugOverlay
from render.renderer import Renderer
from render.window import create_window

KEY_HINTS = "[F1 ou D] debug  [C] fundo  [ESC] sair"
MOCK_HINTS = (
    "mock: roda ou setas = dedos · clique = pinça",
    "      botão direito ou F = punho · H = 2ª mão",
)
# F1 needs Fn on most Mac keyboards, so D toggles the overlay too.
DEBUG_KEYS = (pygame.K_F1, pygame.K_d)


class App:
    """Owns the window, the hand source and the per-frame loop."""

    def __init__(self, cfg: Config, mock: bool = False) -> None:
        self.cfg = cfg
        self.mock = MockHandSource(cfg) if mock else None
        self.source: HandSource = self.mock if self.mock is not None else HandTracker(cfg)
        self._camera_target = cfg.background.start_with_camera
        self._camera_mix = 1.0 if self._camera_target else 0.0
        self._fps = RateMeter(cfg.debug.rate_window_s)
        self._running = True

    def run(self) -> None:
        # Start the source before the window: on macOS the first camera open
        # may block on the permission prompt.
        self.source.start()
        ctx, size = create_window(self.cfg.window)
        renderer = Renderer(ctx, size, self.cfg)
        overlay = DebugOverlay(renderer, self.cfg.debug)
        smoother = HandStateSmoother(self.cfg.hand_state)
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
                self._camera_mix = damp(
                    self._camera_mix, float(self._camera_target), self.cfg.background.fade_tau_s, dt
                )

                renderer.begin_frame()
                renderer.draw_background(self._camera_mix)
                overlay.draw(hands, self._debug_info(), dt)
                pygame.display.flip()

                self._fps.tick(now)
                if self.cfg.window.max_fps:
                    clock.tick(self.cfg.window.max_fps)
        finally:
            self.source.stop()
            renderer.release()
            pygame.quit()

    def _debug_info(self) -> DebugInfo:
        if self.mock is not None:
            background = "preto (mock não usa câmera)"
            hints = (*MOCK_HINTS, f"      dedos do mock: {self.mock.finger_count}", KEY_HINTS)
        else:
            background = "câmera" if self._camera_target else "preto"
            hints = (KEY_HINTS,)
        return DebugInfo(
            render_fps=self._fps.rate,
            source=self.source.status(),
            mode="— (nenhum ainda)",
            gesture="—",
            background=background,
            hints=hints,
        )

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
