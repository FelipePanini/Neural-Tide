"""Offscreen render of full frames: shaders compile, effects and overlay draw."""

from __future__ import annotations

import math
import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

moderngl = pytest.importorskip("moderngl")

from config import CONFIG  # noqa: E402
from core.hand import SourceStatus, TrackingFrame  # noqa: E402
from core.hand_state import HandStateSmoother  # noqa: E402
from core.input_mock import FINGER_PRESETS, synthetic_hand  # noqa: E402
from core.tracker import HandFilterBank  # noqa: E402
from effects import EFFECTS, EffectEnv  # noqa: E402
from effects.manager import ModeManager  # noqa: E402
from render.overlay import DebugInfo, DebugOverlay  # noqa: E402
from render.palette import PaletteState  # noqa: E402
from render.renderer import Renderer  # noqa: E402

SIZE = (480, 270)
FPS = 60.0
CFG = replace(CONFIG, debug=replace(CONFIG.debug, start_visible=True))


@pytest.fixture(scope="module")
def ctx():  # type: ignore[no-untyped-def]
    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # no GPU / driver in this environment
        pytest.skip(f"OpenGL 3.3 unavailable: {exc}")
    yield context
    context.release()


def hand_path(t: float) -> tuple[float, float]:
    """Image-space palm position circling the left half of the frame."""
    return (0.35 + 0.08 * math.cos(t * 2.0), 0.55 + 0.1 * math.sin(t * 2.0))


def render_frames(  # type: ignore[no-untyped-def]
    ctx,
    frames: int = 45,
    *,
    size: tuple[int, int] = SIZE,
    camera_visibility: float = 1.0,
    effects: bool = True,
    overlay_visible: bool = True,
    pinch: float = 0.0,
    hand: bool = True,
    capture_every: int = 0,
) -> list[np.ndarray]:
    """Simulate ``frames`` frames offscreen; returns the last frame (plus every ``capture_every``-th)."""
    fbo = ctx.simple_framebuffer(size)
    cfg = replace(CFG, debug=replace(CFG.debug, start_visible=overlay_visible))
    renderer = Renderer(ctx, size, cfg, target=fbo)
    overlay = DebugOverlay(renderer, cfg.debug)
    gradient = np.linspace(0, 255, 64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3)
    renderer.set_camera_image(gradient)
    modes = ModeManager(
        EffectEnv(ctx, renderer.shaders, cfg, renderer.viewport, PaletteState(cfg.palette)),
        EFFECTS if effects else [],
        cfg.modes,
    )
    bank = HandFilterBank(cfg.tracker, cfg.geometry)
    smoother = HandStateSmoother(cfg.hand_state)
    captured = []
    for i in range(frames):
        t = i / FPS
        aspect = renderer.viewport.image_aspect
        obs = [synthetic_hand(hand_path(t), FINGER_PRESETS[5], pinch=pinch, image_aspect=aspect)] if hand else []
        hands = smoother.update(TrackingFrame(i + 1, t, bank.update(obs, t)), renderer.viewport, 1.0 / FPS)
        modes.update(1.0 / FPS, hands)
        renderer.begin_frame()
        renderer.draw_background(camera_visibility)
        renderer.begin_effects(1.0 / FPS)
        modes.draw()
        renderer.finish_effects()
        info = DebugInfo(FPS, SourceStatus("mock (mouse)", FPS), "1", "—", "câmera")
        overlay.draw(hands, info, 1.0 / FPS)
        if i == frames - 1 or (capture_every and i % capture_every == 0):
            pixels = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(size[1], size[0], 3)
            captured.append(pixels[::-1].copy())  # OpenGL rows start at the bottom
    modes.release()
    renderer.release()
    fbo.release()
    return captured


def test_frame_renders_background_hand_and_panel(ctx) -> None:  # type: ignore[no-untyped-def]
    (pixels,) = render_frames(ctx, frames=30, effects=False)
    assert pixels.mean() > 1.0  # dimmed camera background is there
    assert pixels[60:220, 100:240].max() > 200  # bright landmarks / bones
    assert pixels[10:120, 10:200].max() > 150  # HUD text


def test_black_background_when_camera_hidden(ctx) -> None:  # type: ignore[no-untyped-def]
    (pixels,) = render_frames(ctx, frames=30, effects=False, camera_visibility=0.0)
    assert pixels[-20:, -60:].max() == 0  # bottom-right corner has neither camera nor overlay


def test_data_tide_lights_up_the_screen(ctx) -> None:  # type: ignore[no-untyped-def]
    (pixels,) = render_frames(ctx, camera_visibility=0.0, overlay_visible=False)
    lit = (pixels.max(axis=2) > 10).mean()
    assert 0.05 < lit < 0.95  # particles and trails cover part of the screen, not all of it
    assert pixels.max() > 120  # bright streaks / bloom


def test_tide_keeps_moving_without_hands(ctx) -> None:  # type: ignore[no-untyped-def]
    first, *_, last = render_frames(ctx, frames=31, hand=False, camera_visibility=0.0, overlay_visible=False, capture_every=30)
    assert np.abs(first.astype(int) - last.astype(int)).mean() > 1.0
