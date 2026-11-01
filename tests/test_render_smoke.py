"""Offscreen render of a full frame: shaders compile and the overlay draws."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

moderngl = pytest.importorskip("moderngl")

from config import CONFIG  # noqa: E402
from core.hand import SourceStatus, TrackingFrame  # noqa: E402
from core.hand_state import HandStateSmoother  # noqa: E402
from core.input_mock import FINGER_PRESETS, synthetic_hand  # noqa: E402
from core.tracker import HandFilterBank  # noqa: E402
from render.overlay import DebugInfo, DebugOverlay  # noqa: E402
from render.renderer import Renderer  # noqa: E402

SIZE = (480, 270)


@pytest.fixture(scope="module")
def ctx():  # type: ignore[no-untyped-def]
    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as exc:  # no GPU / driver in this environment
        pytest.skip(f"OpenGL 3.3 unavailable: {exc}")
    yield context
    context.release()


def render_frame(ctx, camera_visibility: float = 1.0) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Render one mock frame offscreen and return it as an (H, W, 3) array."""
    fbo = ctx.simple_framebuffer(SIZE)
    renderer = Renderer(ctx, SIZE, CONFIG, target=fbo)
    overlay = DebugOverlay(renderer, CONFIG.debug)
    gradient = np.linspace(0, 255, 64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3)
    renderer.set_camera_image(gradient)

    bank = HandFilterBank(CONFIG.tracker, CONFIG.geometry)
    smoother = HandStateSmoother(CONFIG.hand_state)
    hands = []
    for i in range(30):  # let presence fade in
        obs = [synthetic_hand((0.35, 0.55), FINGER_PRESETS[3], image_aspect=renderer.viewport.image_aspect)]
        frame = TrackingFrame(i + 1, i / 30.0, bank.update(obs, i / 30.0))
        hands = smoother.update(frame, renderer.viewport, 1.0 / 30.0)

    renderer.begin_frame()
    renderer.draw_background(camera_visibility)
    info = DebugInfo(60.0, SourceStatus("mock (mouse)", 60.0), "—", "—", "câmera")
    overlay.draw(hands, info, 1.0 / 30.0)
    pixels = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(SIZE[1], SIZE[0], 3)
    renderer.release()
    fbo.release()
    return pixels[::-1]  # OpenGL rows start at the bottom


def test_frame_renders_background_hand_and_panel(ctx) -> None:  # type: ignore[no-untyped-def]
    pixels = render_frame(ctx)
    assert pixels.mean() > 1.0  # dimmed camera background is there
    hand_region = pixels[60:220, 100:240]
    assert hand_region.max() > 200  # bright landmarks / bones
    panel_region = pixels[10:120, 10:200]
    assert panel_region.max() > 150  # HUD text


def test_black_background_when_camera_hidden(ctx) -> None:  # type: ignore[no-untyped-def]
    pixels = render_frame(ctx, camera_visibility=0.0)
    assert pixels[-20:, -60:].max() == 0  # bottom-right corner has neither camera nor overlay
