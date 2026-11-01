"""Offscreen render of full frames: shaders compile, effects and overlay draw."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

moderngl = pytest.importorskip("moderngl")

from config import CONFIG  # noqa: E402
from core.input_mock import FINGER_PRESETS, synthetic_hand  # noqa: E402
from effects import EFFECTS  # noqa: E402
from render.offscreen import OffscreenStage  # noqa: E402

SIZE = (480, 270)
FPS = 60.0


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
    mode: int = 0,
    curls: tuple[float, ...] = FINGER_PRESETS[5],
    focus: bool = False,
    capture_every: int = 0,
) -> list[np.ndarray]:
    """Simulate ``frames`` frames offscreen; returns the last frame (plus every ``capture_every``-th)."""
    stage = OffscreenStage(ctx, size, CONFIG, mode=mode, focus=focus, overlay=overlay_visible, effects=EFFECTS if effects else [])
    stage.set_camera_image(np.linspace(0, 255, 64 * 48 * 3, dtype=np.uint8).reshape(48, 64, 3), camera_visibility)
    aspect = stage.renderer.viewport.image_aspect
    captured = []
    for i in range(frames):
        obs = [synthetic_hand(hand_path(stage.time), curls, pinch=pinch, image_aspect=aspect)] if hand else []
        stage.step(obs, 1.0 / FPS)
        if i == frames - 1 or (capture_every and i % capture_every == 0):
            captured.append(stage.read())
    stage.release()
    return captured


def test_frame_renders_background_hand_and_panel(ctx) -> None:  # type: ignore[no-untyped-def]
    (pixels,) = render_frames(ctx, frames=30, effects=False)
    assert pixels.mean() > 1.0  # dimmed camera background is there
    assert pixels[60:220, 100:240].max() > 200  # bright landmarks / bones
    assert pixels[10:120, 10:200].max() > 150  # HUD text


def test_black_background_when_camera_hidden(ctx) -> None:  # type: ignore[no-untyped-def]
    (pixels,) = render_frames(ctx, frames=30, effects=False, camera_visibility=0.0)
    assert pixels[-20:, -60:].max() == 0  # bottom-right corner has neither camera nor overlay


@pytest.mark.parametrize("mode", range(len(EFFECTS)))
def test_every_mode_lights_up_the_screen(ctx, mode: int) -> None:  # type: ignore[no-untyped-def]
    (pixels,) = render_frames(ctx, camera_visibility=0.0, overlay_visible=False, mode=mode, focus=True)
    lit = (pixels.max(axis=2) > 10).mean()
    assert 0.02 < lit < 0.8  # as in the app (hand spotlight on): part of the screen, not all of it
    assert pixels.max() > 100  # bright highlights / bloom


def test_tide_keeps_moving_without_hands(ctx) -> None:  # type: ignore[no-untyped-def]
    first, *_, last = render_frames(ctx, frames=31, hand=False, camera_visibility=0.0, overlay_visible=False, capture_every=30)
    assert np.abs(first.astype(int) - last.astype(int)).mean() > 1.0


def test_focus_keeps_the_effect_around_the_hand(ctx) -> None:  # type: ignore[no-untyped-def]
    full, *_ = render_frames(ctx, camera_visibility=0.0, overlay_visible=False, focus=False)
    focused, *_ = render_frames(ctx, camera_visibility=0.0, overlay_visible=False, focus=True)
    hand = (slice(90, 200), slice(110, 230))  # around the circling hand (left half)
    far = (slice(20, 120), slice(380, 470))  # top-right corner
    assert focused[far].mean() < 0.25 * full[far].mean()
    assert focused[hand].mean() > 0.5 * full[hand].mean()
