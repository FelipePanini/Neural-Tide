"""MP4 recorder and the GIF helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from config import CONFIG
from render.recorder import Recorder
from tools.gif import write_gif
from tools.make_gif import read_frames

moderngl = pytest.importorskip("moderngl")


@pytest.fixture(scope="module")
def ctx():  # type: ignore[no-untyped-def]
    try:
        context = moderngl.create_standalone_context(require=330)
    except Exception as exc:
        pytest.skip(f"OpenGL 3.3 unavailable: {exc}")
    yield context
    context.release()


def _record(ctx, tmp_path: Path, render_hz: float, seconds: float = 1.0) -> Path:  # type: ignore[no-untyped-def]
    size = (65, 49)  # odd on purpose: the recorder crops to even sizes
    fbo = ctx.simple_framebuffer(size)
    recorder = Recorder(CONFIG.recording, tmp_path)
    recorder.start(size, now=0.0)
    for i in range(round(seconds * render_hz)):
        fbo.clear(i / (seconds * render_hz), 0.3, 0.6, 1.0)
        recorder.capture(fbo, size, now=i / render_hz)
    path = recorder.stop()
    fbo.release()
    assert path is not None and not recorder.active
    return path


def _video_info(path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return count, *size


def test_recorder_writes_a_playable_mp4_at_its_own_rate(ctx, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    count, width, height = _video_info(_record(ctx, tmp_path, render_hz=60.0))
    assert (width, height) == (64, 48)
    assert abs(count - CONFIG.recording.fps) <= 2  # 60 render frames → ~30 video frames


def test_recorder_repeats_frames_to_keep_real_time(ctx, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    count, _, _ = _video_info(_record(ctx, tmp_path, render_hz=12.0))
    assert abs(count - CONFIG.recording.fps) <= 3  # a slow render still gives a 1-second video


def test_write_gif_resizes_and_loops(tmp_path: Path) -> None:
    frames = [np.full((90, 160, 3), (i * 20, 40, 200), dtype=np.uint8) for i in range(8)]
    size = write_gif(frames, tmp_path / "out.gif", fps=10, width=80)
    with Image.open(tmp_path / "out.gif") as gif:
        assert size > 0 and gif.n_frames == 8 and gif.size == (80, 45)
        assert gif.info.get("loop") == 0


def test_make_gif_reads_the_requested_section(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(60):  # 2 seconds getting brighter
        writer.write(np.full((48, 64, 3), i * 4, dtype=np.uint8))
    writer.release()
    frames = list(read_frames(path, start=0.5, duration=1.0, fps=10))
    assert 9 <= len(frames) <= 11
    brightness = [f.mean() for f in frames]
    assert brightness == sorted(brightness) and brightness[0] > 40
