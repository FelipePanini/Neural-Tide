"""MP4 recording of the rendered frames, encoded in a background thread."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import cv2
import moderngl
import numpy as np

from config import RecordingConfig

_STOP = None


class Recorder:
    """Captures a framebuffer at a fixed rate and hands frames to a writer thread.

    Frames are read back once per video frame (not every render frame) and the
    encoding happens off the render thread, so recording barely touches the
    frame rate. If rendering stutters, the last frame is repeated so the video
    keeps real time; if the writer falls behind, frames are dropped and counted.
    """

    def __init__(self, cfg: RecordingConfig, directory: Path) -> None:
        self._cfg = cfg
        self._dir = directory
        self._queue: queue.Queue[tuple[bytes, tuple[int, int], int] | None] = queue.Queue(cfg.queue_frames)
        self._thread: threading.Thread | None = None
        self._writer: cv2.VideoWriter | None = None
        self._next_frame = 0.0
        self._started = 0.0
        self.path: Path | None = None
        self.dropped = 0
        self.frames = 0

    @property
    def active(self) -> bool:
        return self._thread is not None

    def elapsed(self, now: float | None = None) -> float:
        return ((time.perf_counter() if now is None else now) - self._started) if self.active else 0.0

    def start(self, size: tuple[int, int], now: float | None = None) -> Path:
        """Open a new file for frames of ``size`` (cropped to even dimensions)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        size = (size[0] - size[0] % 2, size[1] - size[1] % 2)  # most codecs need even sizes
        path = self._dir / time.strftime("neural-tide-%Y%m%d-%H%M%S.mp4")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self._cfg.codec), self._cfg.fps, size)
        if not writer.isOpened():
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self._cfg.fallback_codec), self._cfg.fps, size)
        if not writer.isOpened():
            raise RuntimeError("não foi possível abrir o gravador de vídeo")
        self._writer, self.path, self.dropped, self.frames = writer, path, 0, 0
        self._started = self._next_frame = time.perf_counter() if now is None else now
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        return path

    def capture(self, fbo: moderngl.Framebuffer, size: tuple[int, int], now: float | None = None) -> None:
        """Call once per rendered frame, after drawing what should be recorded."""
        if not self.active:
            return
        now = time.perf_counter() if now is None else now
        if now < self._next_frame:
            return
        period = 1.0 / self._cfg.fps
        repeats = min(1 + int((now - self._next_frame) / period), self._cfg.max_repeats)
        self._next_frame = max(self._next_frame + repeats * period, now - period)
        even = (size[0] - size[0] % 2, size[1] - size[1] % 2)
        pixels = fbo.read(viewport=(0, 0, *even), components=3)
        try:
            self._queue.put_nowait((pixels, even, repeats))
        except queue.Full:
            self.dropped += repeats

    def stop(self) -> Path | None:
        """Finish the file and return its path."""
        if self._thread is None:
            return None
        self._queue.put(_STOP)
        self._thread.join()
        self._thread = None
        return self.path

    def _run(self) -> None:
        assert self._writer is not None
        while (item := self._queue.get()) is not _STOP:
            pixels, (w, h), repeats = item
            frame = cv2.cvtColor(np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 3)[::-1], cv2.COLOR_RGB2BGR)
            for _ in range(repeats):
                self._writer.write(frame)
                self.frames += 1
        self._writer.release()
        self._writer = None
