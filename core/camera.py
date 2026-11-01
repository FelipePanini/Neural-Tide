"""Webcam capture in a background thread that always exposes the newest frame."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from config import CameraConfig
from core.stats import RateMeter

_BACKENDS = {
    "dshow": "CAP_DSHOW",
    "msmf": "CAP_MSMF",
    "avfoundation": "CAP_AVFOUNDATION",
    "v4l2": "CAP_V4L2",
    "any": "CAP_ANY",
}


def _unavailable_message(index: int) -> str:
    if sys.platform == "darwin":
        return (
            f"câmera {index} indisponível: libere o app do terminal em Ajustes do Sistema > "
            "Privacidade e Segurança > Câmera e abra o terminal de novo"
        )
    return f"câmera {index} indisponível (tentando de novo)"


def _resolve_backend(name: str) -> int:
    """OpenCV capture API for ``name``; "auto" picks the fast one per platform."""
    if name == "auto":
        if sys.platform == "win32":
            name = "dshow"  # opens much faster than MSMF on most notebooks
        elif sys.platform == "darwin":
            name = "avfoundation"
        elif sys.platform.startswith("linux"):
            name = "v4l2"
        else:
            name = "any"
    return int(getattr(cv2, _BACKENDS[name]))


@dataclass(frozen=True)
class CameraFrame:
    seq: int
    timestamp: float  # time.perf_counter() when the frame was read
    image: np.ndarray  # (H, W, 3) RGB uint8, already mirrored when configured


class Camera:
    """Reads the webcam continuously; consumers only ever see the latest frame.

    If the camera is missing or stops delivering frames, the thread keeps
    retrying (with back-off) and reports the problem through ``error`` instead
    of crashing.
    """

    def __init__(self, cfg: CameraConfig, rate_window_s: float = 1.0) -> None:
        self._cfg = cfg
        self._cond = threading.Condition()
        self._frame: CameraFrame | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._meter = RateMeter(rate_window_s)
        self._first_capture: cv2.VideoCapture | None = None
        self.error: str | None = None
        self.resolution: tuple[int, int] | None = None

    @property
    def fps(self) -> float:
        return self._meter.rate

    def start(self) -> None:
        """Start capturing. On macOS this blocks until the first open finishes.

        macOS only shows the camera permission prompt when the capture is
        opened on the main thread (OpenCV cannot spin the run loop from another
        thread), so the first open happens here and is handed to the worker.
        """
        if _resolve_backend(self._cfg.backend) == cv2.CAP_AVFOUNDATION:
            self._first_capture = self._open()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)

    def latest(self) -> CameraFrame | None:
        return self._frame

    def wait_for_frame(self, after_seq: int, timeout: float) -> CameraFrame | None:
        """Block until a frame newer than ``after_seq`` exists (or timeout/stop)."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._stop.is_set() or (self._frame is not None and self._frame.seq > after_seq),
                timeout,
            )
            frame = self._frame
        return frame if frame is not None and frame.seq > after_seq else None

    def _open(self) -> cv2.VideoCapture | None:
        cfg = self._cfg
        cap = cv2.VideoCapture(cfg.index, _resolve_backend(cfg.backend))
        if not cap.isOpened():
            cap.release()
            return None
        # FOURCC first: some drivers only offer high resolutions in MJPG.
        if cfg.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)
        return cap

    def _run(self) -> None:
        cfg = self._cfg
        seq = 0
        retry_delay = cfg.retry_delay_s
        while not self._stop.is_set():
            cap = self._first_capture if self._first_capture is not None else self._open()
            self._first_capture = None
            if cap is None:
                self.error = _unavailable_message(cfg.index)
                self._stop.wait(retry_delay)
                retry_delay = min(retry_delay * 2.0, cfg.retry_max_delay_s)
                continue
            self.error = None
            retry_delay = cfg.retry_delay_s
            failures = 0
            try:
                while not self._stop.is_set():
                    ok, bgr = cap.read()
                    if not ok or bgr is None:
                        failures += 1
                        if failures >= cfg.read_failures_before_reopen:
                            self.error = "a câmera parou de enviar imagens (reabrindo)"
                            break
                        self._stop.wait(cfg.read_failure_sleep_s)
                        continue
                    failures = 0
                    if cfg.mirror:
                        bgr = cv2.flip(bgr, 1)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    seq += 1
                    frame = CameraFrame(seq, time.perf_counter(), rgb)
                    with self._cond:
                        self._frame = frame
                        self._cond.notify_all()
                    self._meter.tick(frame.timestamp)
                    self.resolution = (rgb.shape[1], rgb.shape[0])
            finally:
                cap.release()
