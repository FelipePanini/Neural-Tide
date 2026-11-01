"""Person segmentation (MediaPipe selfie segmenter) in a background thread."""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

from config import ModelFile, SegmentationConfig
from core.camera import Camera


class PersonSegmenter:
    """Segments the newest camera frame while enabled; idle (no CPU) otherwise.

    The thread starts on the first ``set_enabled(True)``, so the model is only
    loaded if a mode actually needs it.
    """

    def __init__(self, camera: Camera, cfg: SegmentationConfig, model: ModelFile) -> None:
        self._camera = camera
        self._cfg = cfg
        self._model = model
        self._enabled = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mask: np.ndarray | None = None
        self.error: str | None = None

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled.set()
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="segmenter", daemon=True)
                self._thread.start()
        else:
            self._enabled.clear()
            self._mask = None

    def latest(self) -> np.ndarray | None:
        return self._mask if self._enabled.is_set() else None

    def stop(self) -> None:
        self._stop.set()
        self._enabled.set()  # wake the thread so it can exit
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _create(self) -> Any:
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions, RunningMode

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(self._model.path)),
            running_mode=RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        return ImageSegmenter.create_from_options(options)

    def _run(self) -> None:
        import mediapipe as mp

        if not self._model.path.exists():
            self.error = "modelo de segmentação ausente: rode python tools/download_model.py"
            return
        try:
            segmenter = self._create()
        except Exception as exc:  # shown in the overlay instead of crashing
            self.error = f"falha ao iniciar a segmentação: {exc}"
            return

        seq, last_ms = 0, -1
        epoch = time.perf_counter()
        with segmenter:
            while not self._stop.is_set():
                if not self._enabled.wait(self._cfg.idle_wait_s):
                    continue
                frame = self._camera.wait_for_frame(seq, self._cfg.idle_wait_s)
                if frame is None or not self._enabled.is_set():
                    continue
                seq = frame.seq
                h, w = frame.image.shape[:2]
                size = (self._cfg.width, max(round(self._cfg.width * h / w), 1))
                small = np.ascontiguousarray(cv2.resize(frame.image, size, interpolation=cv2.INTER_AREA))
                timestamp_ms = max(int((frame.timestamp - epoch) * 1000.0), last_ms + 1)
                last_ms = timestamp_ms
                result = segmenter.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=small), timestamp_ms)
                if result.confidence_masks:
                    self._mask = np.array(result.confidence_masks[0].numpy_view()[..., 0], dtype=np.float32)
