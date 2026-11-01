"""Hand tracking: MediaPipe HandLandmarker in its own thread + One Euro smoothing.

Pipeline: ``Camera`` thread → ``HandTracker`` thread (detect, assign slots,
filter) → the render loop reads ``latest()`` without ever waiting.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import Config, HandGeometryConfig, TrackerConfig
from core.camera import Camera, CameraFrame
from core.filters import OneEuroFilter
from core.hand import (
    EMPTY_FRAME,
    RIGHT,
    HandObservation,
    SourceStatus,
    TrackedHand,
    TrackingFrame,
    fingers_extended,
    other_hand,
    palm_center,
    pinch_strength,
)
from core.stats import RateMeter


@dataclass
class _Slot:
    slot_id: int
    image_filter: OneEuroFilter
    world_filter: OneEuroFilter
    right_score: float  # moving average of "this is a right hand" votes
    last_seen: float
    raw: np.ndarray
    landmarks: np.ndarray
    world: np.ndarray
    fingers: tuple[bool, ...] = (False,) * 5
    pinch: float = 0.0

    @property
    def handedness(self) -> str:
        return RIGHT if self.right_score >= 0.5 else other_hand(RIGHT)


class HandFilterBank:
    """Gives detections stable slots and smooths each slot with One Euro filters.

    MediaPipe returns hands in arbitrary order, so every detection is matched to
    the nearest existing slot (palm centre distance, with a small penalty when
    the handedness label disagrees). A slot that misses a few detections is
    kept as "stale" for ``lost_timeout_s`` so a one-frame dropout does not make
    the hand blink out.
    """

    def __init__(self, tracker: TrackerConfig, geometry: HandGeometryConfig) -> None:
        self._cfg = tracker
        self._geometry = geometry
        self._slots: dict[int, _Slot] = {}
        self._ids = itertools.count()

    def reset(self) -> None:
        self._slots.clear()

    def update(self, observations: Sequence[HandObservation], t: float) -> tuple[TrackedHand, ...]:
        """Feed one detection result taken at time ``t`` and return the tracked hands."""
        self._slots = {sid: s for sid, s in self._slots.items() if t - s.last_seen <= self._cfg.lost_timeout_s}
        for obs, slot in self._assign(observations):
            if slot is None:
                slot = self._new_slot(obs, t)
            self._apply(slot, obs, t)
        return tuple(
            TrackedHand(
                slot=s.slot_id,
                handedness=s.handedness,
                landmarks=s.landmarks,
                raw_landmarks=s.raw,
                world=s.world,
                fingers=s.fingers,
                pinch=s.pinch,
                stale=s.last_seen != t,
            )
            for s in sorted(self._slots.values(), key=lambda s: s.slot_id)
        )

    def _assign(self, observations: Sequence[HandObservation]) -> list[tuple[HandObservation, _Slot | None]]:
        candidates: list[tuple[float, int, int]] = []
        centers = [palm_center(obs.landmarks[:, :2]) for obs in observations]
        for oi, obs in enumerate(observations):
            for slot in self._slots.values():
                dist = float(np.linalg.norm(centers[oi] - palm_center(slot.landmarks[:, :2])))
                if dist > self._cfg.match_max_distance:
                    continue
                penalty = self._cfg.handedness_penalty if slot.handedness != obs.handedness else 0.0
                candidates.append((dist + penalty, oi, slot.slot_id))
        candidates.sort()

        matched: dict[int, _Slot] = {}
        used_slots: set[int] = set()
        for _, oi, sid in candidates:
            if oi in matched or sid in used_slots:
                continue
            matched[oi] = self._slots[sid]
            used_slots.add(sid)
        return [(obs, matched.get(oi)) for oi, obs in enumerate(observations)]

    def _new_slot(self, obs: HandObservation, t: float) -> _Slot:
        slot = _Slot(
            slot_id=next(self._ids),
            image_filter=OneEuroFilter.from_config(self._cfg.image_filter),
            world_filter=OneEuroFilter.from_config(self._cfg.world_filter),
            right_score=self._vote(obs),
            last_seen=t,
            raw=obs.landmarks,
            landmarks=obs.landmarks,
            world=obs.world,
        )
        self._slots[slot.slot_id] = slot
        return slot

    @staticmethod
    def _vote(obs: HandObservation) -> float:
        return obs.score if obs.handedness == RIGHT else 1.0 - obs.score

    def _apply(self, slot: _Slot, obs: HandObservation, t: float) -> None:
        slot.raw = obs.landmarks
        slot.landmarks = slot.image_filter(obs.landmarks, t)
        slot.world = slot.world_filter(obs.world, t)
        slot.right_score += self._cfg.handedness_ema * (self._vote(obs) - slot.right_score)
        slot.last_seen = t
        slot.fingers = fingers_extended(slot.world, self._geometry)
        slot.pinch = pinch_strength(slot.world, self._geometry)


def needs_label_swap(image_mirrored: bool, model_expects_mirrored: bool) -> bool:
    """Whether MediaPipe's handedness labels must be swapped to name the real hand."""
    return image_mirrored != model_expects_mirrored


def observations_from_result(result: Any, swap_labels: bool) -> list[HandObservation]:
    """Convert a ``HandLandmarkerResult`` into observations.

    ``swap_labels`` comes from :func:`needs_label_swap`, so the handedness of
    every observation always names the user's real hand.
    """
    observations = []
    for image_lms, world_lms, categories in zip(
        result.hand_landmarks, result.hand_world_landmarks, result.handedness, strict=False
    ):
        best = categories[0]
        label = other_hand(best.category_name) if swap_labels else best.category_name
        observations.append(
            HandObservation(
                landmarks=np.array([(p.x, p.y, p.z) for p in image_lms], dtype=np.float64),
                world=np.array([(p.x, p.y, p.z) for p in world_lms], dtype=np.float64),
                handedness=label,
                score=float(best.score),
            )
        )
    return observations


class HandTracker:
    """Webcam hand source: owns the camera thread and the detection thread."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._camera = Camera(cfg.camera, cfg.debug.rate_window_s)
        self._bank = HandFilterBank(cfg.tracker, cfg.geometry)
        self._frame = EMPTY_FRAME
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hand-tracker", daemon=True)
        self._meter = RateMeter(cfg.debug.rate_window_s)
        self._inference_ms = 0.0
        self._error: str | None = None

    def start(self) -> None:
        self._camera.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._camera.stop()

    def latest(self) -> TrackingFrame:
        return self._frame

    def camera_frame(self) -> CameraFrame | None:
        return self._camera.latest()

    def status(self) -> SourceStatus:
        return SourceStatus(
            label="webcam",
            tracking_fps=self._meter.rate,
            camera_fps=self._camera.fps,
            inference_ms=self._inference_ms,
            resolution=self._camera.resolution,
            error=self._error or self._camera.error,
        )

    def _create_landmarker(self) -> Any:
        # Imported here so tests and the mock mode never pay mediapipe's import time.
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        cfg = self._cfg.tracker
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._cfg.models.hand_landmarker.path)),
            running_mode=RunningMode.VIDEO,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        return HandLandmarker.create_from_options(options)

    def _run(self) -> None:
        import mediapipe as mp

        try:
            landmarker = self._create_landmarker()
        except Exception as exc:  # surfaced in the overlay instead of killing the app
            self._error = f"falha ao iniciar o MediaPipe: {exc}"
            return

        cfg = self._cfg.tracker
        swap_labels = needs_label_swap(self._cfg.camera.mirror, cfg.model_expects_mirrored)
        seq = 0
        last_ms = -1
        epoch = time.perf_counter()
        with landmarker:
            while not self._stop.is_set():
                frame = self._camera.wait_for_frame(seq, cfg.idle_wait_s)
                if frame is None:
                    # Camera stalled: still age the slots so lost hands fade out.
                    hands = self._bank.update((), time.perf_counter())
                    self._frame = TrackingFrame(self._frame.seq + 1, time.perf_counter(), hands)
                    continue
                seq = frame.seq
                # VIDEO mode needs strictly increasing integer milliseconds.
                timestamp_ms = max(int((frame.timestamp - epoch) * 1000.0), last_ms + 1)
                last_ms = timestamp_ms

                started = time.perf_counter()
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame.image)
                result = landmarker.detect_for_video(image, timestamp_ms)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._inference_ms += cfg.inference_ema * (elapsed_ms - self._inference_ms)

                observations = observations_from_result(result, swap_labels)
                hands = self._bank.update(observations, frame.timestamp)
                self._frame = TrackingFrame(self._frame.seq + 1, frame.timestamp, hands)
                self._meter.tick()
