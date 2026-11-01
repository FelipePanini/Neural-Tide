"""Lightweight runtime statistics (frame rates) shared across threads."""

from __future__ import annotations

import time
from collections import deque


class RateMeter:
    """Counts events and reports their rate over a sliding time window.

    Safe for one writer thread and any number of readers: readers only touch
    the deque ends and its length.
    """

    def __init__(self, window_s: float = 1.0) -> None:
        self._window = window_s
        self._stamps: deque[float] = deque()

    def tick(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self._stamps.append(now)
        while self._stamps and now - self._stamps[0] > self._window:
            self._stamps.popleft()

    @property
    def rate(self) -> float:
        try:
            first, last = self._stamps[0], self._stamps[-1]
        except IndexError:
            return 0.0
        count = len(self._stamps)
        if count < 2 or last <= first or time.perf_counter() - last > self._window:
            return 0.0
        return (count - 1) / (last - first)
