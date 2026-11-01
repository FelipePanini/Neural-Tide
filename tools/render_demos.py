"""Render a demo GIF of every mode offscreen, with scripted synthetic hands.

No webcam needed: the hands follow fixed paths, so the GIFs are reproducible.
Real recordings (R key + tools/make_gif.py) can replace them any time.

Usage::

    python tools/render_demos.py                 # assets/modo-1.gif … modo-8.gif + showreel.gif
    python tools/render_demos.py --modes 1 4     # only some modes
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import moderngl  # noqa: E402
import numpy as np  # noqa: E402

from config import CONFIG  # noqa: E402
from core.gestures import GestureEvent, GestureKind  # noqa: E402
from core.hand import LEFT, RIGHT, HandObservation  # noqa: E402
from core.input_mock import FINGER_PRESETS, synthetic_hand  # noqa: E402
from render.offscreen import OffscreenStage  # noqa: E402
from tools.gif import write_gif  # noqa: E402

RENDER_SIZE = (960, 540)
GIF_WIDTH = 640
SIM_FPS = 60
GIF_FPS = 15
WARMUP_S = 2.0
CLIP_S = 4.0
ASPECT = RENDER_SIZE[0] / RENDER_SIZE[1]
OPEN, POINT = FINGER_PRESETS[5], FINGER_PRESETS[1]


@dataclass(frozen=True)
class Hand:
    center: tuple[float, float]  # image space
    curls: tuple[float, ...] = OPEN
    pinch: float = 0.0
    handedness: str = RIGHT
    roll: float = 0.0  # degrees


Choreography = Callable[[float], list[Hand]]


def ramp(t: float, start: float, length: float = 0.25) -> float:
    """0 → 1 → 0 bump: a pinch held for ``length`` seconds starting at ``start``."""
    if t < start or t > start + length + 0.3:
        return 0.0
    rise = min((t - start) / 0.12, 1.0)
    fall = min(max((start + length + 0.3 - t) / 0.2, 0.0), 1.0)
    return min(rise, fall)


def tide(t: float) -> list[Hand]:
    return [Hand((0.5 + 0.22 * math.sin(t * 1.1), 0.55 + 0.15 * math.sin(t * 2.2)), pinch=ramp(t, 4.6, 0.8))]


def neural(t: float) -> list[Hand]:
    return [Hand((0.42 + 0.12 * math.sin(t * 0.7), 0.55 + 0.08 * math.cos(t * 0.9)), pinch=max(ramp(t, 2.6), ramp(t, 4.4)))]


def grid(t: float) -> list[Hand]:
    return [Hand((0.25 + 0.5 * (0.5 - 0.5 * math.cos(t * 0.9)), 0.66 + 0.05 * math.sin(t * 1.7)), pinch=ramp(t, 4.8))]


def core(t: float) -> list[Hand]:
    spread = 0.17 + 0.07 * math.sin(t * 1.2)
    tilt = 0.04 * math.sin(t * 0.8)
    roll = 25.0 * math.sin(t * 0.9)
    return [
        Hand((0.5 - spread, 0.55 + tilt), roll=roll),
        Hand((0.5 + spread, 0.55 - tilt), handedness=LEFT, roll=roll),
    ]


def ribbons(t: float) -> list[Hand]:
    return [Hand((0.5 + 0.2 * math.cos(t * 2.0), 0.55 + 0.17 * math.sin(t * 3.0)))]


def attention(t: float) -> list[Hand]:
    return [Hand((0.3 + 0.4 * (0.5 - 0.5 * math.cos(t * 0.8)), 0.5 + 0.12 * math.sin(t * 1.1)), POINT, pinch=ramp(t, 4.0))]


def body(t: float) -> list[Hand]:
    return [
        Hand((0.3 + 0.05 * math.sin(t * 1.3), 0.45 + 0.1 * math.sin(t * 1.7))),
        Hand((0.7 + 0.05 * math.sin(t * 1.1), 0.42 + 0.1 * math.cos(t * 1.5)), handedness=LEFT),
    ]


def rain(t: float) -> list[Hand]:
    return [Hand((0.5 + 0.18 * math.sin(t * 0.6), 0.6 + 0.05 * math.sin(t * 1.4)), pinch=ramp(t, 4.5))]


CHOREOGRAPHIES: dict[int, Choreography] = {1: tide, 2: neural, 3: grid, 4: core, 5: ribbons, 6: attention, 7: body, 8: rain}
# Events that the gesture engine would send: (mode, time, kind) — pinches and one fist.
EVENTS: dict[int, list[tuple[float, GestureKind]]] = {
    2: [(2.6, GestureKind.PINCH_START), (4.4, GestureKind.PINCH_START)],
    3: [(4.8, GestureKind.PINCH_START)],
    4: [(3.6, GestureKind.PINCH_START), (5.0, GestureKind.EXPLODE)],
    6: [(4.0, GestureKind.PINCH_START)],
    7: [(4.6, GestureKind.PINCH_START)],
    8: [(4.5, GestureKind.PINCH_START)],
}


def observations(hands: list[Hand]) -> list[HandObservation]:
    return [
        synthetic_hand(h.center, h.curls, pinch=h.pinch, handedness=h.handedness, image_aspect=ASPECT, roll_deg=h.roll)
        for h in hands
    ]


def render_mode(ctx: moderngl.Context, mode: int) -> list[np.ndarray]:
    stage = OffscreenStage(ctx, RENDER_SIZE, CONFIG, mode=mode - 1)
    choreography, events = CHOREOGRAPHIES[mode], list(EVENTS.get(mode, []))
    frames, dt = [], 1.0 / SIM_FPS
    total = round((WARMUP_S + CLIP_S) * SIM_FPS)
    every = SIM_FPS // GIF_FPS
    for i in range(total):
        t = i * dt
        hands = choreography(t)
        due = [e for e in events if e[0] <= t]
        events = [e for e in events if e[0] > t]
        obs = observations(hands)
        fired = [GestureEvent(kind, 0, stage.hands[0].pinch_point if stage.hands else np.zeros(2)) for _, kind in due]
        stage.step(obs, dt, fired)
        if t >= WARMUP_S and i % every == 0:
            frames.append(stage.read())
    stage.release()
    return frames


def render_showreel(ctx: moderngl.Context, seconds_per_mode: float = 1.8) -> list[np.ndarray]:
    """All modes in a row, switching with the ripple and the toast, like a swipe."""
    stage = OffscreenStage(ctx, RENDER_SIZE, CONFIG, mode=0)
    frames, dt = [], 1.0 / SIM_FPS
    every = SIM_FPS // 12
    total = round((1.0 + seconds_per_mode * 8) * SIM_FPS)
    for i in range(total):
        t = i * dt
        mode = min(int(max(t - 1.0, 0.0) / seconds_per_mode), 7)
        if mode != stage.modes.active_index:
            hands = stage.hands
            stage.select_mode(mode, hands[0].palm if hands else None)
        stage.step(observations(CHOREOGRAPHIES[mode + 1](t + 3.0)), dt)
        if t >= 1.0 and i % every == 0:
            frames.append(stage.read())
    stage.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", type=int, nargs="*", default=list(CHOREOGRAPHIES), help="mode numbers (1–8)")
    parser.add_argument("--no-showreel", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "assets")
    args = parser.parse_args()
    ctx = moderngl.create_standalone_context(require=330)
    for mode in args.modes:
        size = write_gif(render_mode(ctx, mode), args.out / f"modo-{mode}.gif", GIF_FPS, GIF_WIDTH)
        print(f"[ok] modo-{mode}.gif  {size / 1e6:.1f} MB")
    if not args.no_showreel:
        size = write_gif(render_showreel(ctx), args.out / "showreel.gif", 12, 480)
        print(f"[ok] showreel.gif  {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
