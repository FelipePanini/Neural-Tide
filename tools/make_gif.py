"""Turn a recording (R key → recordings/*.mp4) into a GIF for the README.

Usage::

    python tools/make_gif.py recordings/neural-tide-20261101-120000.mp4 assets/modo-1.gif
    python tools/make_gif.py video.mp4 out.gif --start 2.5 --duration 4 --width 640 --fps 15
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tools.gif import write_gif  # noqa: E402


def read_frames(path: Path, start: float, duration: float, fps: float) -> Iterator[np.ndarray]:
    """RGB frames from ``path`` between ``start`` and ``start + duration``, resampled to ``fps``."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"não consegui abrir {path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = source_fps / fps
    first, last = int(start * source_fps), int((start + duration) * source_fps)
    wanted = first
    index = 0
    try:
        while index <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= wanted:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                wanted += step
            index += 1
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("gif", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="seconds to skip")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds to keep")
    parser.add_argument("--width", type=int, default=640, help="output width in pixels")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--colors", type=int, default=128)
    args = parser.parse_args()
    try:
        size = write_gif(read_frames(args.video, args.start, args.duration, args.fps), args.gif, args.fps, args.width, args.colors)
    except (OSError, ValueError) as exc:
        print(f"[erro] {exc}", file=sys.stderr)
        return 1
    print(f"[ok] {args.gif}  {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
