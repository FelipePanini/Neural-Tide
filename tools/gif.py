"""Small GIF writer used by make_gif.py and render_demos.py."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image


def write_gif(frames: Iterable[np.ndarray], path: Path, fps: float, width: int | None = None, colors: int = 128) -> int:
    """Save RGB frames ``(H, W, 3)`` as a looping GIF; returns the file size in bytes."""
    images = []
    for frame in frames:
        image = Image.fromarray(frame)
        if width and image.width != width:
            image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
        images.append(image.quantize(colors=colors, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE))
    if not images:
        raise ValueError("no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000.0 / fps), loop=0, optimize=True)
    return path.stat().st_size
