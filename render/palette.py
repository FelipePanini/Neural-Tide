"""Colour palette shared by the effects, eased whenever it changes."""

from __future__ import annotations

import numpy as np

from config import RGB, Palette, PaletteConfig
from core.easing import damp

_FIELDS = ("primary", "secondary", "highlight", "glow")


def _as_array(palette: Palette) -> np.ndarray:
    return np.array([getattr(palette, name) for name in _FIELDS], dtype=np.float64)


class PaletteState:
    """The selected palette plus the colours currently on screen (easing towards it)."""

    def __init__(self, cfg: PaletteConfig) -> None:
        self._cfg = cfg
        self.index = 0
        self._colors = _as_array(cfg.palettes[0])

    @property
    def name(self) -> str:
        return self._cfg.palettes[self.index].name

    def select(self, index: int) -> None:
        self.index = index % len(self._cfg.palettes)

    def cycle(self) -> None:
        self.select(self.index + 1)

    def update(self, dt: float) -> None:
        target = _as_array(self._cfg.palettes[self.index])
        self._colors = damp(self._colors, target, self._cfg.transition_tau_s, dt)

    def _color(self, name: str) -> RGB:
        r, g, b = self._colors[_FIELDS.index(name)]
        return (float(r), float(g), float(b))

    @property
    def primary(self) -> RGB:
        return self._color("primary")

    @property
    def secondary(self) -> RGB:
        return self._color("secondary")

    @property
    def highlight(self) -> RGB:
        return self._color("highlight")

    @property
    def glow(self) -> RGB:
        return self._color("glow")
