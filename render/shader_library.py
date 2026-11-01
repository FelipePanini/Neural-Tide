"""Loads GLSL programs from ``render/shaders`` with a tiny ``#include`` preprocessor."""

from __future__ import annotations

import re
from pathlib import Path

import moderngl

from config import SHADERS_DIR

_INCLUDE = re.compile(r'^[ \t]*#include[ \t]+"([^"]+)"[ \t]*$', re.MULTILINE)


class ShaderLibrary:
    """Compiles and caches ``<name>.vert.glsl`` + ``<name>.frag.glsl`` programs."""

    def __init__(self, ctx: moderngl.Context, directory: Path = SHADERS_DIR) -> None:
        self._ctx = ctx
        self._dir = directory
        self._programs: dict[tuple[str, str], moderngl.Program] = {}

    def program(self, fragment: str, vertex: str | None = None) -> moderngl.Program:
        """Program built from ``fragment``'s shader pair; ``vertex`` overrides the vertex stage."""
        key = (vertex or fragment, fragment)
        if key not in self._programs:
            self._programs[key] = self._ctx.program(
                vertex_shader=self.source(f"{key[0]}.vert.glsl"),
                fragment_shader=self.source(f"{fragment}.frag.glsl"),
            )
        return self._programs[key]

    def source(self, filename: str, _stack: tuple[str, ...] = ()) -> str:
        """Shader source with every ``#include "file"`` expanded recursively."""
        if filename in _stack:
            raise ValueError(f"circular #include: {' -> '.join((*_stack, filename))}")
        text = (self._dir / filename).read_text(encoding="utf-8")
        return _INCLUDE.sub(lambda m: self.source(m.group(1), (*_stack, filename)), text)

    def release(self) -> None:
        for program in self._programs.values():
            program.release()
        self._programs.clear()
