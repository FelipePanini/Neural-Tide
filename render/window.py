"""pygame window with an OpenGL 3.3 core context for ModernGL."""

from __future__ import annotations

import sys

import moderngl
import pygame

from config import WindowConfig


def create_window(cfg: WindowConfig, fullscreen: bool | None = None) -> tuple[moderngl.Context, tuple[int, int]]:
    """Open the window and return the ModernGL context plus the drawable size."""
    pygame.init()
    major, minor = cfg.gl_version
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, major)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, minor)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    if sys.platform == "darwin":
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

    fullscreen = cfg.fullscreen if fullscreen is None else fullscreen
    flags = pygame.OPENGL | pygame.DOUBLEBUF
    if fullscreen:
        flags |= pygame.FULLSCREEN
    elif cfg.resizable:
        flags |= pygame.RESIZABLE
    size = (0, 0) if fullscreen else (cfg.width, cfg.height)
    try:
        pygame.display.set_mode(size, flags, vsync=int(cfg.vsync))
    except pygame.error:  # some drivers refuse vsync requests; run without it
        pygame.display.set_mode(size, flags)
    pygame.display.set_caption(cfg.title)
    ctx = moderngl.create_context(require=major * 100 + minor * 10)
    return ctx, pygame.display.get_window_size()
