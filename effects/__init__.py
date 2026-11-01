"""Effect registry: a new mode is one file in this package plus one entry here."""

from effects.base import Effect, EffectEnv

EFFECTS: list[type[Effect]] = []

__all__ = ["EFFECTS", "Effect", "EffectEnv"]
