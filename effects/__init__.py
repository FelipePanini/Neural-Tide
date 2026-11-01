"""Effect registry: a new mode is one file in this package plus one entry here."""

from effects.ai_core import AICore
from effects.base import Effect, EffectEnv
from effects.data_tide import DataTide
from effects.holo_grid import HoloGrid
from effects.neural_net import NeuralNet

# Order = mode number: keys 1–8 and the finger-count gestures select by position.
EFFECTS: list[type[Effect]] = [
    DataTide,
    NeuralNet,
    HoloGrid,
    AICore,
]

__all__ = ["EFFECTS", "Effect", "EffectEnv"]
