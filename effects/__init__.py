"""Effect registry: a new mode is one file in this package plus one entry here."""

from effects.ai_core import AICore
from effects.attention import Attention
from effects.base import Effect, EffectEnv
from effects.code_rain import CodeRain
from effects.data_tide import DataTide
from effects.holo_grid import HoloGrid
from effects.light_ribbons import LightRibbons
from effects.neural_net import NeuralNet
from effects.particle_body import ParticleBody

# Order = mode number: keys 1–8 and the finger-count gestures select by position.
EFFECTS: list[type[Effect]] = [
    DataTide,
    NeuralNet,
    HoloGrid,
    AICore,
    LightRibbons,
    Attention,
    ParticleBody,
    CodeRain,
]

__all__ = ["EFFECTS", "Effect", "EffectEnv"]
