"""Environment wrappers and preprocessing.

This module exposes the wrappers and factory that are actually
implemented in :mod:`wrappers`. Older code referenced symbols
that no longer exist (e.g. ``ActionRepeat``, ``FrameStack``,
``PreprocessObs``); export only the current ones to avoid import
errors when other modules import ``envs``.
"""
from .wrappers import (
    make_env,
    StepAPICompat,
    TimeLimit,
    RewardClip,
    NormalizeObs,
)
from . import preprocess

__all__ = [
    "make_env",
    "StepAPICompat",
    "TimeLimit",
    "RewardClip",
    "NormalizeObs",
    "preprocess",
]