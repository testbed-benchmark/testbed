"""Standalone evaluation utilities for trained BED policy checkpoints.

``chunked_spce_snmc`` and ``eval_spce_snmc`` work on any task and rollout. The
checkpoint evaluators rebuild runs from their saved Hydra configs, so they need
the optional ``[train]`` dependencies and load only when first touched.
"""

import importlib

from .base import chunked_spce_snmc, eval_spce_snmc

_LAZY = {
    "build_design_net": "eval_policy",
    "evaluate_checkpoint": "eval_policy",
    "build_losses": "eval_loss",
    "eval_losses": "eval_loss",
    "evaluate_loss_checkpoint": "eval_loss",
    "evaluate_static_checkpoint": "eval_static",
}

__all__ = [
    "build_design_net",
    "build_losses",
    "chunked_spce_snmc",
    "eval_losses",
    "eval_spce_snmc",
    "evaluate_checkpoint",
    "evaluate_loss_checkpoint",
    "evaluate_static_checkpoint",
]


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
