"""TestBED."""

import importlib

from . import (
    core,
    design_nets,
    design_objectives,
    designers,
    envs,
    inference,
    metrics,
    tasks,
)

# Training methods and checkpoint evaluation need the optional ``[train]``
# dependencies (Hydra/OmegaConf), so they load only when first touched.
_LAZY_SUBMODULES = ("cli", "eval", "methods")


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
