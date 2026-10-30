"""TestBED."""

import importlib
from importlib.metadata import PackageNotFoundError, version

import torch

from . import (
    core,
    design_nets,
    design_objectives,
    designers,
    envs,
    inference,
    tasks,
)

# Running from a source tree that was never installed has no metadata to read.
try:
    __version__ = version("torch-testbed")
except PackageNotFoundError:
    __version__ = "unknown"
del PackageNotFoundError, version

# Training methods and checkpoint evaluation need the optional ``[train]``
# dependencies (Hydra/OmegaConf), so they load only when first touched.
_LAZY_SUBMODULES = ("eval", "methods")

try:
    from omegaconf import OmegaConf
except ImportError:
    pass
else:
    # Lets recipes write `device: ${auto_device:}`.
    OmegaConf.register_new_resolver(
        "auto_device",
        lambda: "cuda" if torch.cuda.is_available() else "cpu",
        replace=True,
    )


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
