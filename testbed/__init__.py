"""TestBED."""

from importlib.metadata import PackageNotFoundError, version

import torch
from omegaconf import OmegaConf

from . import (
    core,
    design_nets,
    design_objectives,
    designers,
    envs,
    eval,
    inference,
    methods,
    tasks,
)

# Running from a source tree that was never installed has no metadata to read.
try:
    __version__ = version("testbed")
except PackageNotFoundError:
    __version__ = "unknown"
del PackageNotFoundError, version

# Lets recipes write `device: ${auto_device:}`.
OmegaConf.register_new_resolver(
    "auto_device", lambda: "cuda" if torch.cuda.is_available() else "cpu", replace=True
)
