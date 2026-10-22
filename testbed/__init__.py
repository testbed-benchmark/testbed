"""TestBED."""

from importlib.metadata import version

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

__version__ = version("testbed")
del version

# Lets recipes write `device: ${auto_device:}`.
OmegaConf.register_new_resolver(
    "auto_device", lambda: "cuda" if torch.cuda.is_available() else "cpu", replace=True
)
