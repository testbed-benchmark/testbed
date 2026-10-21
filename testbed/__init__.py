"""TestBED."""

import torch
from omegaconf import OmegaConf

from . import (
    core,
    design_nets,
    design_objectives,
    designers,
    envs,
    inference,
    methods,
    tasks,
)

# Lets recipes write `device: ${auto_device:}`.
OmegaConf.register_new_resolver(
    "auto_device", lambda: "cuda" if torch.cuda.is_available() else "cpu", replace=True
)
