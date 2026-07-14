"""Design objective exports."""

# Included so that you can do `from testbed.design_objectives import MSELoss` (for consistency with the other losses)
from torch.nn import MSELoss

from .base_objective import DesignObjective, Objective, RolloutObjective
from .log_loss import LogLoss
from .loss import (
    LogMSELoss,
    PermutationInvariantMSELoss,
    SortedLogMSELoss,
    SortedMSELoss,
)
from .reinforce import LossReinforce, Reinforce, SPCEReinforce
from .snmc import NMC, SNMC
from .spce import PCE, SPCE

__all__ = [
    "NMC",
    "PCE",
    "SNMC",
    "SPCE",
    "DesignObjective",
    "LossReinforce",
    "MSELoss",
    "LogLoss",
    "LogMSELoss",
    "Objective",
    "Reinforce",
    "RolloutObjective",
    "SPCEReinforce",
    "SortedMSELoss",
    "SortedLogMSELoss",
    "PermutationInvariantMSELoss",
]
