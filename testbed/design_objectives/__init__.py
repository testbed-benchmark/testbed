"""Design objective exports."""

# Included so that you can do `from testbed.design_objectives import MSELoss` (for consistency with the other losses)
from torch.nn import MSELoss

from .barber_agakov import BarberAgakov, posterior_nll
from .base_objective import DesignObjective, Objective, RolloutObjective
from .log_loss import LogLoss
from .loss import (
    LogMSELoss,
    PermutationInvariantMSELoss,
    SortedLogMSELoss,
    SortedMSELoss,
)
from .reinforce import BarberAgakovReinforce, LossReinforce, Reinforce, SPCEReinforce
from .snmc import NMC, SNMC
from .spce import PCE, SPCE

__all__ = [
    "NMC",
    "PCE",
    "SNMC",
    "SPCE",
    "BarberAgakov",
    "BarberAgakovReinforce",
    "DesignObjective",
    "LogLoss",
    "LogMSELoss",
    "LossReinforce",
    "MSELoss",
    "Objective",
    "PermutationInvariantMSELoss",
    "Reinforce",
    "RolloutObjective",
    "SPCEReinforce",
    "SortedLogMSELoss",
    "SortedMSELoss",
    "posterior_nll",
]
