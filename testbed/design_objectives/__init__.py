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
from .reinforce import LossReinforce, Reinforce, SPCEReinforce, VarPostReinforce
from .snmc import NMC, SNMC
from .spce import PCE, SPCE
from .var_post import VariationalPosteriorNLL, VarPost

__all__ = [
    "NMC",
    "PCE",
    "SNMC",
    "SPCE",
    "DesignObjective",
    "VariationalPosteriorNLL",
    "LossReinforce",
    "MSELoss",
    "LogLoss",
    "LogMSELoss",
    "Objective",
    "PermutationInvariantMSELoss",
    "Reinforce",
    "RolloutObjective",
    "SPCEReinforce",
    "SortedLogMSELoss",
    "SortedMSELoss",
    "VarPost",
    "VarPostReinforce",
]
