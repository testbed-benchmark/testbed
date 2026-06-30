"""Design objective exports."""

# Included so that you can do `from testbed.design_objectives import MSELoss` (for consistency with the other losses)
from torch.nn import MSELoss

from .loss import LogMSELoss, SortedLogMSELoss, SortedMSELoss
from .snmc import NMC, SNMC
from .spce import PCE, SPCE

__all__ = [
    "NMC",
    "PCE",
    "SNMC",
    "SPCE",
    "MSELoss",
    "LogMSELoss",
    "SortedMSELoss",
    "SortedLogMSELoss",
]
