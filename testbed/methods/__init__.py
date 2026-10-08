"""Policy-training methods.

These build their components from Hydra configs, so they need the optional
``[train]`` dependencies: ``pip install testbed[train]``. Their command-line
entry points live in ``testbed.cli``.
"""

from .actionbed import Trainer as ActionBEDTrainer
from .aline import Trainer as AlineTrainer
from .dad import Trainer as DADTrainer
from .rlboed import Trainer as RLBOEDTrainer
from .var_post import Trainer as VarPostTrainer
