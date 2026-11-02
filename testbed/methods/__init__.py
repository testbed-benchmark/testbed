"""Policy-training methods.

These build their components from Hydra configs, so they need the optional
``[train]`` dependencies: ``pip install "torch-testbed[train]"``. Each module's ``main``
is its Hydra command-line entry point::

    python testbed/methods/<name>.py [overrides...]
"""

from .actionbed import Trainer as ActionBEDTrainer
from .aline import Trainer as AlineTrainer
from .dad import Trainer as DADTrainer
from .posthoc_action import Trainer as PosthocActionTrainer
from .rlboed import Trainer as RLBOEDTrainer
from .static import Trainer as StaticTrainer
from .var_post import Trainer as VarPostTrainer
