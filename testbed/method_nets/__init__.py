"""Networks specific to a single training method.

Unlike :mod:`testbed.design_nets`, these are not interchangeable building
blocks: each is consumed only by its own method (and that method's designer).
They still follow the shared history convention of
:func:`testbed.design_nets.prepare_history`.
"""

from .aline import AlineModel, AlineOutput
from .rlboed import RLBOEDDiscreteQNetwork, RLBOEDQNetwork
