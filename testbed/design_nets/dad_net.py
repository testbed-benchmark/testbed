"""DAD-style continuous policy networks."""

from collections.abc import Sequence

import torch
import torch.nn as nn

from testbed.design_nets.base_design_net import DesignNet
from testbed.design_nets.modules import MLP


class DADNet(DesignNet):
    """Canonical continuous DAD-style policy network.

    The default architecture follows the public DAD location-finding code more
    closely than Appendix D.1 of the DAD paper. Appendix D.1 describes a deeper
    linear readout, but the original code uses a summed history embedding and a
    single linear emitter:

    - Paper: https://ardesignv.org/abs/2103.02438
    - DAD code: https://github.com/ae-foster/dad/blob/main/location_finding.py#L31
    - DAD module: https://github.com/ae-foster/dad/blob/main/neural/modules.py#L61

    Set ``encoder_depth=2`` and pass ``emitter_dims=[128, 16]`` to recover the
    extra depth used by the earlier StepDAD-inspired experimental variant:
    https://github.com/marcelhedman/stepdad/blob/master/src/stepdad/policy/dad.py#L73.
    """

    def __init__(
        self,
        dim_y: int = 1,
        dim_design: int = 2,
        dim_h: int = 256,
        dim_emb: int = 16,
        encoder_depth: int = 1,
        emitter_dims: Sequence[int] | None = None,
    ):
        """Initializes network layers.

        Args:
            dim_y: Outcome event dimension.
            dim_design: Design event dimension.
            dim_h: Hidden dimension for the history encoder.
            dim_emb: Summed history embedding dimension.
            encoder_depth: Number of hidden ``dim_h`` layers before ``dim_emb``.
            emitter_dims: Optional hidden dimensions for the readout. ``None``
                gives the DAD-code-compatible single linear emitter.
        """
        super().__init__()

        encoder_dims = [dim_y + dim_design, *([dim_h] * encoder_depth), dim_emb]
        self.encoder_net = MLP(encoder_dims, activation=nn.ReLU)

        emitter_hidden_dims = [] if emitter_dims is None else list(emitter_dims)
        emitter_dims_full = [dim_emb, *emitter_hidden_dims, dim_design]
        if emitter_hidden_dims:
            self.emitter_net = MLP(emitter_dims_full, activation=nn.ReLU)
        else:
            self.emitter_net = nn.Linear(dim_emb, dim_design)

        self.dim_emb = dim_emb

    def forward(self, y_history, design_history):
        """Maps histories to designs.

        y_history: [b, T, dim_y]
        design_history: [b, T, dim_design].
        """
        b, T, _ = y_history.shape

        if T == 0:
            enc = y_history.new_zeros((b, self.dim_emb))
        else:
            enc_input = torch.cat(
                [y_history, design_history], dim=-1
            )  # [b, T, dim_y + dim_design]
            enc = self.encoder_net(enc_input)  # [b, T, dim_emb]
            enc = enc.sum(dim=1)  # [b, dim_emb]

        out = self.emitter_net(enc)  # [b, dim_design]
        return out
