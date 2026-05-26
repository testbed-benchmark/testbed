"""DAD policy networks."""

import torch
import torch.nn as nn

from testbed.design_nets import DesignNet
from testbed.design_nets.modules import MLP


class DADNet(DesignNet):
    """Location Finding policy network from the DAD paper.

    c.f. Appendix D.1, https://arxiv.org/abs/2103.02438.
    """

    def __init__(self, dim_y=1, dim_xi=2, dim_h=256, dim_emb=16):
        """Initializes network layers."""
        super().__init__()

        self.encoder_net = MLP([dim_y + dim_xi, dim_h, dim_emb], activation=nn.ReLU)
        self.emitter_net = nn.Sequential(
            nn.Linear(dim_emb, dim_xi),
            nn.Linear(dim_xi, dim_xi),
            nn.Linear(dim_xi, dim_xi),
        )

        # Encoding used when T=0
        # ToDo: how does DAD actually handle this?
        self.null_embedding = nn.Parameter(torch.randn(1, dim_emb))

    def forward(self, y_history, xi_history):
        """Maps histories to designs.

        y_history: [b, T, dim_y]
        xi_history: [b, T, dim_xi].
        """
        b, T, _ = y_history.shape

        # Handle T=0 with a learnable null embedding
        emb = self.null_embedding.expand(b, -1)

        if T > 0:
            x = torch.cat([y_history, xi_history], dim=-1)  # [b, T, dim_y + dim_xi]

            enc = self.encoder_net(x)  # [b, T, dim_enc]
            emb = emb + enc.sum(dim=1)  # [b, dim_enc]

        out = self.emitter_net(emb)  # [b, dim_xi]
        return out


class DADNet_v2(DesignNet):
    """Location Finding policy network from the DAD code.

    c.f. Appendix D.1, https://arxiv.org/abs/2103.02438
    NB: Appendix D.1 does not actually match the DAD code
    c.f. https://github.com/ae-foster/dad/blob/main/location_finding.py#L31
    c.f. https://github.com/ae-foster/dad/blob/main/neural/modules.py#L61.

    This network re-implements the network from that file as closely as possible
    """

    def __init__(self, dim_y=1, dim_xi=2, dim_h=256, dim_emb=16):
        """Initializes network layers."""
        super().__init__()

        self.encoder_net = MLP([dim_y + dim_xi, dim_h, dim_emb], activation=nn.ReLU)
        self.emitter_net = nn.Sequential(nn.Linear(dim_emb, dim_xi))

        self.dim_emb = dim_emb

    def forward(self, y_history, xi_history):
        """Maps histories to designs.

        y_history: [b, T, dim_y]
        xi_history: [b, T, dim_xi].
        """
        b, T, _ = y_history.shape

        # Handle T=0 by setting the embedding to zero
        if T == 0:
            enc = torch.zeros((b, self.dim_emb), device=y_history.device)
        else:
            enc_input = torch.cat(
                [y_history, xi_history], dim=-1
            )  # [b, T, dim_y + dim_xi]
            enc = self.encoder_net(enc_input)  # [b, T, dim_enc]
            enc = enc.sum(dim=1)  # [b, dim_enc]

        out = self.emitter_net(enc)  # [b, dim_xi]
        return out


class DADNet_v3(DesignNet):
    """Policy network inspired by StepDAD.

    https://github.com/marcelhedman/stepdad/blob/master/src/stepdad/policy/dad.py#L73.
    """

    def __init__(self, dim_y=1, dim_xi=2, dim_h=256, dim_emb=16):
        """Initializes network layers."""
        super().__init__()

        self.encoder_net = MLP(
            [dim_y + dim_xi, dim_h, dim_h, dim_emb], activation=nn.ReLU
        )
        self.emitter_net = MLP([dim_emb, 128, 16, dim_xi], activation=nn.ReLU)

        # Embedding for T=0 i.e. no history -- initialized to zero but learnable
        self.null_enc = nn.Parameter(torch.zeros(dim_emb))

    def forward(self, y_history, xi_history):
        """Maps histories to designs.

        y_history: [b, T, dim_y]
        xi_history: [b, T, dim_xi].
        """
        b, T, _ = y_history.shape

        if T == 0:
            enc = self.null_enc.unsqueeze(0).expand(b, -1)
        else:
            enc_input = torch.cat(
                [y_history, xi_history], dim=-1
            )  # [b, T, dim_y + dim_xi]
            enc = self.encoder_net(enc_input)  # [b, T, dim_enc]
            enc = enc.sum(dim=1)  # [b, dim_enc]

        out = self.emitter_net(enc)  # [b, dim_xi]
        return out
