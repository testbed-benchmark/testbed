"""Design-net wrapper for the hyperbolic temporal-discounting task."""

import torch
from torch import Tensor

from ..base_design_net import DesignNet


class HTDDesignNetWrapper(DesignNet):
    """Maps unconstrained design-net outputs to valid HTD designs (D_1, R_0).

    The task assumes valid designs, so the constraint transform lives here rather
    than in the task. Following Step-DAD, the wrapped net's raw outputs (x_d, x_r)
    are mapped to D_1 = exp(x_d - log_k_mean) and R_0 = 100 * sigmoid(x_r). The
    log_k_mean shift keeps k * D_1 order 1 near zero outputs, so gradients do not
    vanish for an untrained policy; pass prior.k_loc + 0.5 * prior.k_scale**2.

    design_history handed to the wrapper (by the rollout loop) holds valid,
    transformed designs, since that's what the task and objectives consume. But
    D_1 = exp(...) is unbounded, so feeding it straight back into the inner net
    as conditioning would let the net's own past outputs blow up its future ones.
    The wrapper un-transforms design_history back to (x_d, x_r) before it reaches
    the inner net, so the net only ever conditions on its own raw output scale.
    """

    def __init__(self, design_net: DesignNet, log_k_mean: float = 0.0):
        """Wraps design_net; log_k_mean shifts D_1 (default 0.0, no shift)."""
        super().__init__()
        self.net = design_net
        self.log_k_mean = float(log_k_mean)

    def forward(self, y_history, design_history) -> Tensor:
        """Maps the wrapped net's raw output to (D_1, R_0); see class docstring."""

        # This is a hack to make this wrapper match previous DAD implementations
        # TODO clean-up
        design_history_raw = self._invert(design_history)
        
        design_raw = self.net(y_history, design_history_raw)  # [*batch, 2]
        D_raw = design_raw[..., 0:1]
        R_raw = design_raw[..., 1:2]
        D1 = torch.exp(D_raw - self.log_k_mean)
        R0 = 100 * torch.sigmoid(R_raw)
        return torch.cat((D1, R0), dim=-1)  # [*batch, 2]

    def _invert(self, design: Tensor | None) -> Tensor | None:
        """Recovers (x_d, x_r) from a history of valid (D_1, R_0) designs.

        Exact inverse of the forward transform: log undoes exp, logit undoes
        sigmoid. eps clamps logit's input away from 0/1 so a saturated R_0
        (sigmoid rounded to exactly 0 or 1) can't inject +-inf into the net.
        """
        if design is None:
            return None
        D1 = design[..., 0:1]
        R0 = design[..., 1:2]
        D_raw = torch.log(D1.clamp(min=1e-30)) + self.log_k_mean
        R_raw = torch.logit(R0 / 100, eps=1e-6)
        return torch.cat((D_raw, R_raw), dim=-1)
