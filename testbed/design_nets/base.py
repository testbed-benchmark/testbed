"""Base design-network interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class DesignProposal:
    """A raw design proposed by a design network.

    Attributes:
        raw_design: Design in the network's output coordinates.
        log_prob: Optional log probability of the raw design.
        deterministic_raw_design: Optional deterministic counterpart to a sampled
            raw design.
        info: Additional proposal metadata.
    """

    raw_design: Tensor
    log_prob: Tensor | None = None
    deterministic_raw_design: Tensor | None = None
    info: dict[str, Any] = field(default_factory=dict)


class DesignNet(torch.nn.Module, ABC):
    """Maps experimental history to a new design."""

    produces_policy_log_probs = False
    supports_reparameterized_proposals = False

    def __init__(self):
        """Initializes the network."""
        super().__init__()

    @property
    def device(self) -> torch.device:
        """Returns the network device."""
        return next(self.parameters()).device

    @abstractmethod
    def forward(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Maps past (design, outcome) pairs to the next design x_t.

        Args:
            y_history: Outcome history.
            design_history: Design history.
            mask: Optional mask for padded histories.

        Returns:
            Tensor: The next design x_t with shape
            ``[*batch_shape, *event_shape_design]``.
        """
        ...

    def propose(
        self,
        y_history,
        design_history,
        *,
        mask: Tensor | None = None,
        reparameterize_policy: bool = False,
    ) -> DesignProposal:
        """Returns a raw design proposal for an experimental history.

        Deterministic design networks use their forward output as the raw design.
        Stochastic design networks may override this method to sample a proposal.

        Args:
            y_history: Outcome history.
            design_history: Design history.
            mask: Optional mask for padded histories.
            reparameterize_policy: Whether stochastic policies should use a
                reparameterized sample. Ignored by the deterministic default.

        Returns:
            A proposal containing the network's raw design.
        """
        del reparameterize_policy
        raw_design = self(y_history, design_history, mask=mask)
        return DesignProposal(raw_design=raw_design)
