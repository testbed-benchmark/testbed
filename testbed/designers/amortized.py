"""Amortized (policy-net-backed) designer abstractions."""

from collections.abc import Sequence

import torch

from ..core.hooks import RolloutHook
from .base import Designer


class PolicyDesigner(Designer):
    """Designer backed by a trainable design network.

    Every design network supplies a raw :class:`DesignProposal`. This class is
    the single owner of the task design-space transform and any corresponding
    log-probability correction.
    """

    def __init__(
        self,
        task,
        T,
        design_net,
        *,
        reparameterize_policy: bool = False,
        hooks: Sequence[RolloutHook] = (),
        use_raw_history: bool = False,
        transform_designs: bool = True,
    ):
        """Stores the policy network and proposal options.

        Args:
            task: Task for which designs are produced.
            T: Number of experimental steps.
            design_net: Network implementing the common ``propose()`` API.
            reparameterize_policy: Request reparameterized samples from policy
                networks that support them.
            hooks: Optional rollout hooks.
            use_raw_history: Condition on raw rather than physical design
                history. This is an advanced opt-in; physical designs are the
                common public history convention.
            transform_designs: Map raw proposals through the task design space
                and correct proposal log probabilities.
        """
        super().__init__(
            task,
            T,
            hooks=hooks,
            transform_designs=transform_designs,
        )
        self.design_net = design_net
        self.reparameterize_policy = bool(reparameterize_policy)
        self.use_raw_history = bool(use_raw_history)

    @property
    def device(self) -> torch.device:
        """Returns the policy network device."""
        return self.design_net.device

    @property
    def requires_policy_log_probs(self) -> bool:
        """Whether training must use proposal score-function log probabilities."""
        produces_log_probs = bool(
            getattr(self.design_net, "produces_policy_log_probs", False)
        )
        supports_reparameterization = bool(
            getattr(self.design_net, "supports_reparameterized_proposals", False)
        )
        uses_pathwise_samples = (
            self.reparameterize_policy and supports_reparameterization
        )
        return produces_log_probs and not uses_pathwise_samples

    def design_history_for_step(self, state) -> torch.Tensor:
        """Returns the physical history, or raw history when explicitly requested."""
        if self.use_raw_history and state.raw_design_history is not None:
            return state.raw_design_history
        return state.design_history

    def design_from_history(
        self,
        y_history,
        design_history,
        *,
        mask=None,
    ):
        """Proposes and transforms one design for the supplied history.

        Returns:
            A ``(design, raw_design, log_prob)`` tuple. ``log_prob`` is ``None``
            for deterministic proposals.
        """
        proposal = self.design_net.propose(
            y_history,
            design_history,
            mask=mask,
            reparameterize_policy=self.reparameterize_policy,
        )

        if not self.transform_designs:
            return proposal.raw_design, proposal.raw_design, proposal.log_prob

        if proposal.log_prob is None:
            design = self.task.design_space.to_design(proposal.raw_design)
            return design, proposal.raw_design, None

        design, log_prob = self.task.design_space.to_design(
            proposal.raw_design,
            proposal.log_prob,
        )
        return design, proposal.raw_design, log_prob

    def _step_design(self, state):
        """Adapts the common proposal path to the rollout state."""
        return self.design_from_history(
            state.y_history,
            self.design_history_for_step(state),
            mask=state.mask,
        )


def build_policy_designer(
    task,
    T,
    design_net,
    *,
    reparameterize_policy: bool = False,
    hooks: Sequence[RolloutHook] = (),
    use_raw_history: bool = False,
    transform_designs: bool = True,
):
    """Compatibility alias that constructs :class:`PolicyDesigner`."""
    return PolicyDesigner(
        task,
        T,
        design_net,
        reparameterize_policy=reparameterize_policy,
        hooks=hooks,
        use_raw_history=use_raw_history,
        transform_designs=transform_designs,
    )
