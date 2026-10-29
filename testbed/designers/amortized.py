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
            transform_designs: Map raw proposals through the task design transform
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
        normalizes_physical_history = bool(self.design_net.normalize_design_history)
        raw_designs_are_physical = bool(self.design_net.raw_designs_are_physical)
        if (
            self.use_raw_history
            and normalizes_physical_history
            and not raw_designs_are_physical
        ):
            raise ValueError(
                "use_raw_history=True is incompatible with a design network "
                "that normalizes physical design histories."
            )

    @property
    def device(self) -> torch.device:
        """Returns the policy network device."""
        return self.design_net.device

    @property
    def requires_policy_log_probs(self) -> bool:
        """Whether training must use proposal score-function log probabilities."""
        produces_log_probs = bool(self.design_net.produces_policy_log_probs)
        supports_reparameterization = bool(
            self.design_net.supports_reparameterized_proposals
        )
        uses_pathwise_samples = (
            self.reparameterize_policy and supports_reparameterization
        )
        return produces_log_probs and not uses_pathwise_samples

    def design_history_for_step(self, state) -> torch.Tensor:
        """Returns the history passed to the policy for a rollout step.

        Args:
            state: Mutable rollout state maintained by :class:`Designer`.

        Returns:
            Tensor: Raw history when ``use_raw_history`` is enabled and
            available; otherwise, physical design history.
        """
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

        Args:
            y_history: Outcome history accepted by the design network.
            design_history: Physical or raw design history accepted by the
                design network according to ``use_raw_history``.
            mask: Optional validity mask for padded history positions.

        Returns:
            tuple[Tensor, Tensor, Tensor | None]: Physical design, raw design,
            and physical-space log probability. The log probability is ``None``
            for deterministic proposals.
        """
        proposal = self.design_net.propose(
            y_history,
            design_history,
            mask=mask,
            reparameterize_policy=self.reparameterize_policy,
        )

        # Some policies, such as CategoricalPolicyNet, choose directly from a
        # finite set of physical designs.  Their log_prob is a categorical
        # log-mass, not a density in the task's raw coordinates, so neither
        # the task transform nor its Jacobian correction applies.
        raw_designs_are_physical = bool(self.design_net.raw_designs_are_physical)
        if not self.transform_designs or raw_designs_are_physical:
            return proposal.raw_design, proposal.raw_design, proposal.log_prob

        transform = self.task.design_transform
        design = transform(proposal.raw_design)
        if proposal.log_prob is None:
            return design, proposal.raw_design, None

        # change of variables into the design's density
        log_prob = proposal.log_prob - transform.log_abs_det_jacobian(
            proposal.raw_design, design
        )
        return design, proposal.raw_design, log_prob

    def _step_design(self, state):
        """Adapts the common proposal path to the rollout state."""
        return self.design_from_history(
            state.y_history,
            self.design_history_for_step(state),
            mask=state.mask,
        )
