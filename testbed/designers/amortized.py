"""Amortized (policy-net-backed) designer abstractions."""

from collections.abc import Sequence

import torch

from ..design_nets.stochastic import StochasticDesignNet
from .base import Designer, StochasticDesigner
from .hooks import RolloutHook


class PolicyDesigner(Designer):
    """Common base for designers backed by a trainable policy network.

    Construct via build_policy_designer(...), which dispatches to
    DeterministicPolicyDesigner or StochasticPolicyDesigner based on the
    policy net's type -- there is no runtime "run this stochastic net
    deterministically" mode; a net is one or the other.
    """

    def __init__(
        self,
        task,
        T,
        design_net,
        *,
        hooks: Sequence[RolloutHook] = (),
        use_raw_history: bool = True,
    ):
        """Stores the policy network."""
        super().__init__(task, T, hooks=hooks, use_raw_history=use_raw_history)
        self.design_net = design_net

    @property
    def device(self) -> torch.device:
        """Returns the policy network device."""
        return self.design_net.device


class DeterministicPolicyDesigner(PolicyDesigner):
    """Policy designer backed by a plain deterministic net (forward() only)."""

    def _propose(self, y_history, design_history):
        """Evaluates the net once; returns (constrained_design, raw_design)."""
        raw_design = self.design_net(y_history, design_history)
        design = self.task.design_space.to_design(raw_design)
        return design, raw_design

    def propose_design(self, y_history, design_history, *, mask=None):
        """Evaluates the policy network and maps its raw output into the task's design space.

        mask is accepted for interface parity but unused: a deterministic
        policy net's forward() has no mask argument. For padded-history eval,
        call the net directly instead of going through this designer.
        """
        # y_history, design_history: [b, T, **dim]
        # returns: [b, **dim_design]
        design, _ = self._propose(y_history, design_history)
        return design

    def _step_design(self, y_history, design_history):
        """Proposes the next design, returning (design, raw_design, policy_log_prob=None)."""
        design, raw_design = self._propose(y_history, design_history)
        return design, raw_design, None

    @property
    def reinforce_requires_policy_log_probs(self) -> bool:
        """Deterministic policies never need policy log probs (pure pathwise gradient)."""
        # TODO possibly remove
        return False


class StochasticPolicyDesigner(PolicyDesigner, StochasticDesigner):
    """Policy designer backed by a StochasticDesignNet."""

    def __init__(
        self,
        task,
        T,
        design_net,
        *,
        reparameterize_policy: bool = False,
        hooks: Sequence[RolloutHook] = (),
        use_raw_history: bool = True,
    ):
        """Stores the policy network.

        reparameterize_policy: sample designs with the reparameterization
        trick so they carry the graph (pathwise gradients); False gives pure
        score log probs (score-function/REINFORCE gradients).
        """
        super().__init__(
            task, T, design_net, hooks=hooks, use_raw_history=use_raw_history
        )
        self.reparameterize_policy = bool(reparameterize_policy)

    def _sample(self, y_history, design_history, *, mask=None):
        """Evaluates the net's sample() once."""
        return self.design_net.sample(
            y_history,
            design_history,
            mask=mask,
            reparameterize_policy=self.reparameterize_policy,
        )

    def sample_design(self, y_history, design_history, *, mask=None):
        """Samples a design from the policy net and its log prob."""
        # y_history, design_history: [b, T, **dim]
        # mask: optional boolean mask [b, T] for padded histories
        sample = self._sample(y_history, design_history, mask=mask)
        return sample.design, sample.log_prob

    def _step_design(self, y_history, design_history):
        """Proposes the next design, returning (design, raw_design, policy_log_prob).

        sample.raw_design (the pre-squash Gaussian value) is what StochasticDesignNet
        already computes internally -- surfacing it here is what lets rollout()
        feed it back as history instead of the squashed design.
        """
        sample = self._sample(y_history, design_history)
        return sample.design, sample.raw_design, sample.log_prob

    @property
    def reinforce_requires_policy_log_probs(self) -> bool:
        """True unless designs are reparameterized (pathwise gradient instead)."""
        return not self.reparameterize_policy


def build_policy_designer(
    task,
    T,
    design_net,
    *,
    reparameterize_policy: bool = False,
    hooks: Sequence[RolloutHook] = (),
    use_raw_history: bool = True,
):
    """Constructs the PolicyDesigner matching the policy net's type.

    Dispatches to StochasticPolicyDesigner for a StochasticDesignNet, else
    DeterministicPolicyDesigner. reparameterize_policy is only meaningful
    (and only forwarded) for the stochastic case.
    """
    if isinstance(design_net, StochasticDesignNet):
        return StochasticPolicyDesigner(
            task,
            T,
            design_net,
            reparameterize_policy=reparameterize_policy,
            hooks=hooks,
            use_raw_history=use_raw_history,
        )
    return DeterministicPolicyDesigner(
        task, T, design_net, hooks=hooks, use_raw_history=use_raw_history
    )
