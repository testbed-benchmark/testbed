"""Amortized (policy-net-backed) designer abstractions."""

from collections.abc import Sequence

import torch

from ..design_nets.stochastic import StochasticDesignNet
from .base import Designer
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
        transform_designs: bool = True,
        use_raw_history: bool = True,
    ):
        """Stores the policy network.

        use_raw_history: if True (default), condition the net on the raw
        design history once one is available, instead of the constrained
        history -- for numerically stable conditioning when a task's
        raw<->design transform has a large dynamic range. Lives here rather
        than on Designer because a policy net is the only kind of designer
        that conditions on the design history at all; the others ignore it.
        """
        super().__init__(
            task,
            T,
            hooks=hooks,
            transform_designs=transform_designs,
        )
        self.design_net = design_net
        self.use_raw_history = bool(use_raw_history)

    @property
    def device(self) -> torch.device:
        """Returns the policy network device."""
        return self.design_net.device

    def design_history_for_step(self, state) -> torch.Tensor:
        """Returns the design history to condition the net on.

        The raw history when use_raw_history is set and one exists (so from
        t=1 onwards); the constrained history otherwise, including at t=0,
        when no design has been produced yet.
        """
        if self.use_raw_history and state.raw_design_history is not None:
            return state.raw_design_history
        return state.design_history


class DeterministicPolicyDesigner(PolicyDesigner):
    """Policy designer backed by a plain deterministic net (forward() only)."""

    def design_from_history(self, y_history, design_history):
        """Evaluates the policy net on a history and constrains its output.

        No mask argument: a deterministic policy net's ``forward()`` has none.
        For padded-history evaluation, call the net directly.

        Args:
            y_history (Tensor):
                Outcomes so far, ``[b, t, *event_shape_y]``.
            design_history (Tensor):
                Designs so far, ``[b, t, *event_shape_design]``, in whichever
                coordinates the net expects (see ``use_raw_history``).

        Returns:
            Tensor:
                The design, ``[b, *event_shape_design]``.
            Tensor:
                The net's raw output it was mapped from, same shape.
        """
        raw_design = self.design_net(y_history, design_history)
        return self._to_design(raw_design), raw_design

    def _step_design(self, state):
        """Adapts design_from_history to the rollout state."""
        design, raw_design = self.design_from_history(
            state.y_history, self.design_history_for_step(state)
        )
        return design, raw_design, None


class StochasticPolicyDesigner(PolicyDesigner):
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
        Note there is no transform_designs param here -- this net owns its own squash
        transform (baked into its sampled log-prob), so it can't be swapped
        out at the designer level; use build_policy_designer if you want a
        clear error when transform_designs=False is combined with a
        StochasticDesignNet.
        """
        super().__init__(
            task,
            T,
            design_net,
            hooks=hooks,
            use_raw_history=use_raw_history,
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

    def _step_design(self, state):
        """Adapts the net's sample() to the rollout state.

        Returns sample.raw_design (the pre-squash Gaussian value) alongside the
        design; that is what lets rollout() feed the raw value back as history
        instead of the squashed design.
        """
        sample = self._sample(
            state.y_history, self.design_history_for_step(state), mask=state.mask
        )
        return sample.design, sample.raw_design, sample.log_prob


def build_policy_designer(
    task,
    T,
    design_net,
    *,
    reparameterize_policy: bool = False,
    hooks: Sequence[RolloutHook] = (),
    use_raw_history: bool = True,
    transform_designs: bool = True,
):
    """Constructs the PolicyDesigner matching the policy net's type.

    Dispatches to StochasticPolicyDesigner for a StochasticDesignNet, else
    DeterministicPolicyDesigner. reparameterize_policy is only meaningful
    (and only forwarded) for the stochastic case. transform_designs is only
    meaningful (and only forwarded) for the deterministic case -- a
    StochasticDesignNet owns its own squash transform, baked into its
    sampled log-prob, so it can't be swapped out; passing
    transform_designs=False alongside a StochasticDesignNet raises, rather
    than silently ignoring it.
    """
    if isinstance(design_net, StochasticDesignNet):
        if not transform_designs:
            raise TypeError(
                "transform_designs=False is not supported for "
                "StochasticPolicyDesigner: its net always owns its own squash "
                "transform, baked into the sampled log-prob, so there is no "
                "designer-level way to bypass it."
            )
            if isinstance(design_net, CategoricalPolicyNet):
                use_raw_history = False
        return StochasticPolicyDesigner(
            task,
            T,
            design_net,
            reparameterize_policy=reparameterize_policy,
            hooks=hooks,
            use_raw_history=use_raw_history,
        )
    return DeterministicPolicyDesigner(
        task,
        T,
        design_net,
        hooks=hooks,
        use_raw_history=use_raw_history,
        transform_designs=transform_designs,
    )
