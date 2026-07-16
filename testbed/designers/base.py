"""Abstract base classes for Designer objects."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from testbed.core.rollout import Rollout

from .hooks import RolloutHook, RolloutState


class Designer(ABC, nn.Module):
    """Base class for deterministic designers: produce a design, no log-prob.

    Subclass this directly for designers whose design isn't randomly sampled
    (e.g. StaticDesigner, SequentialDesigner, DeterministicPolicyDesigner).
    For designers whose design *is* randomly sampled, subclass
    StochasticDesigner instead.
    """

    def __init__(
        self,
        task,
        T,
        *,
        hooks: Sequence[RolloutHook] = (),
        use_raw_history: bool = True,
        expose_raw_design: bool = False,
        transform_designs: bool = True,
    ):
        """Stores the task, horizon, and the hooks this designer always runs.

        hooks: composable "extra bits" run on every rollout() call from this
        designer (e.g. a posterior update after every step, or an action net
        at the end). Fixed at construction, not passed to rollout() -- a
        designer's hooks don't change between calls, and self.hooks is a
        plain mutable list if you need to append one later.
        use_raw_history: if True (default), rollout() feeds the raw design
        history back into _step_design once one is available, instead of the
        constrained history -- for numerically stable conditioning when a
        task's raw<->design transform has a large dynamic range. No-op for
        designers that never produce a raw design (e.g. StaticDesigner,
        SequentialDesigner, RandomDesigner).
        expose_raw_design: should we return the raw designs?
        transform_designs: if True (default), a raw value a designer
        produces (a learned parameter, an optimized tensor, or a
        design_net's output) is mapped into a valid design via
        task.design_space.to_design/.to_raw. A Task only ever consumes and
        validates already-constrained designs (Task._check_design) -- it
        never performs this mapping itself, so it's a designer-layer policy
        choice, not a task concern. Set False if your design_net/parameter
        already produces valid, constrained designs and you don't want them
        remapped -- raw and design then coincide exactly (self._to_design/
        self._to_raw become the identity). This is an assumption you are
        making, not something enforced here: Task._check_design still
        validates the final design downstream when task.validate_designs is
        True, so an invalid passthrough design still raises there. Only
        meaningful for designers that compute a raw value via a pointwise
        transform (StaticDesigner, SequentialDesigner,
        DeterministicPolicyDesigner) -- unused by StochasticPolicyDesigner,
        whose net owns its own squash transform because it must enter the
        sampled log-prob (see TanhGaussianPolicyNet), and unused by
        RandomDesigner, which never produces a raw value.
        """
        super().__init__()

        self.task = task
        self.T = T
        self.hooks = list(hooks)
        self.use_raw_history = bool(use_raw_history)
        self.expose_raw_design = bool(expose_raw_design)
        self.transform_designs = bool(transform_designs)

    def _to_design(self, raw_design):
        """Maps a raw value into a valid design via task.design_space.to_design.

        Identity when self.transform_designs is False.
        """
        if not self.transform_designs:
            return raw_design
        return self.task.design_space.to_design(raw_design)

    def _to_raw(self, design):
        """Inverse of _to_design, via task.design_space.to_raw.

        Identity when self.transform_designs is False -- always exactly
        consistent with _to_design, since both are gated by the same flag.
        """
        if not self.transform_designs:
            return design
        return self.task.design_space.to_raw(design)

    @abstractmethod
    def propose_design(self, y_history, design_history, *, mask=None):
        """Proposes the next design."""
        # y_history, design_history: [b=1, T, **dim]
        # mask: optional boolean mask [b=1, T] for padded histories
        # returns: [b=1, **dim_design]
        ...

    def _step_design(self, y_history, design_history):
        """Proposes the next design, returning (design, raw_design, policy_log_prob)."""
        return self.propose_design(y_history, design_history), None, None

    def make_empty_history(self, b):
        """Returns an empty rollout (i.e., corresponding to t=0)."""
        y_history = torch.empty(
            (b, 0, *self.task.event_shape_y), device=self.task.device
        )
        design_history = torch.empty(
            (b, 0, *self.task.event_shape_design), device=self.task.device
        )
        return y_history, design_history

    def rollout(self, theta) -> Rollout:
        """Runs a rollout for batched theta values.

        theta: [b, *event_shape_theta]
        returns: a Rollout with per-step policy log probs populated whenever
            the designer's proposal exposes them. Runs self.hooks (set at
            construction) alongside the rollout.
        """
        hooks = self.hooks
        b = theta.shape[0]
        T = self.T

        # Initialize a mutable state; hooks will modify this in-place
        y_history, design_history = self.make_empty_history(b)
        state = RolloutState(
            task=self.task,
            theta=theta,
            T=T,
            y_history=y_history,
            design_history=design_history,
        )

        y_list = []
        design_list = []
        raw_design_list = []
        policy_log_prob_list = []

        for t in range(T):
            state.t = t

            for hook in hooks:
                hook.before_propose(state)

            # Should we use design_history or raw_design_history for this step?
            # At t=0, use the empty design history
            design_history_for_step = (
                state.raw_design_history
                if self.use_raw_history and state.raw_design_history is not None
                else state.design_history
            )

            # Produce the next design, possibly with raw/logprobs
            design, raw_design, policy_log_prob = self._step_design(
                state.y_history, design_history_for_step
            )

            state.design = design
            state.raw_design = raw_design
            state.policy_log_prob = policy_log_prob

            for hook in hooks:
                hook.after_propose(state)

            for hook in hooks:
                hook.before_sample_outcome(state)

            # Sample observation; always using the (possibly transformed) design
            y = self.task.sample_outcome(theta, design)  # [b, *event_shape_y]
            state.y = y

            y_list.append(y)
            design_list.append(design)
            raw_design_list.append(raw_design)
            policy_log_prob_list.append(policy_log_prob)
            state.y_history = torch.stack(y_list, dim=1)
            state.design_history = torch.stack(design_list, dim=1)
            state.raw_design_history = (
                torch.stack(raw_design_list, dim=1)
                if all(rd is not None for rd in raw_design_list)
                else None
            )

            for hook in hooks:
                hook.after_step(state)

        policy_log_probs = None
        if all(log_prob is not None for log_prob in policy_log_prob_list):
            policy_log_probs = torch.stack(policy_log_prob_list, dim=1)  # [b, T]

        extra = {}
        if state.raw_design_history is not None and self.expose_raw_design:
            extra["raw_design_history"] = state.raw_design_history
        if policy_log_probs is not None:
            extra["policy_log_probs"] = policy_log_probs

        rollout = Rollout(
            theta=theta,
            y_history=state.y_history,
            design_history=state.design_history,
            extra=extra,
        )

        for hook in hooks:
            extra_update = hook.on_rollout_end(state)
            if extra_update is None:
                continue
            colliding_keys = rollout.extra.keys() & extra_update.keys()
            if colliding_keys:
                raise ValueError(
                    f"Multiple hooks wrote the same Rollout.extra key(s): {colliding_keys}"
                )
            rollout.extra.update(extra_update)

        return rollout


class StochasticDesigner(Designer):
    """Base class for designers whose design is randomly sampled.

    sample_design's log-prob is optional, not guaranteed: a stochastic
    designer is defined by its design being randomly sampled, not by
    exposing a log-prob for it. An implicit/black-box sampler is still a
    StochasticDesigner even if it returns log_prob=None; that just means
    it can't be used with a score-function/REINFORCE gradient that needs one.
    """

    @abstractmethod
    def sample_design(self, y_history, design_history, *, mask=None):
        """Samples the next design.

        returns: (design, log_prob or None). log_prob has shape [b].
        """
        ...

    def propose_design(self, y_history, design_history, *, mask=None):
        """Returns just the sampled design, discarding the log-prob."""
        design, _ = self.sample_design(y_history, design_history, mask=mask)
        return design

    def _step_design(self, y_history, design_history):
        """Proposes the next design, returning (design, raw_design=None, policy_log_prob)."""
        design, log_prob = self.sample_design(y_history, design_history)
        return design, None, log_prob
