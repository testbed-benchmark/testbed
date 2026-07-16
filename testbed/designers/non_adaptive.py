"""Non-adaptive designers."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Distribution as TorchDist

from testbed.core.distributions import Distribution, TorchDistribution
from testbed.core.rollout import Rollout
from testbed.designers.base import Designer, StochasticDesigner
from testbed.designers.hooks import RolloutHook
from testbed.tasks import Task


class StaticDesigner(Designer):
    """A designer with a single learnable design sequence ``[T, *event_shape_design]``.

    Note that this is a static/batch design, and designs are not selected adaptively.
    """

    def __init__(
        self,
        task,
        T,
        *,
        hooks: Sequence[RolloutHook] = (),
        expose_raw_design: bool = False,
        transform_designs: bool = True,
    ):
        """Initializes learnable static designs.

        transform_designs: see Designer.__init__. Default True: self.raw_design
        is mapped through task.design_space.to_design before use.
        """
        Designer.__init__(
            self,
            task,
            T,
            hooks=hooks,
            expose_raw_design=expose_raw_design,
            transform_designs=transform_designs,
        )
        self.raw_design = nn.Parameter(
            torch.randn(T, *task.event_shape_design, device=task.device)
        )

    @property
    def device(self) -> torch.device:
        """Returns the design parameter device."""
        return self.raw_design.device

    def _propose(self, y_history):
        """Returns (constrained_design, raw_design) for the current step.

        self.raw_design is a raw/latent parameter -- legitimately Gaussian-shaped
        by construction -- transformed through self._to_design (task.design_space
        by default) on the way out, so it can never produce an out-of-domain
        design regardless of where gradient descent takes it, unless
        transform_designs=False opted out of that guarantee.
        """
        # y_history: [b, t, *event_shape_y]; t is the current step index
        t = y_history.shape[1]
        b = y_history.shape[0]
        raw = self.raw_design[t].unsqueeze(0).expand(b, *self.task.event_shape_design)
        design = self._to_design(raw)
        return design, raw

    def propose_design(self, y_history, design_history, *, mask=None):
        """Returns the constrained design for the current step."""
        design, _ = self._propose(y_history)
        return design

    def _step_design(self, y_history, design_history):
        """Proposes the next design, returning (design, raw_design, policy_log_prob=None).

        raw_design is always the actual raw value; Designer.rollout() gates
        whether it's surfaced in rollout.extra based on self.expose_raw_design.
        """
        design, raw = self._propose(y_history)
        return design, raw, None

    def rollout(self, theta) -> Rollout:
        """Runs the static design sequence.

        Vectorized fast path when no hooks are attached (the common case);
        falls back to the shared per-step template whenever hooks need per-step access.
        """
        if self.hooks:
            return super().rollout(theta)

        b = theta.shape[0]
        design = self._to_design(self.raw_design)  # [T, *event_shape_design]

        y_history = self.task.sample_outcome(theta.unsqueeze(1), design.unsqueeze(0))
        design_history = design.unsqueeze(0).expand(
            b, self.T, *self.task.event_shape_design
        )
        extra = {}
        if self.expose_raw_design:
            extra["raw_design_history"] = self.raw_design.unsqueeze(0).expand(
                b, self.T, *self.task.event_shape_design
            )
        return Rollout(
            theta=theta,
            y_history=y_history,
            design_history=design_history,
            extra=extra,
        )


class RandomDesigner(StochasticDesigner):
    """Samples designs randomly from a fixed (non-learned) Distribution."""

    def __init__(
        self,
        task: Task,
        T: int,
        dist: Distribution | TorchDist,
        *,
        hooks: Sequence[RolloutHook] = (),
    ):
        """Initializes with a design distribution."""
        super().__init__(task, T, hooks=hooks)

        if isinstance(dist, TorchDist):
            dist = TorchDistribution(dist)
        self.dist = dist

    @property
    def device(self) -> torch.device:
        """Returns the distribution device."""
        return self.dist.device

    def sample_design(self, y_history, design_history, *, mask=None):
        """Samples one design per batch item and its log prob under dist."""
        # y_history, design_history: [b, T, **dim]
        # returns: (design [b, **dim_design], log_prob [b])
        b = y_history.shape[0]
        design = self.dist.sample(b)  # [b, *event_shape_design]
        log_prob = self.dist.log_prob(design)  # [b]
        return design, log_prob

    def rollout(self, theta) -> Rollout:
        """Runs a random design rollout.

        Vectorized fast path when no hooks are attached; falls back to the
        shared per-step template when hooks need per-step access.
        """
        if self.hooks:
            return super().rollout(theta)

        b = theta.shape[0]

        design_flat = self.dist.sample(b * self.T)  # [b*T, *event_shape_design]
        log_prob_flat = self.dist.log_prob(design_flat)  # [b*T]
        design_history = design_flat.reshape(b, self.T, *self.task.event_shape_design)
        policy_log_probs = log_prob_flat.reshape(b, self.T)

        theta_expanded = theta.unsqueeze(1)
        y_history = self.task.sample_outcome(theta_expanded, design_history)

        return Rollout(
            theta=theta,
            y_history=y_history,
            design_history=design_history,
            extra={"policy_log_probs": policy_log_probs},
        )
