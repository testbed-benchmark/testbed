"""Sequential design optimization."""

from collections.abc import Sequence

import torch
from torch.distributions import Distribution as TorchDist

from testbed.core.distributions import Distribution, TorchDistribution
from testbed.core.rollout import Rollout
from testbed.design_objectives import DesignObjective
from testbed.inference.is_sampler import run_importance_sampling
from testbed.tasks import Task

from .base import Designer
from .hooks import PosteriorUpdateHook, RolloutHook


class SequentialDesigner(Designer):
    """Optimizes each design online via per-step gradient ascent on an objective.

    Only supports batch size 1: each design is chosen by running its own
    Adam optimization loop, which doesn't vectorize across independent
    per-example optimizations the way an amortized policy net does.

    Composes a PosteriorUpdateHook by default (update_posterior=True): a
    DesignObjective like NMC/PCE reads task.current_prior, and online
    posterior tracking is what makes this designer "sequential" rather than
    a repeated static design. Pass extra_hooks to run alongside it.
    """

    def __init__(
        self,
        task: Task,
        T: int,
        objective: DesignObjective,
        lr: float,
        n_step_opt: int,
        init_design_dist: Distribution | TorchDist | None = None,
        *,
        update_posterior: bool = True,
        inference_fn=run_importance_sampling,
        extra_hooks: Sequence[RolloutHook] = (),
        expose_raw_design: bool = False,
        **inference_kwargs,
    ):
        """Initializes the sequential designer.

        objective: a DesignObjective -- `estimate_gain(design) -> scalar` for
            a single design (e.g. NMC, PCE). NOT a RolloutObjective like SPCE/
            SNMC/Reinforce, which score a full rollout; use their
            DesignObjective wrapper instead, e.g. PCE(task, n_outer=...,
            n_inner=...) rather than SPCE(task, n_contrastive=...).
        init_design_dist controls how the initial design is drawn before each gradient optimization:
        None             -- standard normal N(0, I)  [default]
        "current_prior"  -- sample one design from task.current_prior (useful when the design
                            space and theta space coincide, e.g. source-finding)
        Distribution     -- any testbed Distribution (or torch.distributions.Distribution);
                            one sample is drawn per call.
        update_posterior: if True (default), a PosteriorUpdateHook runs after
            every step using inference_fn/inference_kwargs, updating
            task.current_prior. Set False to manage posterior updates
            yourself (e.g. via extra_hooks).
        extra_hooks: additional hooks to run alongside the posterior update
            (after it, so they see the up-to-date posterior).
        expose_raw_design: if True, rollout() reports the raw (pre-transform)
            optimized design in rollout.extra["raw_design_history"], for
            inspection/debugging. False by default; the raw design is always
            computed either way, this only gates whether it's surfaced.
        """
        if not isinstance(objective, DesignObjective):
            raise TypeError(
                "SequentialDesigner requires a DesignObjective (estimate_gain(design) "
                f"for a single design, e.g. NMC/PCE); got {type(objective).__name__}. "
                "RolloutObjectives (SPCE/SNMC/Reinforce) score a full rollout instead -- "
                "use their DesignObjective wrapper, e.g. PCE(task, n_outer=..., n_inner=...) "
                "instead of SPCE(task, n_contrastive=...)."
            )

        hooks = list(extra_hooks)
        if update_posterior:
            hooks = [PosteriorUpdateHook(inference_fn, **inference_kwargs), *hooks]
        super().__init__(task, T, hooks=hooks, expose_raw_design=expose_raw_design)

        self.objective = objective
        self.lr = lr
        self.n_step_opt = n_step_opt

        if isinstance(init_design_dist, TorchDist):
            init_design_dist = TorchDistribution(init_design_dist)
        self.init_design_dist = init_design_dist

    @property
    def device(self) -> torch.device:
        """Returns the task device."""
        return self.task.device

    def initialize_design(self):
        """Creates an initial *raw* design for optimization.

        Returns [*event_shape_design] with requires_grad=True, in raw
        coordinates -- propose_design's Adam loop maps this through
        task.design_space.to_design() on every iteration. Override this
        method for custom initialization logic.
        """
        dist = self.init_design_dist
        if dist is None:
            design_raw = torch.randn(
                self.task.event_shape_design, device=self.task.device
            )
        elif dist == "current_prior":
            # sample_current_prior() returns a constrained-space value (e.g. a
            # real source location); convert to raw so every branch returns
            # the same (raw) coordinates for the optimizer.
            design_constrained = self.task.sample_current_prior().to(self.task.device)
            design_raw = self.task.design_space.to_raw(design_constrained)
        else:
            design_raw = (
                dist.sample(1)
                .reshape(self.task.event_shape_design)
                .to(self.task.device)
            )
        return design_raw.requires_grad_(True)

    def _propose(self, y_history, *, mask=None):
        """Runs the Adam loop; returns (constrained_design, raw_design), both [1, *event_shape_design]."""
        if y_history.shape[0] != 1:
            raise ValueError(
                f"SequentialDesigner only supports batch size 1; got batch {y_history.shape[0]}."
            )

        design_raw = self.initialize_design()
        optimizer = torch.optim.Adam([design_raw], lr=self.lr)

        for _ in range(self.n_step_opt):
            optimizer.zero_grad()
            design = self.task.design_space.to_design(design_raw)
            gain = self.objective.estimate_gain(design)
            loss = -gain
            loss.backward()
            optimizer.step()

        # Recompute after the final optimizer.step()
        # Required as the current design is one step stale
        design_raw = design_raw.detach()
        design = self.task.design_space.to_design(design_raw)
        return design.unsqueeze(0), design_raw.unsqueeze(0)

    def propose_design(self, y_history, design_history, *, mask=None):
        """Optimizes and returns the next design via gradient ascent on the objective."""
        # y_history, design_history: [b=1, t, **dim]
        # returns: [b=1, *event_shape_design], constrained coordinates
        design, _ = self._propose(y_history, mask=mask)
        return design

    def _step_design(self, y_history, design_history):
        """Proposes the next design, returning (design, raw_design, policy_log_prob=None)."""
        design, raw_design = self._propose(y_history)
        return design, (raw_design if self.expose_raw_design else None), None

    def rollout(self, theta) -> Rollout:
        """Runs the full sequential experimental design procedure for a single theta.

        theta: ``[*event_shape_theta]`` (unbatched) or ``[1, *event_shape_theta]``.
        """
        if theta.ndim == len(self.task.event_shape_theta):
            theta = theta.unsqueeze(0)
        if theta.shape[0] != 1:
            raise ValueError(
                f"SequentialDesigner only supports batch size 1 rollouts; got batch {theta.shape[0]}."
            )
        return super().rollout(theta)
