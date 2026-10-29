"""Non-adaptive designers."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Distribution as TorchDist

from testbed.core.distributions import Distribution, TorchDistribution
from testbed.core.hooks import RolloutHook
from testbed.core.rollout import Rollout
from testbed.designers.base import Designer
from testbed.tasks import Task


class StaticDesigner(Designer):
    """A designer with a single learnable design sequence ``[T, *event_shape_design]``.

    Note that this is a static/batch design, and designs are not selected adaptively.

    By default, this method parametrises a raw design which is reparametrised using the Task's
    design_transform. The raw design is initialized as a random standard Gaussian. Other strategies
    (e.g., different intialisations) should subclass this, or otherwise implement their own Designer.
    """

    def __init__(
        self,
        task,
        T,
        *,
        hooks: Sequence[RolloutHook] = (),
        transform_designs: bool = True,
    ):
        """Initializes learnable static designs.

        transform_designs: see Designer.__init__. Default True: self.raw_design
        is mapped through task.design_transform before use.
        """
        Designer.__init__(
            self,
            task,
            T,
            hooks=hooks,
            transform_designs=transform_designs,
        )

        self.raw_design = self._init_design()

    @property
    def device(self) -> torch.device:
        """Returns the design parameter device."""
        return self.raw_design.device

    @property
    def design(self) -> Tensor:
        """Returns the physical design."""
        return self._to_design(self.raw_design)

    def _init_design(self) -> Tensor:
        """Initializes the (raw) design as a standard Gaussian.

        You can override this method for custom initialisation logic.
        """
        task = self.task
        return nn.Parameter(
            torch.randn(self.T, *task.event_shape_design, device=task.device)
        )

    def design_at_step(self, t: int, batch_size: int) -> tuple[Tensor, Tensor]:
        """Returns step ``t``'s learnable design, broadcast over the batch.

        The design sequence is static, so it depends only on the step index --
        not on anything observed so far.

        Args:
            t (int):
                Step index, in ``range(T)``.
            batch_size (int):
                How many copies of the design to return.

        Returns:
            tuple[Tensor, Tensor]: The physical design and the raw design it
            was mapped from, each with shape
            ``[batch_size, *event_shape_design]``.
        """
        raw = (
            self.raw_design[t]
            .unsqueeze(0)
            .expand(batch_size, *self.task.event_shape_design)
        )
        return self._to_design(raw), raw

    def _step_design(self, state):
        """Adapts design_at_step to the rollout state."""
        design, raw = self.design_at_step(state.t, state.y_history.shape[0])
        return design, raw, None

    def rollout(self, theta) -> Rollout:
        """Rolls out the experiment with a static design.

        Vectorized fast path when no hooks are attached and the task is
        memoryless (the common case); falls back to the shared per-step
        template otherwise.
        """
        if self.hooks or not self.task.memoryless:
            return super().rollout(theta)

        b = theta.shape[0]
        design = self.design  # [T, *event_shape_design]

        y_history = self.task.sample_outcome(theta.unsqueeze(1), design.unsqueeze(0))
        design_history = design.unsqueeze(0).expand(
            b, self.T, *self.task.event_shape_design
        )

        extra = {
            "raw_design_history": self.raw_design.unsqueeze(0).expand(
                b, self.T, *self.task.event_shape_design
            )
        }
        return Rollout(
            theta=theta,
            y_history=y_history,
            design_history=design_history,
            extra=extra,
        )


class RandomDesigner(Designer):
    """Samples designs randomly from a fixed distribution.

    Args:
        task (Task):
            the Task we are producing designs for.
        T (int):
            The number of experiments to design.
        dist (Distribution, optional):
            The distribution from which to draw random design (default: None).
            If no distribution is passed, designs are drawn from the task's
            design_prior.
        hooks: (Sequence[RolloutHook]):
            Optional additional hooks during the rollout.
    """

    def __init__(
        self,
        task: Task,
        T: int,
        *,
        dist: Distribution | TorchDist | None = None,
        hooks: Sequence[RolloutHook] = (),
    ):
        """Initializes with a design distribution."""
        super().__init__(task, T, hooks=hooks)

        if isinstance(dist, TorchDist) and not isinstance(dist, Distribution):
            dist = TorchDistribution(dist)
        self.dist = dist

    @property
    def device(self) -> torch.device:
        """Returns the distribution device, or the task's when sampling the space."""
        return self.task.device if self.dist is None else self.dist.device

    def _sample_designs(self, n: int) -> tuple[Tensor, Tensor | None]:
        """Draws ``n`` designs, and their log probs when a dist was supplied.

        The single place designs are drawn, so sample_design and rollout's
        vectorized fast path cannot disagree about which source is used.

        Args:
            n (int):
                How many designs to draw.

        Returns:
            tuple[Tensor, Tensor | None]: Random designs with shape
            ``[n, *event_shape_design]`` and their log probabilities with
            shape ``[n]``. The log probabilities are ``None`` when sampling
            from the task's design prior.
        """
        if self.dist is None:
            design = self.task.design_prior.sample((n,)).to(self.task.device)
            return design, None
        design = self.dist.sample(n)
        return design, self.dist.log_prob(design)

    def sample_design(self, y_history: Tensor, design_history: Tensor, *, mask=None):
        """Samples one design per batch item, and its log prob under dist.

        Args:
            y_history (Tensor):
                Shape ``[b, T, *event_shape_y]``.
            design_history (Tensor):
                Shape ``[b, T, *event_shape_design]``.
            mask (Tensor, optional):
                Unused; accepted for interface parity, as designs do not
                depend on the history.

        Returns:
            tuple[Tensor, Tensor | None]: Random designs with shape
            ``[b, *event_shape_design]`` and their log probabilities with
            shape ``[b]``. The log probabilities are ``None`` when sampling
            from the task's design prior.
        """
        return self._sample_designs(y_history.shape[0])

    def _maybe_to_raw(self, design: Tensor) -> Tensor | None:
        """Returns the raw counterpart of ``design``, or None if there is none.

        Designs are drawn physically here, so the raw values a policy would have
        proposed are recovered by inverting the design transform. Anything
        conditioning on raw history (a policy net, an ``ActionHook``) then sees
        the same coordinates during random warmup as during the policy phase.
        Transforms with no inverse report None, and callers fall back to
        physical designs as before.
        """
        try:
            return self._to_raw(design)
        except NotImplementedError:
            return None

    def _step_design(self, state):
        """Adapts sample_design to the rollout state."""
        design, log_prob = self.sample_design(
            state.y_history, state.design_history, mask=state.mask
        )
        return design, self._maybe_to_raw(design), log_prob

    def rollout(self, theta) -> Rollout:
        """Runs a random design rollout.

        Vectorized fast path when no hooks are attached and the task is
        memoryless; falls back to the shared per-step template otherwise.
        """
        if self.hooks or not self.task.memoryless:
            return super().rollout(theta)

        b = theta.shape[0]

        design_flat, log_prob_flat = self._sample_designs(b * self.T)
        design_history = design_flat.reshape(b, self.T, *self.task.event_shape_design)

        theta_expanded = theta.unsqueeze(1)
        y_history = self.task.sample_outcome(theta_expanded, design_history)

        extra = {}
        if log_prob_flat is not None:
            extra["policy_log_probs"] = log_prob_flat.reshape(b, self.T)
        raw_design_history = self._maybe_to_raw(design_history)
        if raw_design_history is not None:
            extra["raw_design_history"] = raw_design_history

        return Rollout(
            theta=theta,
            y_history=y_history,
            design_history=design_history,
            extra=extra,
        )
