"""Shared fixed-pool rollouts for ALINE training and evaluation."""

import torch
from torch import Tensor
from torch.distributions import Categorical

from testbed.core.design_spaces import BoxDesignSpace
from testbed.core.hooks import ActionHook, apply_rollout_end
from testbed.core.rollout import Rollout, RolloutState
from testbed.core.util import sort_by_first_coordinate
from testbed.design_nets import AlineOutput

from .base import Designer


class AlineDesigner(Designer):
    """Selects designs from an ALINE candidate pool without replacement."""

    def __init__(
        self,
        task,
        T,
        model,
        *,
        pool_size: int,
        deterministic: bool = True,
    ) -> None:
        """Stores the model and fixed-pool settings."""
        super().__init__(task, T, transform_designs=False)
        if T <= 0:
            raise ValueError("T must be positive.")
        if pool_size < T:
            raise ValueError("pool_size must be at least T.")
        space = task.design_space
        if not isinstance(space, BoxDesignSpace):
            raise ValueError(
                "ALINE requires a finite BoxDesignSpace on task.design_space."
            )
        if space.event_shape != task.event_shape_design:
            raise ValueError("The design-space event shape must match the task.")
        if (
            not torch.all(torch.isfinite(space.low))
            or not torch.all(torch.isfinite(space.high))
            or torch.any(space.low >= space.high)
        ):
            raise ValueError("ALINE requires finite design bounds with low < high.")
        self.model = model
        self.pool_size = int(pool_size)
        self.deterministic = bool(deterministic)

    @property
    def device(self) -> torch.device:
        """Returns the ALINE model device."""
        return next(self.model.parameters()).device

    def _step_design(self, state):
        """Requires the complete fixed-pool rollout owned by this designer."""
        raise NotImplementedError("ALINE selects designs through rollout().")

    def _as_theta_batch(self, theta: Tensor) -> Tensor:
        if torch.Size(theta.shape) == self.task.event_shape_theta:
            theta = theta.unsqueeze(0)
        if torch.Size(theta.shape[1:]) != self.task.event_shape_theta:
            raise ValueError(
                "theta must have one batch dimension and the task event shape."
            )
        if theta.device != self.device:
            raise ValueError("theta and the ALINE model must be on the same device.")
        return theta

    def _mean_log_prob(self, output: AlineOutput, theta: Tensor) -> Tensor:
        # Canonicalize only the inference targets; retain the original theta
        # for simulation and likelihood-based evaluation.
        if self.model.canonicalize_theta:
            theta = sort_by_first_coordinate(theta)
        theta_flat = theta.reshape(theta.shape[0], self.model.dim_theta)
        return output.posterior.log_prob(theta_flat).mean(dim=1)

    @staticmethod
    def _random_action(available_mask: Tensor) -> Tensor:
        return torch.multinomial(available_mask.float(), 1).squeeze(1)

    @staticmethod
    def _policy_action(
        output: AlineOutput, *, deterministic: bool
    ) -> tuple[Tensor, Tensor]:
        policy = Categorical(logits=output.acquisition_logits)
        action = (
            output.acquisition_logits.argmax(dim=1)
            if deterministic
            else policy.sample()
        )
        return action, policy.log_prob(action)

    def _check_hooks(self) -> None:
        """Rejects hooks this rollout cannot run.

        TODO: follow the base class's hook protocol. Until then only the
        terminal ActionHook fires, and anything working step by step would
        attach and silently never run.
        """
        for hook in self.hooks:
            if isinstance(hook, ActionHook) and not hook.every_step:
                continue
            raise NotImplementedError(
                f"{type(self).__name__}.rollout() runs a terminal ActionHook "
                f"only; {type(hook).__name__} needs the per-step callbacks."
            )

    def rollout(
        self,
        theta: Tensor,
        *,
        random_policy: bool = False,
        deterministic: bool | None = None,
        pool_size: int | None = None,
        return_posterior_scores: bool = False,
    ) -> Rollout:
        """Simulates a fixed candidate pool, selecting without replacement.

        ``random_policy`` selects uniformly among available candidates for
        warm-up, overriding deterministic selection. Pool and policy overrides
        apply only to this call. Outcomes are detached for score-function
        training, while posterior and policy scores retain their gradients.

        ``extra["posterior_log_probs"]`` holds mean marginal log probabilities
        for D_0 through D_T, shaped [batch, T + 1], when requested.
        ``extra["policy_log_probs"]`` holds [batch, T] acquisition log
        probabilities and is absent during random-policy warm-up.

        Selection over a fixed pool does not fit the base class's per-step
        template, so only ``on_rollout_end`` hooks run here; a hook needing any
        other callback is rejected rather than silently skipped.
        """
        self._check_hooks()
        theta = self._as_theta_batch(theta)
        pool_size = self.pool_size if pool_size is None else pool_size
        if pool_size < self.T:
            raise ValueError("pool_size must be at least T.")
        deterministic = self.deterministic if deterministic is None else deterministic
        batch_size = theta.shape[0]
        candidates = self.task.design_space.sample(
            (batch_size, pool_size), device=self.device
        )
        available_mask = torch.ones(
            batch_size, pool_size, device=self.device, dtype=torch.bool
        )
        design_history = candidates.new_empty(
            batch_size, 0, *self.task.event_shape_design
        )
        y_history = candidates.new_empty(batch_size, 0, *self.task.event_shape_y)
        posterior_log_probs = []
        policy_log_probs = []
        batch_indices = torch.arange(batch_size, device=self.device)

        # Include D_0 as the first posterior training state. Evaluation needs
        # only the T acquisition forwards, so skip its unused D_T forward.
        for t in range(self.T + 1):
            output = None
            if return_posterior_scores or (t < self.T and not random_policy):
                output = self.model(
                    y_history, design_history, candidates, available_mask
                )
            if return_posterior_scores:
                posterior_log_probs.append(self._mean_log_prob(output, theta))
            if t == self.T:
                break

            if random_policy:
                action = self._random_action(available_mask)
            else:
                action, log_prob = self._policy_action(
                    output, deterministic=deterministic
                )
                policy_log_probs.append(log_prob)

            design = candidates[batch_indices, action]
            outcome = self.task.sample_outcome(
                theta, design, y_history=y_history, design_history=design_history
            ).detach()
            design_history = torch.cat([design_history, design.unsqueeze(1)], dim=1)
            y_history = torch.cat([y_history, outcome.unsqueeze(1)], dim=1)
            # Earlier logits retain the old mask for autograd.
            available_mask = available_mask.clone()
            available_mask[batch_indices, action] = False

        extra = {}
        if return_posterior_scores:
            extra["posterior_log_probs"] = torch.stack(posterior_log_probs, dim=1)
        if policy_log_probs:
            extra["policy_log_probs"] = torch.stack(policy_log_probs, dim=1)
        rollout = Rollout(theta, y_history, design_history, extra=extra)

        state = RolloutState(
            task=self.task,
            theta=theta,
            T=self.T,
            y_history=y_history,
            design_history=design_history,
            t=self.T,
        )
        apply_rollout_end(self.hooks, state, rollout)
        return rollout
