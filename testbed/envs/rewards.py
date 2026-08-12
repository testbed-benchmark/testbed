"""Reward functions for BED environments."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .bed_env import BEDEnv


def _with_particle_axis(tensor: Tensor, event_shape: torch.Size) -> Tensor:
    """Inserts a singleton particle axis before event dimensions."""
    batch_shape = tensor.shape[: tensor.ndim - len(event_shape)]
    return tensor.reshape(*batch_shape, 1, *event_shape)


def batched_logp_against_theta_all(
    task, outcome: Tensor, design: Tensor, theta_all: Tensor
):
    """Evaluates p(outcome | theta_l, design) for all contrastive particles.

    outcome: ``[*batch_shape, *event_shape_y]``
    design: ``[*batch_shape, *event_shape_design]``
    theta_all: ``[*batch_shape, L, *event_shape_theta]``
    returns: ``[*batch_shape, L]``
    """
    outcome = _with_particle_axis(outcome, task.event_shape_y)
    design = _with_particle_axis(design, task.event_shape_design)
    return task.outcome_logp(outcome, theta_all, design)


class RewardFunction(ABC):
    """Base class for environment rewards."""

    requires_likelihood = True
    n_contrastive = 0

    def reset(self, env: BEDEnv, *, theta_all: Tensor | None = None) -> None:
        """Resets reward state."""

    @abstractmethod
    def step(self, env: BEDEnv, design: Tensor, outcome: Tensor) -> Tensor:
        """Computes reward for one transition."""
        ...


class ZeroReward(RewardFunction):
    """Returns zero reward at every step."""

    requires_likelihood = False

    def step(self, env: BEDEnv, design: Tensor, outcome: Tensor) -> Tensor:
        """Returns zeros with env batch shape."""
        return torch.zeros(env.batch_shape, device=env.device)


class CallableReward(RewardFunction):
    """Adapts a callable into a RewardFunction."""

    def __init__(
        self,
        fn: Callable[[BEDEnv, Tensor, Tensor], Tensor],
        *,
        requires_likelihood: bool = True,
    ):
        """Initializes the callable reward."""
        self.fn = fn
        self.requires_likelihood = requires_likelihood

    def step(self, env: BEDEnv, design: Tensor, outcome: Tensor) -> Tensor:
        """Delegates to the wrapped callable."""
        return self.fn(env, design, outcome)


class _ContrastiveReward(RewardFunction):
    def __init__(self, n_contrastive: int):
        """Initializes contrastive reward state."""
        self.n_contrastive = int(n_contrastive)
        if self.n_contrastive < 0:
            raise ValueError("n_contrastive must be non-negative.")
        self.log_c: Tensor | None = None
        self.theta_all: Tensor | None = None

    def reset(self, env: BEDEnv, *, theta_all: Tensor | None = None) -> None:
        """Initializes theta_all and accumulated contrastive log likelihoods.

        If *theta_all* is provided it is used directly (after shape validation).
        Otherwise theta_all is sampled from the task prior using env.theta_0 as
        the true parameter and drawing n_contrastive additional particles.
        """
        batch_shape = env.batch_shape
        event_shape = env.task.event_shape_theta
        L = self.n_contrastive + 1
        particle_dim = len(batch_shape)

        if theta_all is not None:
            expected = (*batch_shape, L, *event_shape)
            if theta_all.shape != expected:
                raise ValueError(
                    f"theta_all must have shape {expected}; got {tuple(theta_all.shape)}."
                )
            self.theta_all = theta_all.detach().to(env.device)
        elif self.n_contrastive == 0:
            self.theta_all = env.theta_0.unsqueeze(particle_dim).detach()
        else:
            n = math.prod(batch_shape) * self.n_contrastive
            theta_contrastive = (
                env.task.sample_current_prior(n)
                .reshape(*batch_shape, self.n_contrastive, *event_shape)
                .detach()
            )
            self.theta_all = torch.cat(
                [env.theta_0.unsqueeze(particle_dim), theta_contrastive],
                dim=particle_dim,
            )

        self.log_c = torch.zeros(*batch_shape, L, device=env.device)

    def update_log_c(self, env: BEDEnv, design: Tensor, outcome: Tensor) -> Tensor:
        """Accumulates log likelihoods and returns the per-step increment."""
        if self.log_c is None or self.theta_all is None:
            raise RuntimeError("Reward function has not been reset.")
        logp = batched_logp_against_theta_all(env.task, outcome, design, self.theta_all)
        self.log_c += logp
        return logp


class RLBOEDDenseReward(_ContrastiveReward):
    """Dense RL-BOED contrastive reward."""

    def __init__(self, n_contrastive: int):
        """Initializes the dense reward."""
        super().__init__(n_contrastive)
        self.last_logsumexp: Tensor | None = None

    def reset(self, env: BEDEnv, *, theta_all: Tensor | None = None) -> None:
        """Initializes running contrastive likelihood products."""
        super().reset(env, theta_all=theta_all)
        self.last_logsumexp = torch.logsumexp(self.log_c, dim=-1)

    def step(self, env: BEDEnv, design: Tensor, outcome: Tensor) -> Tensor:
        """Computes the dense RL-BOED reward."""
        if self.log_c is None or self.last_logsumexp is None:
            raise RuntimeError("Reward function has not been reset.")
        logp = self.update_log_c(env, design, outcome)
        logsumexp = torch.logsumexp(self.log_c, dim=-1)
        reward = logp[..., 0] + self.last_logsumexp - logsumexp
        self.last_logsumexp = logsumexp
        return reward


class RLBOEDTerminalReward(_ContrastiveReward):
    """Terminal SPCE-style contrastive reward."""

    def step(self, env: BEDEnv, design: Tensor, outcome: Tensor) -> Tensor:
        """Returns zero until terminal, then SPCE reward."""
        if self.log_c is None:
            raise RuntimeError("Reward function has not been reset.")
        self.update_log_c(env, design, outcome)
        if env.t < env.horizon - 1:
            return torch.zeros(env.batch_shape, device=env.device)
        log_L = math.log(self.n_contrastive + 1)
        return self.log_c[..., 0] - torch.logsumexp(self.log_c, dim=-1) + log_L
