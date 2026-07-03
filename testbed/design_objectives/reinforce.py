"""REINFORCE (score-function) gradient estimator."""

import torch
from torch import Tensor

from ..tasks.base_task import Task
from .base_objective import Objective


def leave_one_out_baseline(reward: Tensor):
    """Per-sample mean of the other samples' rewards.

    reward: [b]
    returns: [b]
    """
    b = reward.shape[0]
    if b < 2:
        return torch.zeros_like(reward)
    return (reward.sum() - reward) / (b - 1)


class Reinforce(Objective):
    """REINFORCE estimator of the gradient of an expected per-rollout reward.

    Estimates d/dphi E[R] = E[ dR_phi/dphi + (R - b) * d/dphi log p_phi(h_T | theta_0, pi_phi) ],
    where the score-function term handles non-reparameterizable (e.g. discrete) outcomes.
    When using the per-sample SPCE reward and differentiable_reward=True, this becomes Eq. 16 of the
    DAD paper (https://arxiv.org/pdf/2103.02438).

    The returned surrogate's value is always mean(R); the flags only change its gradient:
    - differentiable_reward: include dR/dphi (requires R differentiable through the designs)
    - baseline: None (b=0) or "leave_one_out" for variance reduction
    """

    def __init__(
        self,
        task: Task,
        reward_fn,
        baseline: str | None = None,
        differentiable_reward: bool = False,
    ):
        """Initializes the estimator.

        reward_fn: (y_history, design_history, theta_0) -> [b] per-sample rewards.
        """
        super().__init__(task)

        if baseline not in (None, "leave_one_out"):
            raise ValueError(f"Unknown baseline: {baseline}")

        self.reward_fn = reward_fn
        self.baseline = baseline
        self.differentiable_reward = bool(differentiable_reward)

    def __call__(self, y_history: Tensor, design_history: Tensor, theta_0: Tensor):
        """Estimates gain for a simulated history."""
        return self.estimate_gain(y_history, design_history, theta_0)

    def estimate_gain(self, y_history: Tensor, design_history: Tensor, theta_0: Tensor):
        """Builds the REINFORCE surrogate from batched histories.

        y_history:      [b, T, *event_shape_y]
        design_history: [b, T, *event_shape_design]
        theta_0:        [b, *event_shape_theta]  -- true theta used to generate the history
        returns: scalar whose value is mean(R) and whose gradient is the REINFORCE estimate.
        """
        reward = self.reward_fn(y_history, design_history, theta_0)  # [b]

        if self.baseline == "leave_one_out":
            b = leave_one_out_baseline(reward.detach())
        else:
            b = 0.0

        # log p(h_T | theta_0, pi_phi); the policy's score, differentiable via the designs
        score = self.task.history_logp(y_history, design_history, theta_0)  # [b]

        # (score - score.detach()) has value 0 and gradient d(score)/dphi, so the score
        # term contributes only to the gradient and mean(surrogate) == mean(R).
        score_term = (reward.detach() - b) * (score - score.detach())

        if self.differentiable_reward:
            # Eq 16 of DAD paper
            surrogate = reward + score_term
        else:
            # standard RL style reinforce objective
            # first term has gradient zero but is for cacluating the reward
            # second term has value zero but contributes the gradient
            surrogate = reward.detach() + score_term

        return surrogate.mean()
