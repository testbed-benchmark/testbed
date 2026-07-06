"""REINFORCE gradient estimator."""

import torch
from torch import Tensor

from ..design_nets.stochastic import StochasticDesignNet
from ..tasks.base import Task
from .base_objective import Objective


def leave_one_out_baseline(reward: Tensor):
    """Per-sample mean of the other samples' rewards.

    reward: [b]
    returns: [b]
    """
    batch_size = reward.shape[0]
    if batch_size < 2:
        return torch.zeros_like(reward)
    return (reward.sum() - reward) / (batch_size - 1)


class Reinforce(Objective):
    """REINFORCE estimator of the gradient of an expected per-rollout value.

    The form of the estimator depends on the policy type:

    Stochastic policy, score function (classic REINFORCE):
        d/dphi E[V] ~= E[ (V - b) * d/dphi sum_t log pi_phi(x_t | h_{t-1}) ],  V detached.
    Stochastic policy with reparameterize_policy=True, or deterministic policy (DDPG-style):
        d/dphi E[V] = d/dphi mean(value_fn(...)), pathwise through the designs.

    value_fn plays the role of Q: a critic backprops through the network, while
    SPCE.reinforce_value supplies the DAD Eq. 16 gradient in the pathwise case.

    Pathwise estimation requires designs that carry the graph, so the rollout must
    use the matching PolicyDesigner reparameterize_policy setting.
    """

    def __init__(
        self,
        task: Task,
        value_fn,
        policy_net,
        baseline: str | None = None,
        reparameterize_policy: bool = False,
    ):
        """value_fn: (y_history, design_history, theta_0) -> [b] per-sample values."""
        super().__init__(task)

        if baseline not in (None, "leave_one_out"):
            raise ValueError(f"Unknown baseline: {baseline}")
        self.stochastic = isinstance(policy_net, StochasticDesignNet)
        self.reparameterize_policy = bool(reparameterize_policy)
        if self.reparameterize_policy and not self.stochastic:
            raise ValueError("reparameterize_policy only applies to stochastic policies.")
        if baseline is not None and not self.requires_policy_log_probs:
            raise ValueError(
                "baseline only applies to the score-function estimator; for pathwise "
                "estimation pass it to the value_fn (e.g. SPCE.reinforce_value)."
            )

        self.value_fn = value_fn
        self.baseline = baseline

    @property
    def requires_policy_log_probs(self) -> bool:
        """True when estimate_gain needs per-step policy log probs."""
        return self.stochastic and not self.reparameterize_policy

    def __call__(
        self,
        y_history: Tensor,
        design_history: Tensor,
        theta_0: Tensor,
        policy_log_probs: Tensor | None = None,
    ):
        """Estimates gain for a simulated history."""
        return self.estimate_gain(y_history, design_history, theta_0, policy_log_probs)

    def estimate_gain(
        self,
        y_history: Tensor,
        design_history: Tensor,
        theta_0: Tensor,
        policy_log_probs: Tensor | None = None,
    ):
        """Builds the REINFORCE surrogate from batched histories.

        y_history:      [b, T, *event_shape_y]
        design_history: [b, T, *event_shape_design]
        theta_0:        [b, *event_shape_theta]  -- true theta used to generate the history
        policy_log_probs: [b, T] per-step policy log probs, required by the score-function form
        returns: scalar whose value is mean(V) and whose gradient is the estimate.
        """
        if not self.requires_policy_log_probs:
            return self.value_fn(y_history, design_history, theta_0).mean()

        if policy_log_probs is None:
            raise ValueError(
                "score-function estimator: rollout with return_policy_log_probs=True and pass policy_log_probs"
            )

        value = self.value_fn(y_history, design_history, theta_0).detach()  # [b]
        if self.baseline == "leave_one_out":
            baseline_value = leave_one_out_baseline(value)
        else:
            baseline_value = 0.0

        score = policy_log_probs.sum(-1)  # [b]
        # (score - score.detach()) has value 0 and gradient d(score)/dphi
        surrogate = value + (value - baseline_value) * (score - score.detach())
        return surrogate.mean()
