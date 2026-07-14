"""REINFORCE gradient estimator."""

import torch
from torch import Tensor

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


def _add_score_gradient(value: Tensor, score: Tensor, baseline_value):
    """REINFORCE surrogate: value plus a value-zero score-function term.

    value: [b] per-sample value, either carrying a pathwise graph or detached.
    score: [b] log density whose parameter gradient the score term injects.
    Forward value equals ``value``; the extra gradient is
    (value.detach() - baseline_value) * d(score)/dphi.
    """
    return value + (value.detach() - baseline_value) * (score - score.detach())


class Reinforce(Objective):
    """REINFORCE estimator of the gradient of an expected per-rollout value.

    value_fn maps (y_history, design_history, theta_0) -> [b] per-sample values.
    Three estimator forms, selected automatically from the policy type (read from
    the designer) and whether value_fn belongs to an expected-log-ratio objective
    (SPCE / SNMC, flagged by Objective.is_expected_log_ratio):

    (1) Stochastic policy without reparameterization - classic score function:
        d/dphi E[V] = E[ (V - b) * d/dphi sum_t log pi_phi(x_t | h_{t-1}) ].
        V is detached, so it may be any opaque (even non-differentiable) reward;
        the gradient comes from the per-step policy score.
    (2) Deterministic / reparameterized policy with an expected-log-ratio value
        (SPCE / SNMC) - the pathwise DAD Eq. 16 gradient:
        d/dphi E[V] = E[ d/dphi V + (V - b) * d/dphi log p(h_T | theta_0) ].
        The sampled log ratio is not a complete Q, so a reusable internal helper
        adds the outcome-model score.
    (3) Deterministic / reparameterized policy with any other value - pure
        pathwise (DPG): d/dphi E[V] = d/dphi mean(Q). value_fn must be a
        differentiable Q (a learned critic or analytic value); its pathwise
        gradient is taken as-is.

    Reinforce reads the policy mode from the designer that produced the rollout
    and the value kind from the objective.

    Designed for the episodic (whole-trajectory) reward setting: the value is a
    single per-rollout objective, so every action is credited with the same
    return. It uses no per-step reward or bootstrapping. Case (3) differentiates
    a user-supplied Q pathwise but does not train it; for a learned critic with
    the per-step dense-reward MDP formulation use rlboed.
    """

    def __init__(
        self,
        task: Task,
        value_fn,
        requires_policy_log_probs: bool,
        baseline: str | None = None,
    ):
        """value_fn: (y_history, design_history, theta_0) -> [b] per-sample values.

        For an expected-log-ratio objective (SPCE / SNMC), pass its
        per_sample_gain; the pathwise case is then auto-detected and the DAD
        Eq. 16 outcome score is added. For any other value_fn the pathwise case
        differentiates it as a complete Q. The score-function case treats value_fn
        as an opaque, possibly non-differentiable reward.

        baseline: "leave_one_out" or None; a control variate on the score term.
        designer: True when estimate_gain needs per-step policy log probs.
            Score-function form: the designer samples designs without
            reparameterization (and not greedily), so the gradient must come from
            the policy score rather than a pathwise graph.
        """
        super().__init__(task, requires_policy_log_probs=requires_policy_log_probs)

        if baseline not in (None, "leave_one_out"):
            raise ValueError(f"Unknown baseline: {baseline}")

        self.value_fn = value_fn
        self.baseline = baseline

    @property
    def _value_is_expected_log_ratio(self) -> bool:
        """True when value_fn belongs to an expected-log-ratio objective (SPCE / SNMC).

        Detected via the marker on value_fn.__self__, so it relies on value_fn
        being a bound method. A wrapped
        partial/lambda value_fn loses __self__ and falls back to case 3 (pure
        pathwise) - an accepted trade-off.
        """
        owner = getattr(self.value_fn, "__self__", None)
        return getattr(owner, "is_expected_log_ratio", False)

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
        r"""Builds the REINFORCE surrogate from batched histories.

        y_history:      [b, T, \*event_shape_y]
        design_history: [b, T, \*event_shape_design]
        theta_0:        [b, \*event_shape_theta] -- true theta used to generate the history
        policy_log_probs: [b, T] per-step policy log probs, required by the score-function form
        returns: scalar whose value is mean(V) and whose gradient is the estimate.
        """
        if self.requires_policy_log_probs and policy_log_probs is None:
            raise ValueError(
                "score-function estimator: rollout with return_policy_log_probs=True and pass policy_log_probs"
            )

        value = self.value_fn(y_history, design_history, theta_0)  # [b]

        # (1) Stochastic policy: score-function REINFORCE. The value is an opaque
        # (possibly non-differentiable) reward and is detached; the gradient comes
        # from the per-step policy score.
        if self.requires_policy_log_probs:
            value = value.detach()
            score = policy_log_probs.sum(-1)  # [b]
            return _add_score_gradient(value, score, self._apply_baseline(value)).mean()

        # Pathwise (deterministic or reparameterized policy): differentiate Q.
        if torch.is_grad_enabled() and not design_history.requires_grad:
            raise ValueError(
                "pathwise estimator expects designs that carry a graph, but "
                "design_history has requires_grad=False. Roll out with a "
                "deterministic or reparameterize_policy=True PolicyDesigner "
                "(and not under torch.no_grad())."
            )

        # (2) Pathwise for the Expected-log-ratio value case (SPCE / SNMC):
        # the sampled log ratio is not a complete Q, so add the DAD Eq. 16
        # outcome-model score correction.
        if self._value_is_expected_log_ratio:
            return self._expected_log_ratio_estimate(
                value, y_history, design_history, theta_0
            )

        # (3) Pure pathwise: value_fn is a user-specified differentiable Q whose
        # pathwise gradient is taken as-is.
        return value.mean()

    def _apply_baseline(self, value: Tensor):
        """Leave-one-out control variate on the score term, or 0.0 if disabled."""
        if self.baseline == "leave_one_out":
            return leave_one_out_baseline(value.detach())
        return 0.0

    def _expected_log_ratio_estimate(self, value, y_history, design_history, theta_0):
        """Pathwise estimate for an expected-log-ratio objective (SPCE / SNMC).

        The value is a per-sample log ratio evaluated on sampled outcomes, so its
        pathwise gradient alone misses the outcome-sampling term. Adding the score
        of the task outcome model p(h_T | theta_0) recovers the full DAD Eq. 16
        gradient d/dphi V + (V - b) d/dphi log p(h_T | theta_0).
        """
        score = self.task.history_logp(y_history, design_history, theta_0)  # [b]
        return _add_score_gradient(value, score, self._apply_baseline(value)).mean()
