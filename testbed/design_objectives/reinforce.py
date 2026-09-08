"""REINFORCE gradient estimator."""

from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
from torch import Tensor

from ..core.rollout import Rollout
from ..tasks.base import Task
from .barber_agakov import BarberAgakov
from .base_objective import RolloutObjective
from .spce import SPCE


@contextmanager
def _requires_grad(module: torch.nn.Module, flag: bool):
    """Temporarily forces every parameter of ``module`` to ``requires_grad=flag``.

    Autograd decides whether a leaf accumulates gradient based on its
    `requires_grad` at the time the *forward* ops touching it are recorded,
    not at backward time -- so restoring the original flags on exit doesn't
    retroactively reconnect a graph built with them disabled. Used to make a
    training-time flag (like `BarberAgakovReinforce.estimate_gain`'s
    `train_posterior`) the actual source of truth for whether a module gets
    gradient from one specific call, regardless of whatever `requires_grad`
    state it happens to carry externally (e.g. if the caller forgot to
    freeze it, or froze/unfroze it for an unrelated reason).
    """
    previous = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(flag)
    try:
        yield
    finally:
        for p, was in zip(module.parameters(), previous):
            p.requires_grad_(was)


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


class Reinforce(RolloutObjective, ABC):
    """REINFORCE estimator of the gradient of an expected per-rollout value.

    calculate_value maps a Rollout -> [b] per-sample values. Two estimator
    forms, selected by `stochastic_design_policy`:

    (1) Stochastic policy without reparameterization - classic score function:
        d/dphi E[V] = E[ (V - b) * d/dphi sum_t log pi_phi(x_t | h_{t-1}) ].
        V is detached, so it may be any opaque (even non-differentiable) reward;
        the gradient comes from the per-step policy score (rollout.extra["policy_log_probs"]).
    (2) Deterministic / reparameterized policy - the pathwise DAD Eq. 16
        gradient: d/dphi E[V] = E[ d/dphi V + (V - b) * d/dphi log p(h_T | theta_0) ].
        The sampled value alone is not a complete pathwise gradient, so a
        reusable internal helper adds the outcome-model score.

    Designed for the episodic (whole-trajectory) reward setting: the value is a
    single per-rollout objective, so every action is credited with the same
    return. It uses no per-step reward or bootstrapping.
    """

    def __init__(
        self,
        task: Task,
        stochastic_design_policy: bool,
        baseline: str | None = None,
    ):
        """Initializes.

        stochastic_design_policy: True when the designer samples designs
        without reparameterization (and not greedily), so the gradient must
        come from the policy score (rollout.extra["policy_log_probs"]) rather
        than a pathwise graph.

        baseline: "leave_one_out" or None; a control variate on the score term.
        """
        super().__init__(task)

        if baseline not in (None, "leave_one_out"):
            raise ValueError(f"Unknown baseline: {baseline}")

        # Both estimator forms are derived with the realized history held
        # fixed, so the outcomes must not carry a gradient to the design. If
        # they do, `value` already holds a complete pathwise gradient and the
        # score term adds a spurious Jacobian term on top of it.
        if task.reparameterize_outcomes:
            raise ValueError(
                f"{type(self).__name__} needs outcomes held fixed, but "
                f"{type(task).__name__} samples them pathwise. Either use a "
                "pathwise objective (e.g. SPCE), or construct the task with "
                "reparameterize_outcomes=False."
            )

        self.stochastic_design_policy = stochastic_design_policy
        self.baseline = baseline

    def __call__(self, rollout: Rollout, y_context=None, design_context=None):
        """Estimates gain for a simulated rollout."""
        return self.estimate_gain(
            rollout, y_context=y_context, design_context=design_context
        )

    @abstractmethod
    def calculate_value(
        self, rollout: Rollout, y_context=None, design_context=None
    ) -> Tensor:
        """Computes the [b] per-sample value the REINFORCE surrogate is built from."""
        ...

    def estimate_gain(self, rollout: Rollout, y_context=None, design_context=None):
        """Builds the REINFORCE surrogate from a batched rollout.

        A context is history already absorbed into ``task.current_prior``; both the
        value and the score term condition on it without scoring it. See
        :func:`contrastive_logsumexps`.

        returns: scalar whose value is mean(V) and whose gradient is the estimate.
        """
        value = self.calculate_value(
            rollout, y_context=y_context, design_context=design_context
        )  # [b]

        # Stochastic policy: standard REINFORCE.
        # Does not require differentiable value functions.
        if self.stochastic_design_policy:
            if "policy_log_probs" not in rollout.extra:
                raise ValueError(
                    "Using REINFORCE with a stochastic policy but "
                    "rollout.extra['policy_log_probs'] is missing."
                )

            value = value.detach()
            score = rollout.extra["policy_log_probs"].sum(-1)  # [b]

        # Deterministic policy: requires additional loss term.
        # Requires differentiable value functions.
        # See e.g. DAD Eq. 16
        else:
            if torch.is_grad_enabled() and not rollout.design_history.requires_grad:
                raise ValueError(
                    "pathwise estimator expects designs that carry a graph, but "
                    "rollout.design_history has requires_grad=False. Roll out with a "
                    "deterministic or reparameterize_policy=True PolicyDesigner "
                    "(and not under torch.no_grad())."
                )

            score = self.task.history_logp(
                rollout.y_history,
                rollout.design_history,
                rollout.theta,
                y_context=y_context,
                design_context=design_context,
            )  # [b]

        return _add_score_gradient(value, score, self._apply_baseline(value)).mean()

    def _apply_baseline(self, value: Tensor):
        """Leave-one-out control variate on the score term, or 0.0 if disabled."""
        if self.baseline == "leave_one_out":
            return leave_one_out_baseline(value.detach())
        return 0.0


class LossReinforce(Reinforce):
    """REINFORCE over a per-rollout loss computed from an action (e.g. ActionBED).

    Requires an `ActionHook` in the rollout's hook list, so that
    `rollout.extra["action"]` is populated.
    """

    def __init__(
        self,
        task: Task,
        loss_fn,
        stochastic_design_policy: bool,
        baseline: str | None = None,
    ):
        """Initializes."""
        super().__init__(task, stochastic_design_policy, baseline)

        self.loss_fn = loss_fn

    def calculate_value(
        self, rollout: Rollout, y_context=None, design_context=None
    ) -> Tensor:
        """Computes the value from the rollout's terminal action.

        The action net already sees the full history it was given, so a context
        carries no extra information here and is ignored.
        """
        return self.loss_fn(rollout.extra["action"], rollout.theta)


class SPCEReinforce(Reinforce):
    """REINFORCE over the SPCE lower bound on EIG."""

    def __init__(
        self,
        task: Task,
        n_contrastive: int,
        stochastic_design_policy: bool,
        baseline: str | None = None,
    ):
        """Initializes."""
        super().__init__(task, stochastic_design_policy, baseline)
        self.spce = SPCE(task, n_contrastive)

    def calculate_value(
        self, rollout: Rollout, y_context=None, design_context=None
    ) -> Tensor:
        """Computes the per-sample SPCE gain."""
        return self.spce.per_sample_gain(
            rollout, y_context=y_context, design_context=design_context
        )


class BarberAgakovReinforce(Reinforce):
    """REINFORCE over the Barber-Agakov lower bound on EIG."""

    def __init__(
        self,
        task: Task,
        posterior_net,
        stochastic_design_policy: bool,
        baseline: str | None = None,
        use_current_prior: bool = False,
    ):
        """Initializes."""
        super().__init__(task, stochastic_design_policy, baseline)
        self.barber_agakov = BarberAgakov(
            task,
            posterior_net,
            use_current_prior=use_current_prior,
        )

    def calculate_value(
        self, rollout: Rollout, y_context=None, design_context=None
    ) -> Tensor:
        """Computes the per-sample Barber-Agakov gain for the whole rollout.

        This estimator does not support pre-existing observation/design history.
        If context were supplied, the posterior would be conditioned on both the
        context and the rollout, while the gain would still be measured against
        the rollout's prior. That would incorrectly attribute information from
        the context to the rollout.

        Use MyopicBarberAgakov when scoring additional information conditional
        on an existing history.
        """
        # TODO This may actually work with context because posterior update hook should
        # update the task.current_prior...
        # although the default for this seems to be set to False
        # so it could just be a flag check here instead?
        if y_context is not None or design_context is not None:
            raise NotImplementedError(
                "BarberAgakovReinforce does not support a context; condition "
                "the posterior net on the history instead (see MyopicBarberAgakov)."
            )
        return self.barber_agakov.per_sample_gain(rollout)

    def posterior_gain(self, rollout: Rollout) -> Tensor:
        """The posterior's own (undetached) Barber-Agakov gain, for training posterior_net.

        Equivalent to ``self.barber_agakov.estimate_gain(rollout)``, exposed here
        as a public method rather than making callers reach into
        ``self.barber_agakov`` directly.
        """
        return self.barber_agakov.estimate_gain(rollout)

    def estimate_gain(
        self,
        rollout: Rollout,
        train_posterior: bool = False,
        y_context=None,
        design_context=None,
    ) -> Tensor:
        r"""Estimates the (policy-gradient-ready) Barber-Agakov gain.

        train_posterior: whether this call should also train posterior_net (default
        False). False guarantees zero posterior_net gradient regardless of its
        own requires_grad state -- via `_requires_grad` below, which is safe
        because autograd fixes whether a leaf gets gradient at forward time,
        so restoring requires_grad right after can't reconnect it.
        """
        with _requires_grad(self.barber_agakov.posterior_net, train_posterior):
            gain = super().estimate_gain(
                rollout, y_context=y_context, design_context=design_context
            )
        if train_posterior and self.stochastic_design_policy:
            posterior_gain = self.posterior_gain(rollout)
            gain = gain + (posterior_gain - posterior_gain.detach())
        return gain
