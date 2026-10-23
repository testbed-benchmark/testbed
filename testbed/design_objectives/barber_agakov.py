"""Barber-Agakov variational-posterior objectives."""

import torch
from torch import Tensor

from ..core.rollout import Rollout
from ..tasks.base import Task
from .base_objective import DesignObjective, RolloutObjective
from .loss import _check_reduction


def _prepend_history(context, step):
    """Prepends the history so far to this step's ``[b, 1, *event_shape]`` entry.

    An absent context is a ``T=0`` history, so the step passes through unchanged.
    """
    if context is None:
        return step
    return torch.cat([context.expand(step.shape[0], *context.shape[1:]), step], dim=1)


def posterior_nll(
    posterior_net,
    rollout: Rollout,
    *,
    reduction: str = "mean",
) -> Tensor:
    r"""Negative log-likelihood of a rollout's theta under a variational posterior.

    Used to pretrain or fit a posterior net. Minimizing it over the posterior's
    parameters is equivalent to maximizing any EIG lower bound that depends on
    the posterior only through ``log q(theta | y, d)`` (e.g. :class:`BarberAgakov`):
    whatever posterior-independent term such a bound also carries (``-log
    p(theta)`` for Barber-Agakov) has zero gradient w.r.t. posterior parameters.

    Args:
        posterior_net: Anything exposing ``log_prob(theta, y_history,
            design_history)``, e.g. a ``VariationalPosteriorNet``.
        rollout: Rollout supplying theta and the histories to condition on.
        reduction: One of ``"sum"``, ``"mean"``, or ``"none"``.

    Returns:
        Tensor: Shape ``[b]`` if ``reduction`` is ``"none"``, else a scalar.
    """
    reduction = _check_reduction(reduction)
    nll = -posterior_net.log_prob(
        rollout.theta, rollout.y_history, rollout.design_history
    )
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    return nll


class MyopicBarberAgakov(DesignObjective):
    r"""Barber-Agakov lower bound on the EIG of one design.

    .. math::
        \mathrm{EIG}(d \mid h) \ge
        \mathbb{E}[\log q(\theta \mid h, d, y) - \log p(\theta \mid h)]

    The ``T=1`` case of :class:`BarberAgakov`, for myopic optimization with
    ``SequentialDesigner``: draws ``n_outer`` pairs ``(theta, y)`` for ``design``
    and evaluates the bound with ``posterior_net`` conditioned on the history
    so far plus that step.

    The subtracted term is :attr:`Task.current_prior`, i.e. ``p(theta | h)``
    once ``PosteriorUpdateHook`` has conditioned it on the history. ``q`` must
    therefore see the whole history and not just the new step, or the two sides
    of the bound condition on different things -- and the net is trained on
    full histories anyway (see :func:`posterior_nll`).

    Args:
        task: Task whose EIG is bounded.
        posterior_net: Anything exposing ``log_prob(theta, y_history,
            design_history)``, e.g. a ``VariationalPosteriorNet``.
        n_outer: Number of ``(theta, y)`` pairs per gain estimate.
    """

    def __init__(self, task: Task, posterior_net, n_outer: int):
        """Initializes."""
        super().__init__(task)
        self.posterior_net = posterior_net
        self.n_outer = int(n_outer)

    def __call__(
        self, design: Tensor, y_history: Tensor = None, design_history: Tensor = None
    ):
        """Estimates gain for one design."""
        return self.estimate_gain(
            design, y_history=y_history, design_history=design_history
        )

    def estimate_gain(
        self, design: Tensor, y_history: Tensor = None, design_history: Tensor = None
    ) -> Tensor:
        """Estimates gain for one fixed design.

        design: ``[*event_shape_design]``. Leading size-1 batch dims are
        tolerated and stripped.
        returns: scalar Barber-Agakov estimate.
        """
        n_event_dims = len(self.task.event_shape_design)
        batch_shape = design.shape[: design.ndim - n_event_dims]
        if any(s != 1 for s in batch_shape):
            raise ValueError(
                "MyopicBarberAgakov expects a single design; got shape "
                f"{tuple(design.shape)}"
            )
        design = design.reshape(self.task.event_shape_design)

        b = self.n_outer
        y, theta = self.task.sample_joint(
            design, y_history=y_history, design_history=design_history, n_samples=b
        )
        # A context history contributes a batch dim of its own, so reshape
        # explicitly rather than trusting the returned shape.
        y = y.reshape(b, 1, *self.task.event_shape_y)
        theta = theta.reshape(b, *self.task.event_shape_theta)
        design_step = design.expand(b, 1, *self.task.event_shape_design)

        log_q = self.posterior_net.log_prob(
            theta,
            _prepend_history(y_history, y),
            _prepend_history(design_history, design_step),
        )
        return (log_q - self.task.current_prior.log_prob(theta)).mean()


class BarberAgakov(RolloutObjective):
    r"""Barber-Agakov lower bound on EIG via an amortized variational posterior.

    .. math::
        \mathrm{EIG}(d) \ge \mathbb{E}[\log q(\theta \mid y, d) - \log p(\theta)]

    The bound is tight iff ``q == p(theta | y, d)``.
    """

    def __init__(
        self,
        task: Task,
        posterior_net,
        use_current_prior: bool = False,
    ):
        """Initializes.

        use_current_prior: subtract task.current_prior instead of task.prior
        (default: False).
        """
        super().__init__(task)
        self.posterior_net = posterior_net
        self.use_current_prior = use_current_prior

        try:
            task.prior.log_prob(task.sample_prior(2))
        except Exception as e:
            raise ValueError(
                f"{type(task).__name__}.prior.log_prob is not evaluable; BarberAgakov "
                "requires a prior with a working log_prob."
            ) from e

    def __call__(self, rollout: Rollout):
        """Estimates gain for a simulated rollout."""
        return self.estimate_gain(rollout)

    def per_sample_gain(self, rollout: Rollout) -> Tensor:
        """Estimates per-sample gains from a batched rollout.

        returns: [b] per-sample Barber-Agakov estimates.
        """
        theta = rollout.theta
        log_q = self.posterior_net.log_prob(
            theta, rollout.y_history, rollout.design_history
        )
        prior = self.task.current_prior if self.use_current_prior else self.task.prior
        # This subtraction is only valid if log_q and prior.log_prob(theta) are
        # densities with respect to the same underlying measure. Otherwise it is not
        # log(q(theta) / p(theta)), and the resulting objective is not a valid bound.
        #
        # This matters for constrained theta. For example, in CES theta is stored in
        # R^5 but has only 4 free degrees of freedom because alpha lies on a simplex.
        # An ordinary Gaussian mixture over R^5 therefore defines density per unit 5D
        # volume, while the prior defines density on the 4D constrained space.
        #
        # ConstrainedGaussianMixtureVariationalPosteriorNet avoids this mismatch by
        # defining q in unconstrained R^4 coordinates and pushing it forward onto the
        # same constrained space as the prior. Thus no correction is needed here.
        return log_q - prior.log_prob(theta)

    def estimate_gain(self, rollout: Rollout) -> Tensor:
        """Estimates gain from a batched rollout.

        returns: scalar Barber-Agakov estimate.
        """
        return self.per_sample_gain(rollout).mean()
