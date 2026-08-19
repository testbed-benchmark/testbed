"""Barber-Agakov variational-posterior objectives."""

from torch import Tensor

from ..core.rollout import Rollout
from ..core.util import sort_by_first_coordinate
from ..tasks.base import Task
from .base_objective import RolloutObjective
from .loss import _check_reduction

__all__ = ["BarberAgakov", "posterior_nll"]


def posterior_nll(
    posterior_net,
    rollout: Rollout,
    *,
    canonicalize_theta: bool = False,
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
        canonicalize_theta: Sort the leading axis of theta by first coordinate
            first. Must match the bound this posterior is paired with
            downstream, since it should be fitted to whatever theta it will
            actually be evaluated against.
        reduction: One of ``"sum"``, ``"mean"``, or ``"none"``.

    Returns:
        Tensor: Shape ``[b]`` if ``reduction`` is ``"none"``, else a scalar.
    """
    reduction = _check_reduction(reduction)
    theta = rollout.theta
    if canonicalize_theta:
        theta = sort_by_first_coordinate(theta)
    nll = -posterior_net.log_prob(theta, rollout.y_history, rollout.design_history)
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    return nll


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
        canonicalize_theta: bool = False,
    ):
        """Initializes.

        use_current_prior: subtract task.current_prior instead of task.prior
        (default: False).

        canonicalize_theta: sort the leading axis of ``event_shape_theta`` by
        first coordinate before evaluating theta under either the posterior or
        the prior (default: False), via `sort_by_first_coordinate`. Only
        meaningful for tasks whose theta is an exchangeable set of objects
        along that axis (e.g. LocationFinding's ``n_sources``, whose
        likelihood is exactly invariant to source ordering). Matches
        ScoreBED's (Phillips, Kerrigan, Rainforth, UAI 2026)
        canonicalization of LocationFinding's sources.

        Caveat: sorting by first coordinate flips discontinuously for
        configurations near a first-coordinate tie between two objects, so
        the canonicalized target theta is not a continuous function of the
        true theta everywhere. This is an inherent property of any
        canonicalization-by-sorting scheme for an exchangeable set,
        and doesn't affect the bound's validity -- only how easy the
        posterior is to fit well near those ties.
        """
        super().__init__(task)
        self.posterior_net = posterior_net
        self.use_current_prior = use_current_prior
        self.canonicalize_theta = canonicalize_theta

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
        if self.canonicalize_theta:
            theta = sort_by_first_coordinate(theta)
        log_q = self.posterior_net.log_prob(
            theta, rollout.y_history, rollout.design_history
        )
        prior = self.task.current_prior if self.use_current_prior else self.task.prior
        return log_q - prior.log_prob(theta)

    def estimate_gain(self, rollout: Rollout) -> Tensor:
        """Estimates gain from a batched rollout.

        returns: scalar Barber-Agakov estimate.
        """
        return self.per_sample_gain(rollout).mean()
