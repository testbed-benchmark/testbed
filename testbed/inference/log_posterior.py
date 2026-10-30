"""The unnormalised log posterior that inference engines target."""

import torch
from torch import Tensor
from torch.distributions import biject_to
from torch.distributions.transforms import ComposeTransform, ReshapeTransform

from ..core.util import squeeze_history


class LogPosterior:
    r"""A task's unnormalised log posterior given one history, in raw coordinates.

    Raw coordinates ``z`` are unconstrained and flat, of shape ``[*b, dim]``.
    :attr:`transform` maps them onto the prior's support: it reshapes them and
    applies ``biject_to(task.prior.support)``. Calling the object evaluates

    .. math::
        \log p(\theta) + \log \left|\det \frac{\partial \theta}{\partial z}\right|
        + \log p(h \mid \theta), \qquad \theta = \mathrm{transform}(z),

    the log posterior density of ``z`` up to a constant. It is defined on all of
    :math:`\mathbb{R}^{\mathrm{dim}}`, so an engine such as MCMC or VI can work
    with it directly, whatever constraints the prior has.

    The task needs continuous parameters, for the bijection to exist, and an
    explicit likelihood, for :meth:`~testbed.tasks.base.Task.history_logp`.

    Args:
        task: The task whose ``prior`` is conditioned on the history.
        y_history: Outcomes of one rollout, ``[t, *event_shape_y]``. Leading
            size-one dims are squeezed.
        design_history: Its designs, ``[t, *event_shape_design]``, likewise.
    """

    def __init__(self, task, y_history: Tensor, design_history: Tensor):
        """Builds the raw-coordinate log posterior."""
        name = type(task).__name__
        if task.discrete_theta:
            raise ValueError(
                f"LogPosterior needs continuous theta, but {name}'s prior is "
                "discrete; no bijection onto its support exists."
            )
        if task.implicit_likelihood:
            raise ValueError(
                f"LogPosterior needs an explicit likelihood, but {name} has an "
                "implicit one; it needs simulation-based inference instead."
            )
        self.task = task
        self.y_history = squeeze_history(y_history, task.event_shape_y)
        self.design_history = squeeze_history(design_history, task.event_shape_design)
        bijection = biject_to(task.prior.support)
        raw_shape = torch.Size(bijection.inverse_shape(task.event_shape_theta))
        self.dim = raw_shape.numel()
        self.transform = ComposeTransform(
            [ReshapeTransform(torch.Size([self.dim]), raw_shape), bijection]
        )

    def to_theta(self, z: Tensor) -> Tensor:
        """Maps raw ``[*b, dim]`` to parameters ``[*b, *event_shape_theta]``."""
        return self.transform(z)

    def to_raw(self, theta: Tensor) -> Tensor:
        """Maps parameters ``[*b, *event_shape_theta]`` to raw ``[*b, dim]``."""
        return self.transform.inv(theta)

    def __call__(self, z: Tensor) -> Tensor:
        """Evaluates the log posterior of raw ``[*b, dim]``, giving ``[*b]``."""
        theta = self.transform(z)
        batch_shape = z.shape[:-1]
        # summed over any event dims the transform leaves, to one term per z
        log_jacobian = self.transform.log_abs_det_jacobian(z, theta)
        log_jacobian = log_jacobian.reshape(*batch_shape, -1).sum(-1)
        log_prior = self.task.prior.log_prob(theta)
        log_lik = self.task.history_logp(self.y_history, self.design_history, theta)
        return log_prior + log_jacobian + log_lik
