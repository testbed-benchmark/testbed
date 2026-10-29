"""The unnormalised log posterior that inference engines target."""

import torch
from torch import Tensor
from torch.distributions import TransformedDistribution, biject_to
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

    Args:
        task: The task whose ``prior`` is conditioned on the history.
        y_history: Outcomes of one rollout, ``[t, *event_shape_y]``. Leading
            size-one dims are squeezed.
        design_history: Its designs, ``[t, *event_shape_design]``, likewise.
    """

    def __init__(self, task, y_history: Tensor, design_history: Tensor):
        """Builds the raw-coordinate log posterior."""
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
        log_lik = self.task.history_logp(self.y_history, self.design_history, theta)
        # The prior in raw coordinates, with torch adding the Jacobian term. Built
        # per call because torch can't pickle an inverse transform, and engines
        # running chains in other processes pickle this object.
        raw_prior = TransformedDistribution(
            self.task.prior, [self.transform.inv], validate_args=False
        )
        return raw_prior.log_prob(z) + log_lik
