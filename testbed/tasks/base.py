"""Abstract interface for BED tasks."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
import torch.nn as nn

from ..core.design_spaces import DefaultDesignSpace
from ..core.distributions import Distribution
from ..core.util import broadcast_and_flatten


class Task(ABC, nn.Module):
    r"""Abstract base class for Bayesian experimental design tasks.

    A :class:`Task` defines a probabilistic model consisting of

    - a prior over model parameters,
    - a likelihood (explicit or implicit),
    - methods for sampling observations, and
    - optionally, methods for evaluating likelihood log-probabilities.

    Concrete subclasses implement the task-specific prior and likelihood,
    while inheriting the common broadcasting and sampling interface.

    Args:
        event_shape_y (iterable):
            Shape of observations.
        event_shape_theta (iterable):
            Shape of parameters.
        event_shape_design (iterable):
            Shape of designs.
        prior (Distribution):
            Prior over model parameters.
        discrete_y (bool):
            Whether observations are discrete (default: False).
        discrete_theta (bool):
            Whether parameters are discrete (default: False).
        discrete_design (bool):
            Whether designs are discrete (default: False).
        separable (bool):
            Whether the likelihood factorizes over observations.
            Non-separable tasks must override :meth:`history_logp` (default: True).
        implicit_likelihood (bool):
            Whether the likelihood is implicit (default: False).
        design_space (optional):
            Constraints on the space of designs (default: None).
        validate_designs (bool):
            Whether outcome_logp/sample_outcome/history_logp/sample_joint check
            incoming designs against design_space.contains(...) and raise if a
            design lies outside it. Disable for tight loops where the caller
            already guarantees valid designs (default: True).

    Shape:
        Every tensor consists of

        - batch dimensions
        - event dimensions

        For example, a location-finding task with three sources in five
        spatial dimensions has

        - ``event_shape_theta = [3, 5]``
        - ``event_shape_design = [5]``
        - ``event_shape_y = [1]``
    """

    def __init__(
        self,
        event_shape_y: Iterable[int],
        event_shape_theta: Iterable[int],
        event_shape_design: Iterable[int],
        prior: Distribution,
        discrete_y: bool = False,
        discrete_theta: bool = False,
        discrete_design: bool = False,
        separable: bool = True,
        implicit_likelihood: bool = False,
        design_space=None,
        validate_designs: bool = True,
    ):
        """Initializes the task."""
        nn.Module.__init__(self)

        self.event_shape_y = torch.Size(event_shape_y)
        self.event_shape_theta = torch.Size(event_shape_theta)
        self.event_shape_design = torch.Size(event_shape_design)

        self.prior = prior
        self._current_prior = prior

        # Non-separable likelihoods must overwrite history_logp
        self.separable = separable
        if not self.separable and type(self).history_logp is Task.history_logp:
            raise TypeError(
                f"{type(self).__name__} has separable=False but does not override history_logp."
            )

        self.implicit_likelihood = implicit_likelihood

        # Tracking if there are discrete components in the model
        self.discrete_y = discrete_y
        self.discrete_theta = discrete_theta
        self.discrete_design = discrete_design

        self.design_space = (
            design_space
            if design_space is not None
            else DefaultDesignSpace(self.event_shape_design)
        )
        self.validate_designs = validate_designs

    @property
    def device(self) -> torch.device:
        """Infers device from the first registered buffer or parameter."""
        try:
            return next(self.buffers()).device
        except StopIteration:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")

    @property
    def current_prior(self) -> Distribution:
        """The current prior over parameters, possibly updated in sequential experimental design."""
        return self._current_prior

    @abstractmethod
    def _outcome_logp(self, y, theta, design):
        """Evaluates outcome log probabilities.

        The inputs (y, theta, design) are assumed to be paired,
        so that no broadcasting has to happen in this function.

        Args:
            y (Tensor):
                Observations of shape ``[b, *event_shape_y]``.
            theta (Tensor):
                Parameters of shape ``[b, *event_shape_theta]``.
            design (Tensor):
                Designs of shape ``[b, *event_shape_design]``.

        Returns:
            Tensor:
                Log probability per batch element, shape ``[b]``.
        """
        ...

    @abstractmethod
    def _sample_outcome(self, theta, design):
        """Samples a single outcome given parameters and designs.

        The inputs (theta, design) are assumed to be paired,
        so that no broadcasting has to happen in this function.

        Args:
            theta (Tensor):
                Parameters of shape ``[b, *event_shape_theta]``.
            design (Tensor):
                Designs of shape ``[b, *event_shape_design]``.

        Returns:
            Tensor:
                One sampled observation y for each (theta, design) pair, shape ``[b, *event_shape_y]``.
        """
        ...

    def _check_design(self, design):
        """Raises if design lies outside self.design_space (when validate_designs)."""
        if not self.validate_designs:
            return
        if not self.design_space.contains(design):
            raise ValueError(
                f"Design lies outside {type(self.design_space).__name__}. "
                f"Got design: {design}"
            )

    def outcome_logp(self, y, theta, design):
        """Evaluates outcome log probabilities with broadcasting.

        Broadcasting conventions follow ``torch.distributions``.

        Args:
            y (Tensor):
                Observations of shape ``[*batch_shape_y, *event_shape_y]``.
            theta (Tensor):
                Parameters of shape ``[*batch_shape_theta, *event_shape_theta]``.
            design (Tensor):
                Designs of shape ``[*batch_shape_design, *event_shape_design]``.

        Returns:
            Tensor:
                Log probabilities, shape ``[*broadcast_batch_shape]``.
        """
        self._check_design(design)
        (y_flat, theta_flat, design_flat), batch_shape = broadcast_and_flatten(
            (y, self.event_shape_y),
            (theta, self.event_shape_theta),
            (design, self.event_shape_design),
        )
        logp_flat = self._outcome_logp(y_flat, theta_flat, design_flat)
        return logp_flat.reshape(batch_shape)

    def history_logp(self, y_history, design_history, theta):
        r"""Computes log probabilities for a sequence of observations and designs.

        Assumes a separable likelihood, so that the log-probs are summed over the sequence dimension:

        .. math::
            p(y_{1:T}, x_{1:T} \mid \theta) = \sum_{t=1}^T p(y_t \mid \theta, x_t)

        Tasks with non-separable likelihoods should overwrite this.

        Args:
            y_history (Tensor):
                Observations of shape ``[*batch_shape_y, T, *event_shape_y]``.
            design_history (Tensor):
                Designs of shape ``[*batch_shape_design, T, *event_shape_design]``.
            theta (Tensor):
                Parameters of shape ``[*batch_shape_theta, *event_shape_theta]``.

        Returns:
            Tensor:
                Log probabilities, shape ``[*broadcast_batch_shape]``.
        """
        self._check_design(design_history)
        T = y_history.shape[-(len(self.event_shape_y) + 1)]
        (y_flat, design_flat, theta_flat), batch_shape = broadcast_and_flatten(
            (y_history, (T, *self.event_shape_y)),
            (design_history, (T, *self.event_shape_design)),
            (theta, tuple(self.event_shape_theta)),
        )
        b_flat = theta_flat.shape[0]
        logp = torch.zeros(b_flat, device=theta_flat.device)
        for t in range(T):
            logp += self._outcome_logp(y_flat[:, t], theta_flat, design_flat[:, t])
        return logp.reshape(batch_shape)

    def sample_outcome(self, theta, design, n_samples=1):
        """Samples outcomes from the likelihood.

        Args:
            theta (Tensor):
                Parameters of shape ``[*batch_shape_theta, *event_shape_theta]``.
            design (Tensor):
                Designs of shape ``[*batch_shape_design, *event_shape_design]``.
            n_samples (int):
                How many observation samples to draw for each (theta, design) pair (default: 1).

        Returns:
            Tensor:
                Sampled observations. If ``n_samples==1``, shape ``[*broadcast_batch_shape, *event_shape_y]``.
                Otherwise, shape ``[n_samples, *broadcast_batch_shape, *event_shape_y]``.
        """
        self._check_design(design)
        (theta_flat, design_flat), batch_shape = broadcast_and_flatten(
            (theta, self.event_shape_theta),
            (design, self.event_shape_design),
        )
        b_flat = theta_flat.shape[0]
        if n_samples > 1:
            theta_flat = (
                theta_flat.unsqueeze(0)
                .expand(n_samples, b_flat, *self.event_shape_theta)
                .reshape(n_samples * b_flat, *self.event_shape_theta)
            )
            design_flat = (
                design_flat.unsqueeze(0)
                .expand(n_samples, b_flat, *self.event_shape_design)
                .reshape(n_samples * b_flat, *self.event_shape_design)
            )
        y_flat = self._sample_outcome(theta_flat, design_flat)
        if n_samples == 1:
            return y_flat.reshape(*batch_shape, *self.event_shape_y)
        return y_flat.reshape(n_samples, *batch_shape, *self.event_shape_y)

    def sample_joint(self, design, n_samples=1):
        r"""Samples from the joint over (y, theta) given designs.

        Uses the current (possibly updated) prior.

        Args:
            design (Tensor):
                Designs of shape ``[*batch_shape, *event_shape_design]``.
            n_samples (int):
                Number of samples to draw for each design (default: 1).

        Returns:
            Tensor:
                Sampled observations. If ``n_samples==1``, shape ``[*broadcast_batch_shape, *event_shape_y]``.
                Otherwise, shape ``[n_samples, *broadcast_batch_shape, *event_shape_y]``.
            Tensor:
                Sampled parameters. If ``n_samples==1``, shape ``[*broadcast_batch_shape, *event_shape_theta]``.
                Otherwise, shape ``[n_samples, *broadcast_batch_shape, *event_shape_theta]``.
        """
        self._check_design(design)
        batch_shape = design.shape[: design.ndim - len(self.event_shape_design)]
        b_flat = 1
        for s in batch_shape:
            b_flat *= s

        total = n_samples * b_flat
        # Not sample_current_prior(total): that wrapper squeezes away the
        # leading dim when total == 1, but this code always needs a real
        # [total, *event_theta] batch dim, even when total == 1.
        theta_flat = self.current_prior.sample(n_samples=total)  # [total, *event_theta]
        design_flat = design.reshape(b_flat, *self.event_shape_design)
        if n_samples > 1:
            design_flat = (
                design_flat.unsqueeze(0)
                .expand(n_samples, b_flat, *self.event_shape_design)
                .reshape(total, *self.event_shape_design)
            )
        y_flat = self._sample_outcome(theta_flat, design_flat)  # [total, *event_y]
        if n_samples == 1:
            return (
                y_flat.reshape(*batch_shape, *self.event_shape_y),
                theta_flat.reshape(*batch_shape, *self.event_shape_theta),
            )
        return (
            y_flat.reshape(n_samples, *batch_shape, *self.event_shape_y),
            theta_flat.reshape(n_samples, *batch_shape, *self.event_shape_theta),
        )

    def sample_marginal(self, design, n_samples=1):
        r"""Samples from the marginal over observations given designs.

        Uses the current (possibly updated) prior.

        Args:
            design (Tensor):
                Designs of shape ``[*batch_shape, *event_shape_design]``.
            n_samples (int):
                Number of samples to draw for each design (default: 1).

        Returns:
            Tensor:
                Sampled observations. If ``n_samples==1``, shape ``[*broadcast_batch_shape, *event_shape_y]``.
                Otherwise, shape ``[n_samples, *broadcast_batch_shape, *event_shape_y]``.
        """
        outcome, _ = self.sample_joint(design, n_samples=n_samples)
        return outcome

    def sample_prior(self, n_samples=1):
        r"""Samples from original, non-updated prior.

        Args:
            n_samples (int):
                Number of samples to draw (default: 1).

        Returns:
            Tensor:
                Sampled parameters. If ``n_samples==1``, shape ``[*event_shape_theta]``.
                Otherwise, shape ``[n_samples, *event_shape_theta]``.
        """
        samples = self.prior.sample(
            n_samples=n_samples
        )  # [n_samples, *event_shape_theta]
        if n_samples == 1:
            return samples.squeeze(0)  # [*event_shape_theta]
        return samples

    def sample_current_prior(self, n_samples=1):
        r"""Samples from current, possibly updated prior.

        Args:
            n_samples (int):
                Number of samples to draw (default: 1).

        Returns:
            Tensor:
                Sampled parameters. If ``n_samples==1``, shape ``[*event_shape_theta]``.
                Otherwise, shape ``[n_samples, *event_shape_theta]``.
        """
        samples = self.current_prior.sample(
            n_samples=n_samples
        )  # [n_samples, *event_shape_theta]
        if n_samples == 1:
            return samples.squeeze(0)  # [*event_shape_theta]
        return samples

    def update_prior(self, new_prior):
        r"""Updates the current prior.

        Args:
            new_prior (Distribution):
                Prior to update to.
        """
        self._current_prior = new_prior
