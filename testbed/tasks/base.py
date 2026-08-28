"""Abstract interface for BED tasks."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
from torch import nn

from ..core.design_spaces import DefaultDesignSpace, DesignSpace
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
        separable (bool):
            Whether the likelihood factorizes over observations.
            Non-separable tasks must override :meth:`history_logp` (default: True).
        implicit_likelihood (bool):
            Whether the likelihood is implicit (default: False).
        design_space (optional):
            Constraints on the space of designs. Its event shape must match
            ``event_shape_design`` (default: None).
        validate_designs (bool):
            Whether outcome_logp/sample_outcome/history_logp/sample_joint check
            incoming designs against design_space.contains(...) and raise if a
            design lies outside it. Disable for tight loops where the caller
            already guarantees valid designs (default: True).
        reparameterize_outcomes (bool, optional):
            Whether sampled outcomes carry a pathwise gradient to the design.
            None follows the subclass's :attr:`reparameterizable_outcomes`;
            True raises if the subclass cannot honour it. Set False for
            score-function estimators, which are derived with the history held
            fixed (default: None).

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

    @property
    @abstractmethod
    def reparameterizable_outcomes(self) -> bool:
        """Whether :meth:`_sample_outcome` can sample with a pathwise gradient.

        A fact about the model, not a preference: False for genuinely discrete
        outcomes. Pathwise estimators require it; score-function estimators
        require its absence, so every task must state it explicitly.
        """
        ...

    @property
    def reparameterize_outcomes(self) -> bool:
        """Whether :meth:`sample_outcome` currently returns attached outcomes.

        Resolved against :attr:`reparameterizable_outcomes` in ``__init__``,
        which rejects a pathwise request the task cannot honour.
        """
        return self._reparameterize_outcomes

    def __init__(
        self,
        event_shape_y: Iterable[int],
        event_shape_theta: Iterable[int],
        event_shape_design: Iterable[int],
        prior: Distribution,
        discrete_y: bool = False,
        discrete_theta: bool = False,
        separable: bool = True,
        implicit_likelihood: bool = False,
        design_space: DesignSpace | None = None,
        validate_designs: bool = True,
        reparameterize_outcomes: bool | None = None,
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

        resolved_design_space = (
            design_space
            if design_space is not None
            else DefaultDesignSpace(self.event_shape_design)
        )
        if resolved_design_space.event_shape != self.event_shape_design:
            raise ValueError(
                "design_space.event_shape must match event_shape_design; "
                f"expected {tuple(self.event_shape_design)}, got "
                f"{tuple(resolved_design_space.event_shape)}."
            )
        self.design_space = resolved_design_space
        self.validate_designs = validate_designs

        # None follows the task's own capability. An explicit True the task
        # cannot honour is an error: a
        # pathwise estimator would otherwise run on a task that never
        # supplies the gradient.
        if reparameterize_outcomes and not self.reparameterizable_outcomes:
            raise ValueError(
                f"{type(self).__name__} was constructed with "
                "reparameterize_outcomes=True, but its outcomes are not "
                "reparameterizable (see its reparameterizable_outcomes "
                "docstring for why). Pass False, or None to follow the task's "
                "own capability, and train with a score-function estimator "
                "such as REINFORCE."
            )
        if reparameterize_outcomes is None:
            reparameterize_outcomes = self.reparameterizable_outcomes
        self._reparameterize_outcomes = bool(reparameterize_outcomes)

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

        Always returns the reparameterized draw where the model admits one, so
        implementations need not consult ``reparameterize_outcomes``. Gating
        whether those gradients are kept is :meth:`_gate_outcome_gradients`'s
        job. Sample through :meth:`sample_outcome` or :meth:`sample_joint`
        rather than calling this internal method directly, or the gate is skipped.

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

    def _gate_outcome_gradients(self, y_flat, design_flat):
        """Detaches or validates a raw ``_sample_outcome`` draw.

        Every public sampler routes its draw through here so the
        reparameterization contract holds wherever outcomes are produced.

        Args:
            y_flat (Tensor):
                Outcomes of shape ``[b, *event_shape_y]``.
            design_flat (Tensor):
                The designs they were sampled from, shape ``[b, *event_shape_design]``.

        Returns:
            Tensor:
                The outcomes, detached when ``reparameterize_outcomes`` is False.
        """
        if not self.reparameterize_outcomes:
            return y_flat.detach()
        if (
            torch.is_grad_enabled()
            and design_flat.requires_grad
            and not y_flat.requires_grad
        ):
            # The design carried a graph but the outcome did not, so
            # _sample_outcome sampled without reparameterizing. Pathwise
            # estimators would silently lose the distributional term.
            raise ValueError(
                f"{type(self).__name__}.reparameterize_outcomes is True but "
                "_sample_outcome returned an outcome with no gradient to the "
                "design. Sample reparameterized (e.g. loc + scale * randn_like, "
                "or rsample), or declare reparameterizable_outcomes = False."
            )
        return y_flat

    def _check_design(self, design):
        """Raises if design lies outside self.design_space (when validate_designs)."""
        if not self.validate_designs:
            return
        if not self.design_space.contains(design):
            # contains() rejects both bad values and a bad event shape
            design = torch.as_tensor(design)
            raise ValueError(
                f"Design lies outside {self.design_space} "
                f"(expected trailing shape "
                f"{tuple(self.design_space.event_shape)}). "
                f"Got design of shape {tuple(design.shape)}: {design}"
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

        Detaches the sample when ``reparameterize_outcomes`` is False, so the
        outcome is graph-free before it reaches a designer. Detaching here
        rather than downstream also stops later designs acquiring a gradient
        path through it via the policy's history. Always sample through this
        wrapper, never via ``_sample_outcome`` directly.

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
        y_flat = self._gate_outcome_gradients(
            self._sample_outcome(theta_flat, design_flat), design_flat
        )
        if n_samples == 1:
            return y_flat.reshape(*batch_shape, *self.event_shape_y)
        return y_flat.reshape(n_samples, *batch_shape, *self.event_shape_y)

    def sample_joint(self, design, n_samples=1):
        r"""Samples from the joint over (y, theta) given designs.

        Uses the current (possibly updated) prior. Gates the outcome's
        gradients exactly as :meth:`sample_outcome` does, so the outcome is
        graph-free when ``reparameterize_outcomes`` is False.

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
        y_flat = self._gate_outcome_gradients(
            self._sample_outcome(theta_flat, design_flat), design_flat
        )  # [total, *event_y]
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
