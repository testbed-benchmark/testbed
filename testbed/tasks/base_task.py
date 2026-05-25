"""Base task interface."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ..core.util import broadcast_and_flatten


class Task(ABC, nn.Module):
    """Base class for Bayesian experimental-design tasks.

    Conventions:
    - Generally, we will follow the torch.distributions conventions for shapes
    - Everything (y, theta, design) will have two relevant shapes:
    - An event_shape
    - A batch_shape
    - For example, in location finding in 5D with 3 sources:
        - y event_shape: [1]
        - theta event_shape: [3, 5]
        - design event_shape: [5]
    - The batch_shape will vary based on context.
    """

    event_shape_theta: torch.Size
    event_shape_design: torch.Size
    event_shape_y: torch.Size

    def __init__(
        self,
        prior,
        discrete_y: bool,
        discrete_theta: bool,
        discrete_design: bool,
        separable: bool = True,
    ):
        """Initializes a task with its prior and discreteness flags."""
        # Safety checking that some required attributes are defined in subclasses
        required = ["event_shape_theta", "event_shape_design", "event_shape_y"]
        missing = [attr for attr in required if not hasattr(self, attr)]
        if missing:
            raise TypeError(f"{type(self).__name__} must define: {missing}")

        nn.Module.__init__(self)

        self.prior = prior
        self.current_prior = prior

        # Is the likelihood separable?
        # If so, subclass must overwrite history_logp
        self.separable = separable
        if not self.separable and type(self).history_logp is Task.history_logp:
            raise TypeError(
                f"{type(self).__name__} has separable=False but does not override history_logp."
            )

        # Tracking if there are discrete components in the model
        self.discrete_y = discrete_y
        self.discrete_theta = discrete_theta
        self.discrete_design = discrete_design

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

    @abstractmethod
    def _outcome_logp(self, y, theta, design):
        """Evaluates per-sample outcome log probabilities.

        y: [b, *event_shape_y]
        theta: [b, *event_shape_theta]
        design: [b, *event_shape_design]
        returns: [b,].
        """
        ...

    @abstractmethod
    def _sample_outcome(self, theta, design):
        """Samples one outcome per theta-design pair.

        theta: [b, *event_shape_theta]
        design: [b, *event_shape_design]
        returns: [b, *event_shape_y]  — one sample per (theta, design) pair.
        """
        ...

    def outcome_logp(self, y, theta, design):
        """Evaluates outcome log probabilities with broadcasting.

        y: [*batch_shape_y,  *event_shape_y]
        theta: [*batch_shape_theta, *event_shape_theta]
        design: [*batch_shape_design, *event_shape_design]
        Batch dims are broadcast together (torch.distributions convention).
        returns: [*batch_shape].
        """
        (y_flat, theta_flat, design_flat), batch_shape = broadcast_and_flatten(
            (y, self.event_shape_y),
            (theta, self.event_shape_theta),
            (design, self.event_shape_design),
        )
        logp_flat = self._outcome_logp(y_flat, theta_flat, design_flat)
        return logp_flat.reshape(batch_shape)

    def history_logp(self, y_history, design_history, theta):
        """Computes history log probabilities for a separable likelihood.

        Tasks with non-separable likelihoods should overwrite this.
        p(h_T | theta) = sum_t log p(y_t | theta, design_t).

        y_history: [*batch_shape, T, *event_shape_y]
        design_history: [*batch_shape, T, *event_shape_design]
        theta: [*batch_shape, *event_shape_theta]
        Batch dims are broadcast together across all three arguments.
        returns: [*batch_shape]
        """
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
        """Samples outcomes with broadcast batch shapes.

        theta: [*batch_shape_theta, *event_shape_theta]
        design: [*batch_shape_design, *event_shape_design]
        Batch dims are broadcast together (torch.distributions convention).

        Returns:
            n_samples == 1: [*batch_shape, *event_shape_y]
            n_samples  > 1: [n_samples, *batch_shape, *event_shape_y]
        """
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
        """design: [*batch_shape, *event_shape_design].

        Returns:
            n_samples == 1: (y [*batch_shape, *event_y], theta [*batch_shape, *event_theta])
            n_samples  > 1: (y [n_samples, *batch_shape, *event_y],
                             theta [n_samples, *batch_shape, *event_theta])
        """
        batch_shape = design.shape[: design.ndim - len(self.event_shape_design)]
        b_flat = 1
        for s in batch_shape:
            b_flat *= s

        total = n_samples * b_flat
        theta_flat = self.sample_current_prior(total)  # [total, *event_theta]
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
        """Samples outcomes from the marginal predictive distribution.

        design: [*batch_shape, *event_shape_design]
        returns: y with same shape convention as sample_joint.
        """
        outcome, _ = self.sample_joint(design, n_samples=n_samples)
        return outcome

    def sample_prior(self, n_samples=1):
        """Samples from the original prior."""
        samples = self.prior.sample(
            n_samples=n_samples
        )  # [n_samples, *event_shape_theta]
        if n_samples == 1:
            return samples.squeeze(0)  # [*event_shape_theta]
        return samples

    def sample_current_prior(self, n_samples=1):
        """Samples from the current prior."""
        samples = self.current_prior.sample(
            n_samples=n_samples
        )  # [n_samples, *event_shape_theta]
        if n_samples == 1:
            return samples.squeeze(0)  # [*event_shape_theta]
        return samples

    def update_prior(self, new_prior):
        """Replaces the current prior."""
        self.current_prior = new_prior
