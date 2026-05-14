from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from ..core.util import expand_along_batch


class Task(ABC, nn.Module):
    """
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

    def __init__(self, prior, 
                 discrete_y: bool,
                 discrete_theta: bool,
                 discrete_design: bool,
                 separable: bool = True):
        
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
        """Infer device from the first registered buffer or parameter."""
        try:
            return next(self.buffers()).device
        except StopIteration:
            try:
                return next(self.parameters()).device
            except StopIteration:
                return torch.device("cpu")
            
    @abstractmethod
    def _outcome_logp(self, y, theta, design):
        """
        y: [b, *event_shape_y]
        theta: [b, *event_shape_theta]
        design: [b, *event_shape_design]
        returns: [batch, ]
        """
        ...

    @abstractmethod
    def _sample_outcome(self, theta, design, n_samples=1):
        """
        theta: [b, *event_shape_theta]
        design: [b, *event_shape_design]
        returns: [n_samples, b, **shape]
        """
        ...

    def outcome_logp(self, y, theta, design):
        """
        y, theta, design: [batch, **shape]
        Also supports when any of y, theta, design only have batch=1 by broadcasting.
        """
        y_expanded, theta_expanded, design_expanded = expand_along_batch(y, theta, design)
        return self._outcome_logp(y_expanded, theta_expanded, design_expanded)
    
    def history_logp(self, y_history, design_history, theta):
        """
        Default implementation that assumes a separable likelihood.
        Tasks with non-separable likelihoods should overwrite this.
        p(h_T | theta) = sum_t log p(y_t | theta, design_t)

        y_history: [b, T, *event_shape_y] or [1, T, *event_shape_y]
        design_history: [b, T, *event_shape_design] or [1, T, *event_shape_design]
        theta: [b, *event_shape_theta] or [1, *event_shape_theta]
        returns: [b, ]
        """
        y_history, design_history, theta = expand_along_batch(y_history, design_history, theta)       
        b = y_history.shape[0]
        T = y_history.shape[1]

        logp = torch.zeros(b, device=theta.device)
        for t in range(T):
            logp += self._outcome_logp(
                y_history[:, t],
                theta,
                design_history[:, t],
            )
        return logp
    
    def sample_outcome(self, theta, design, n_samples=1):
        """
        theta: [b, *event_shape_theta] or [1, *event_shape_theta]
        design: [b, *event_shape_design] or [1, *event_shape_design]
        returns: [n_samples, b, *event_shape_y]
        """
        theta_expanded, design_expanded = expand_along_batch(theta, design)
        return self._sample_outcome(theta_expanded, design_expanded, n_samples=n_samples)
        
    def sample_joint(self, design, n_samples=1):
        """
        Samples from p(theta, y | design) for a fixed design.
        n_samples: int
        design: [1, *event_shape_design]
        """
        theta = self.sample_current_prior(n_samples) # [n_samples, *event_shape_theta]
        outcome = self.sample_outcome(theta, design)
        outcome = outcome.squeeze(0)

        return outcome, theta
    
    def sample_marginal(self, design, n_samples=1):
        """
        Samples from p(y | design) for a fixed design.
        n_samples: int
        design: [1, *event_shape_design]
        """
        outcome, theta = self.sample_joint(design, n_samples=n_samples)
        return outcome
    
    def sample_prior(self, n_samples):
        return self.prior.sample(n_samples=n_samples) # [n_samples, *event_shape_theta]
    
    def sample_current_prior(self, n_samples):
        return self.current_prior.sample(n_samples=n_samples) # [n_samples, *event_shape_theta]
    
    def update_prior(self, design):
        # Potentially updates the prior
        # For use in sequential experiments
        raise NotImplementedError
    