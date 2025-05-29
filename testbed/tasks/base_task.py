from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch

###################################################################################
class _DistributionLike(Protocol):
    """
    Minimal torch-style distribution interface used by the Task ABC.
    
    Written using a Protocol to allow for flexibility in the types of distributions
    that can be used, while still ensuring they implement the necessary methods.
    
    """
    def sample(self, sample_shape: torch.Size | tuple = torch.Size(),
               generator: torch.Generator | None = None) -> Any: ...
    
    def log_prob(self, value: Any) -> torch.Tensor: ...


###################################################################################
class TaskPrior(ABC):
    """Abstract prior (or posterior) p(θ)."""

    @abstractmethod
    def sample(self, num_samples: int = 1) -> Any: ...

    @abstractmethod
    def log_prob(self, theta: Any) -> torch.Tensor: ...


###################################################################################
class Task(ABC):
    r"""Generative model p(θ) p(y | θ, ξ)."""

 

    # ---- Priors -----------------------------------------------------------
    @property
    @abstractmethod
    def current_prior(self) -> TaskPrior: 
        """
        Return the current prior distribution p(θ) for the task. In some cases,
        this may be the same as the original prior, but in other cases it may
        be a modified version of the original prior, such as a posterior
        distribution after some inference has been performed.
        """
        ...
    @current_prior.setter
    @abstractmethod
    def current_prior(self, prior: TaskPrior) -> None: 
        """
        Set the current prior distribution p(θ) for the task.
        This may be used to update the prior after some inference has been performed.
        """
        ...

    @property
    @abstractmethod
    def original_prior(self) -> TaskPrior: ...


    # ---- Likelihood -------------------------------------------------------
    @abstractmethod
    def outcome_likelihood(self, theta: Any, design: Any) -> _DistributionLike: 
        """
        Return a *distribution object* representing
        p(y | θ, ξ=design).  Must implement `.sample()` and `.log_prob()`.

        For likelihood free models, this becomes we then just do not
        implment the log_prob method, and instead solely use the
        `outcome_likelihood_sample` method to sample from the
        likelihood distribution.

        """
        ...