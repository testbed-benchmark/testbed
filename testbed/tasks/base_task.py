from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch
from torch import nn, Tensor

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

############################################################################################
class EmpiricalPrior(TaskPrior, nn.Module):
    """
    Prior distribution for:
      theta: [B, K, p] - the location of the sources
    """

    def __init__(
        self,
        device: torch.device | str,
        theta: Tensor,  # [B, K, p]
        *,
        p: int = 1,
        K: int = 2,
    ) -> None:
        TaskPrior.__init__(self)
        nn.Module.__init__(self)
        self.theta = theta.to(device)

    def sample(self, num_samples: int = 1) -> Tensor:
        """
        Sample from the empirical prior.
        
        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw, by default 1
        
        Returns
        -------
        Tensor
            Samples from the empirical prior distribution.
        """
        indices = torch.randint(0, self.theta.size(0), (num_samples,))
        return self.theta[indices]
    
    def log_prob(self, theta: Tensor) -> Tensor:
        pass

###################################################################################
class Task(ABC):
    r"""Generative model p(θ) p(y | θ, ξ)."""

    def __init__(self, prior: TaskPrior, *, empirical_posterior: TaskPrior | None = None) -> None:
        super().__init__()
        self._original_prior: TaskPrior = prior
        self._current_prior: TaskPrior = empirical_posterior or prior
 

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
        return self._current_prior
    
    @current_prior.setter
    @abstractmethod
    def current_prior(self, prior: TaskPrior) -> None: 
        """
        Set the current prior distribution p(θ) for the task.
        This may be used to update the prior after some inference has been performed.
        """
        self._current_prior = prior

    @property
    @abstractmethod
    def original_prior(self) -> TaskPrior:
        return self._original_prior


    # ---- Likelihood -------------------------------------------------------
    @property
    @abstractmethod
    def design_net(self) -> nn.Module:
        ...

    @property
    @abstractmethod
    def T(self) -> int:
        ...

    @property
    @abstractmethod
    def design_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...
    
    
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
    
    # --- Additional methods --------------------------------------------------------
    def forward(
        self,
        batch_size:   int,
        past_designs: Tensor | None = None,   # [B, t, p]
        past_outcomes:Tensor | None = None,   # [B, t, 1]
        *,
        use_gradients: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Simulate an entire T-step experiment.
        Note for greedy there may be cases where T here is 1 and then looped continually
        within the runner.

        Returns
        -------
        theta
            [B, K, p]
        new_designs
            [B, num_past_datapoints, p]  **only** the designs generated in this call
        new_outcomes
            [B, num_past_datapoints, 1]
        """
        theta  = self.current_prior.sample(batch_size)                  # [B, K, p] - #TODO confirm whether to sample a single and then repeat or sample B times as here
        designs  = torch.zeros(batch_size, 0, self.design_dim, device=self.device) if past_designs  is None else past_designs
        outcomes = torch.zeros(batch_size, 0, 1, device=self.device) if past_outcomes is None else past_outcomes

        num_past_datapoints = designs.size(1)
        dist_sampler = lambda d: d.rsample() if use_gradients else d.sample()

        for t in range(num_past_datapoints, self.T):
            xi = self.design_net(designs, outcomes).to(self.device)           # [B, p]
            yi = dist_sampler(self.outcome_likelihood(theta, xi))                # [B, 1]

            designs  = torch.cat([designs,  xi.unsqueeze(1)], dim=1)
            outcomes = torch.cat([outcomes, yi.unsqueeze(1)], dim=1)

        # only the fresh bits:
        return theta, designs[:, num_past_datapoints:], outcomes[:, num_past_datapoints:]

    @torch.no_grad()
    def run_policy(
        self,
        theta: Tensor,
        past_designs: Tensor | None = None,
        past_outcomes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Policy rollout **with θ fixed** (no gradient tracking)."""
        B = theta.size(0)
        designs  = torch.zeros(B, 0, self.design_dim, device=self.device) if past_designs  is None else past_designs
        outcomes = torch.zeros(B, 0, 1, device=self.device) if past_outcomes is None else past_outcomes

        num_past_datapoints = designs.size(1)
        for t in range(num_past_datapoints, self.T):
            xi = self.design_net(designs, outcomes).to(self.device)           # [B, p]
            yi = self.outcome_likelihood(theta, xi).sample()                     # [B, 1]
            designs, outcomes = (
                torch.cat([designs,  xi.unsqueeze(1)], dim=1),
                torch.cat([outcomes, yi.unsqueeze(1)], dim=1),
            )
        return designs, outcomes