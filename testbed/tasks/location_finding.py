from .base_task import Task
from ..core.distributions import Distribution
import torch
from torch import nn, Tensor
import torch.distributions as torch_dist
from typing import Optional, Union


class LocationFindingPrior(Distribution, nn.Module):
    """
    Prior distribution for:
      theta: [B, n_sources, dim_theta] - the location of the sources
    """

    def __init__(
        self,
        n_sources: int,
        dim_theta: int,
        theta_loc: Optional[Tensor] = None,      # [n_sources, dim_theta] or None
        theta_covmat: Optional[Tensor] = None,   # [dim_theta, dim_theta] or None
        misspec_shift: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = 'cpu'
    ) -> None:
        super().__init__()

        self.device   = torch.device(device)

        # Mean and covariance
        if theta_loc is None:
            theta_loc = torch.zeros((n_sources, dim_theta), device=self.device)
        if theta_covmat is None:
            theta_covmat = torch.eye(dim_theta, device=self.device)

        self.register_buffer("theta_loc",     theta_loc)
        self.register_buffer("theta_covmat",  theta_covmat)

        shift = 0.0 if misspec_shift is None else float(misspec_shift)
        self.register_buffer("misspec_shift", torch.full((n_sources, dim_theta), shift, device=self.device))

        # batched MVN ⇒ sample shape (*, n_sources, dim_theta), log_prob shape (*, n_sources)
        self._dist = torch_dist.MultivariateNormal(
            loc=self.theta_loc + self.misspec_shift,             # [n_sources, dim_theta]
            covariance_matrix=self.theta_covmat                  # [dim_theta, dim_theta] – broadcast to n_sources
        )

    # ---- TaskPrior API --------------------------------------------------------
    def sample(self, n_samples: int = 1) -> Tensor:
        # returns [num_samples, n_sources, dim_theta]
        return self._dist.sample((n_samples,)).to(self.device)

    def log_prob(self, theta: Tensor) -> Tensor:
        # theta shape [..., n_sources, dim_theta] → log_prob shape [..., n_sources]
        return self._dist.log_prob(theta)

class LocationFinding(Task):
    r"""
    Implements the generative model used for location source-finding
    

    p(θ)     given by ``LocFindPrior``  
    p(y | θ)  outcome likelihood method,

    where ξ is the design proposed by ``design_net`` at each step.
    """

    def __init__(
        self,
        n_sources:   int,
        dim_theta:   int,
        dim_y:       int = 1,
        base_signal: float = 0.1,
        max_signal:  float = 1e-4,
        noise_scale: float = 0.5,
        prior: Optional[Distribution] = None,
        device: Optional[Union[torch.device, str]] = 'cpu'
    ) -> None:
        
        # Default prior
        if prior is None:
            prior = LocationFindingPrior(dim_theta=dim_theta, n_sources=n_sources, device=device)
    
        super().__init__(prior, device)

        #Model specific properties
        self.n_sources    = n_sources
        self.dim_y        = dim_y
        self.dim_theta    = dim_theta
        self.base_signal  = float(base_signal)
        self.max_signal   = float(max_signal)
        self.noise_scale  = torch.tensor(noise_scale, device=self.device)

    def __repr__(self) -> str:
        return (f'LocationFinding(n_sources={self.n_sources}, '
                f'dim_theta={self.dim_theta}, '
                f'dim_y={self.dim_y}, '
                f'base_signal={self.base_signal}, '
                f'max_signal={self.max_signal}, '
                f'noise_scale={self.noise_scale}, '
                f'device={self.device})'
                )

    def make_outcome_distr(self, theta: Tensor, design: Tensor) -> torch_dist.Distribution:  
        r"""Return **p(y | θ, ξ=design)** as a torch distribution."""
        # theta:  [B, n_sources, dim_theta]
        # design: [B, dim_theta]
        sq_two_norm = (design.unsqueeze(dim=1) - theta).pow(2).sum(dim=-1)        # [B, n_sources] - sum over p dim
        inv_term = (self.max_signal + sq_two_norm).pow(-1)                        # [B, n_sources]
        mean_y = torch.log(self.base_signal + inv_term.sum(dim=-1, keepdim=True)) # [B, 1]

        # Required to handle multiple i.i.d. observations
        loc = mean_y.expand(-1, self.dim_y)
        return torch_dist.Independent(torch_dist.Normal(loc, self.noise_scale), 1)
                                        
    def _outcome_logp(self, y: Tensor, theta: Tensor, design: Tensor) -> Tensor:
        # y:       [n_samples, B, dim_y] or [B, dim_y]
        # theta:   [B, n_sources, dim_theta]
        # design:  [B, dim_theta]        
        # returns: [n_samples, B] or [B, ]
        outcome_distr = self.make_outcome_distr(theta, design)
        return outcome_distr.log_prob(y)

    def _sample_outcome(self, theta: Tensor, design: Tensor, n_samples: int=1) -> Tensor:
        # theta:   [B, n_sources, dim_theta]
        # design:  [B, dim_theta]  
        # returns: [n_samples, B, dim_y]
        outcome_distr = self.make_outcome_distr(theta, design)
        return outcome_distr.rsample((n_samples,))

