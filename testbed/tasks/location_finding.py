import torch
from torch import Tensor
from typing import Optional, Union

from .base_task import Task
from ..core.distributions import Distribution

class LocationFindingPrior(Distribution):
    def __init__(
        self,
        n_sources,
        dim_theta,
        theta_loc=None,
        theta_covmat=None,
        misspec_shift=None,
        device='cpu'
    ):
        super().__init__()

        device = torch.device(device)

        if theta_loc is None:
            theta_loc = torch.zeros((n_sources, dim_theta), device=device)

        if theta_covmat is None:
            theta_covmat = torch.eye(dim_theta, device=device)

        shift = 0.0 if misspec_shift is None else float(misspec_shift)

        self.register_buffer("loc", theta_loc + shift)
        self.register_buffer("cov", theta_covmat)

        # Precompute Cholesky + logdet
        self.register_buffer("chol", torch.linalg.cholesky(theta_covmat))
        self.register_buffer(
            "logdet",
            2 * torch.log(torch.diagonal(self.chol)).sum()
        )

        self.dim = dim_theta

    def sample(self, n_samples):
        # [n_samples, n_sources, dim_theta]
        eps = torch.randn(
            n_samples, *self.loc.shape, device=self.loc.device
        )
        return self.loc + eps @ self.chol.T

    def log_prob(self, theta):
        # theta: [..., n_sources, dim_theta]

        diff = theta - self.loc

        # solve L x = diff  (more stable than inverse)
        solve = torch.linalg.solve_triangular(
            self.chol, diff.transpose(-1, -2), upper=False
        ).transpose(-1, -2)

        maha = (solve ** 2).sum(dim=-1)  # [..., n_sources]

        return -0.5 * (
            maha + self.logdet + self.dim * torch.log(torch.tensor(2 * torch.pi))
        )

class LocationFinding(Task):
    """
    Implements the generative model used for location source-finding
    """

    def __init__(
        self,
        n_sources:   int,
        dim_theta:   int,
        dim_y:       int = 1,
        base_signal: float = 0.1,
        max_signal:  float = 1e-4,
        noise_scale: float = 0.5,
        prior = None,
        device: Optional[Union[torch.device, str]] = 'cpu'
    ) -> None:
        
        # Default prior
        if prior is None:
            prior = LocationFindingPrior(dim_theta=dim_theta, n_sources=n_sources, device=device)

        #Model specific properties
        # TODO: fix redundancy
        self.n_sources    = n_sources
        self.dim_y        = dim_y
        self.dim_theta    = dim_theta

        self.event_shape_theta = torch.Size([n_sources, dim_theta])
        self.event_shape_design = torch.Size([dim_theta])
        self.event_shape_y     = torch.Size([dim_y])

        self.base_signal  = float(base_signal)
        self.max_signal   = float(max_signal)
        self.noise_scale  = torch.tensor(noise_scale, device=device)

        super().__init__(prior)

    def __repr__(self) -> str:
        return (f'LocationFinding(n_sources={self.n_sources}, '
                f'dim_theta={self.dim_theta}, '
                f'dim_y={self.dim_y}, '
                f'base_signal={self.base_signal}, '
                f'max_signal={self.max_signal}, '
                f'noise_scale={self.noise_scale}, '
                f'device={self.device})'
                )
    
    def normal_log_prob(self, y, loc, scale):
        #print(y, loc, scale)
        var = scale ** 2
        return -0.5 * (
            ((y - loc) ** 2) / var
            + torch.log(2 * torch.pi * var)
        )
    
    def make_mean(self, theta, design):
        # theta:  [b, n_sources, dim_theta]
        # design: [b, dim_theta]
        sq_two_norm = (design.unsqueeze(dim=1) - theta).pow(2).sum(dim=-1)        # [B, n_sources] - sum over p dim
        inv_term = (self.max_signal + sq_two_norm).pow(-1)                        # [B, n_sources]
        mean_y = torch.log(self.base_signal + inv_term.sum(dim=-1, keepdim=True)) # [B, 1]
        return mean_y
    
    def _outcome_logp(self, y, theta, design):
        # y: [b, dim_y]
        # theta: [b, n_sources, dim_theta]
        # design: [b, dim_theta]
        loc = self.make_mean(theta, design)
        logp = self.normal_log_prob(y, loc, self.noise_scale)
        return logp.sum(dim=-1)

    def _sample_outcome(self, theta: Tensor, design: Tensor, n_samples: int = 1) -> Tensor:
        # theta:   [B, n_sources, dim_theta]
        # design:  [B, dim_theta]
        # returns: [n_samples, B, dim_y]
        loc = self.make_mean(theta, design)          # [B, 1]
        loc = loc.expand(-1, self.dim_y)             # [B, dim_y]
        loc = loc.unsqueeze(0).expand(n_samples, -1, -1)  # [n_samples, B, dim_y]
        return torch.randn_like(loc) * self.noise_scale + loc

