import torch
from torch import Tensor

from ..core.distributions import Distribution
from ..core.util import normal_log_prob
from .base_task import Task


class LocationFindingPrior(Distribution):
    def __init__(
        self,
        n_sources,
        dim_theta,
        theta_loc=None,
        theta_covmat=None,
    ):
        super().__init__()

        if theta_loc is None:
            theta_loc = torch.zeros((n_sources, dim_theta))
        if theta_covmat is None:
            theta_covmat = torch.eye(dim_theta)

        chol = torch.linalg.cholesky(theta_covmat)
        logdet = 2 * torch.log(torch.diagonal(chol)).sum()

        # All persistent tensors registered as buffers
        self.register_buffer("loc", theta_loc)
        self.register_buffer("cov", theta_covmat)
        self.register_buffer("chol", chol)
        self.register_buffer("logdet", logdet)

        self.dim = dim_theta

    def sample(self, n_samples):
        # [n_samples, n_sources, dim_theta]
        eps = torch.randn(n_samples, *self.loc.shape, device=self.loc.device)
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
        exponent:    float = 2.0,
        prior = None,
    ) -> None:
        
        # Default prior
        if prior is None:
            prior = LocationFindingPrior(dim_theta=dim_theta, n_sources=n_sources)

        self.event_shape_theta = torch.Size([n_sources, dim_theta])
        self.event_shape_design = torch.Size([dim_theta])
        self.event_shape_y     = torch.Size([dim_y])

        self.base_signal  = base_signal
        self.max_signal   = max_signal
        self.noise_scale  = noise_scale

        # Exponent in signal decay; 2 is the standard LF model
        self.exponent = exponent

        super().__init__(prior,
                         discrete_y=False,
                         discrete_theta=False,
                         discrete_design=False,
                         )

    def __repr__(self) -> str:
        return (f'LocationFinding(n_sources={self.event_shape_theta[0]}, '
                f'dim_theta={self.event_shape_theta[1]}, '
                f'dim_y={self.event_shape_y[0]}, '
                f'base_signal={self.base_signal}, '
                f'max_signal={self.max_signal}, '
                f'noise_scale={self.noise_scale}, '
                f'exponent={self.exponent})'
                )
    
    def make_mean(self, theta, design):
        # theta:  [b, n_sources, dim_theta]
        # design: [b, dim_theta]
        sq_2_norm = (design.unsqueeze(dim=1) - theta).pow(2).sum(dim=-1)          # [B, n_sources] - sum over p dim
        signal_term = sq_2_norm.pow(self.exponent/2)
        inv_term = (self.max_signal + signal_term).pow(-1)                        # [B, n_sources]
        mean_y = torch.log(self.base_signal + inv_term.sum(dim=-1, keepdim=True)) # [B, 1]
        return mean_y
    
    def _outcome_logp(self, y, theta, design):
        # y: [b, dim_y]
        # theta: [b, n_sources, dim_theta]
        # design: [b, dim_theta]
        loc = self.make_mean(theta, design)
        logp = normal_log_prob(y, loc, self.noise_scale)
        return logp.sum(dim=-1)

    def _sample_outcome(self, theta: Tensor, design: Tensor) -> Tensor:
        # theta:   [b, n_sources, dim_theta]
        # design:  [b, dim_theta]
        # returns: [b, dim_y]
        loc = self.make_mean(theta, design)          # [b, 1]
        loc = loc.expand(-1, self.event_shape_y[0])  # [b, dim_y]
        return torch.randn_like(loc) * self.noise_scale + loc

