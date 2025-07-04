from .base_task import TaskPrior, Task, EmpiricalPrior
from testbed.testbed.design_nets import DesignNet

import torch
from torch import nn, Tensor
import torch.distributions as dist

from typing import Optional, Any

############################################################################################
class LocFindPrior(TaskPrior, nn.Module):
    """
    Prior distribution for:
      theta: [B, K, p] - the location of the sources
    """

    def __init__(
        self,
        device: torch.device,
        theta_loc: Optional[Tensor] = None,      # [K, p] or None
        theta_covmat: Optional[Tensor] = None,   # [p, p] or None
        *,
        p: int = 2,
        K: int = 1,
        misspec_shift: Optional[float] = None,
    ) -> None:
        TaskPrior.__init__(self)
        nn.Module.__init__(self)

        self.device   = torch.device(device)
        self.p, self.K = p, K

        # ── mean and covariance ────────────────────────────────────────────────
        if theta_loc is None:
            theta_loc = torch.zeros((K, p), device=self.device)
        if theta_covmat is None:
            theta_covmat = torch.eye(p, device=self.device)

        self.register_buffer("theta_loc",     theta_loc)
        self.register_buffer("theta_covmat",  theta_covmat)

        shift = 0.0 if misspec_shift is None else float(misspec_shift)
        self.register_buffer("misspec_shift", torch.full((K, p), shift, device=self.device))

        # batched MVN ⇒ sample shape (*, K, p), log_prob shape (*, K)
        self._dist = dist.MultivariateNormal(
            loc=self.theta_loc + self.misspec_shift,             # [K, p]
            covariance_matrix=self.theta_covmat                  # [p, p] – broadcast to K
        )


    # ---- TaskPrior API --------------------------------------------------------
    def sample(self, sample_shape=torch.Size([])) -> Tensor:
        # returns [num_samples, K, p]
        return self._dist.sample(sample_shape=sample_shape).to(self.device)

    def log_prob(self, theta: Tensor) -> Tensor:
        # theta shape [..., K, p] → log_prob shape [..., K]
        return self._dist.log_prob(theta)




############################################################################################
# ────────────────────────────────────────────────────────────────────────────────
#  ──  Task  ─────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
class LocationFindingTask(Task, nn.Module):
    r"""
    Implements the generative model used for location source-finding
    

    p(θ)     given by ``LocFindPrior``  
    p(y | θ)  outcome likelihood method,

    where ξ is the design proposed by ``design_net`` at each step.
    """

    def __init__(
        self,
        *,
        design_net: nn.Module,
        prior:       TaskPrior,  # LocFindPrior or EmpiricalPrior
        T:           int  = 10,
        base_signal: float = 0.1,
        max_signal:  float = 1e-4,
        noise_scale: Optional[float] = None,
        empirical_posterior: Optional[TaskPrior] = None,
        data_dict:   Optional[dict[str, Any]] = None, # This may be removed depending on how runner abstract class is implemented
        # it was originally implemented to navigate StepDAD specifically
    ) -> None:
        Task.__init__(self)
        nn.Module.__init__(self)

        #Task API properties
        self._design_net   = design_net
        self._T            = int(T)
        self._design_dim            = prior.p
        self._device       = prior.device

        #Model specific properties
        self.K = prior.K  # number of sources
        self.base_signal  = float(base_signal)
        self.max_signal   = float(max_signal)

        scale             = 0.5 if noise_scale is None else float(noise_scale)
        self.register_buffer("noise_scale", torch.tensor(scale, device=self.device))

        # ── priors ----------------------------------------------------------------
        self._original_prior: TaskPrior = prior
        self._current_prior : TaskPrior = empirical_posterior or prior

        # kept for compatibility with legacy pipelines
        self.data_dict = data_dict or {}

    # ---- Task API --------------------------------------------------------------
    @property
    def design_net(self) -> nn.Module:
        return self._design_net

    @property
    def T(self) -> int:
        return self._T

    @property
    def design_dim(self) -> int:
        return self._design_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def outcome_likelihood(self, theta: Tensor, design: Tensor) -> dist.Distribution:  
        r"""Return **p(y | θ, ξ=design)** as a torch distribution."""
        # θ  : [B, K, p]
        # ξ  : [B, p]
        sq_two_norm        = (design.unsqueeze(1) - theta).pow(2).sum(-1)        # [B, K] - sum over p dim
        inv_term           = (self.max_signal + sq_two_norm).pow(-1)             # [B, K]
        mean_y             = torch.log(self.base_signal + inv_term.sum(-1, keepdim=True))# [B, 1]
        return dist.Normal(mean_y, self.noise_scale)  #note that y is a scalar hence univariate normal                           # event_shape ()


    