from .base_task import TaskPrior, Task
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
        data_dict:   Optional[dict[str, Any]] = None, # This may be remmoved depending on how runner abstract class is implemented
        # it was originally implemented to navigate StepDAD specifically
    ) -> None:
        Task.__init__(self)
        nn.Module.__init__(self)

        self.design_net   = design_net
        self.base_signal  = float(base_signal)
        self.max_signal   = float(max_signal)
        self.T            = int(T)

        self.p, self.K    = prior.p, prior.K
        self.device       = prior.device

        scale             = 0.5 if noise_scale is None else float(noise_scale)
        self.register_buffer("noise_scale", torch.tensor(scale, device=self.device))

        # ── priors ----------------------------------------------------------------
        self._original_prior: TaskPrior = prior
        self._current_prior : TaskPrior = empirical_posterior or prior

        # kept for compatibility with legacy pipelines
        self.data_dict = data_dict or {}

    # ---- Task API --------------------------------------------------------------
    @property
    def original_prior(self) -> TaskPrior:
        return self._original_prior

    @property
    def current_prior(self) -> TaskPrior:
        return self._current_prior

    @current_prior.setter
    def current_prior(self, value: TaskPrior) -> None:
        self._current_prior = value

    def outcome_likelihood(self, theta: Tensor, design: Tensor) -> dist.Distribution:  
        r"""Return **p(y | θ, ξ=design)** as a torch distribution."""
        # θ  : [B, K, p]
        # ξ  : [B, p]
        sq_two_norm        = (design.unsqueeze(1) - theta).pow(2).sum(-1)        # [B, K] - sum over p dim
        inv_term           = (self.max_signal + sq_two_norm).pow(-1)             # [B, K]
        mean_y             = torch.log(self.base_signal + inv_term.sum(-1, keepdim=True))# [B, 1]
        return dist.Normal(mean_y, self.noise_scale)  #note that y is a scalar hence univariate normal                           # event_shape ()


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
        designs  = torch.zeros(batch_size, 0, self.p, device=self.device) if past_designs  is None else past_designs
        outcomes = torch.zeros(batch_size, 0, 1, device=self.device) if past_outcomes is None else past_outcomes

        num_past_datapoints = designs.size(1)
        dist_sampler = lambda d: d.rsample() if use_gradients else d.sample()

        for t in range(num_past_datapoints, self.T):
            xi = self.design_net(designs, outcomes, t).to(self.device)           # [B, p]
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
        designs  = torch.zeros(B, 0, self.p, device=self.device) if past_designs  is None else past_designs
        outcomes = torch.zeros(B, 0, 1, device=self.device) if past_outcomes is None else past_outcomes

        num_past_datapoints = designs.size(1)
        for t in range(num_past_datapoints, self.T):
            xi = self.design_net(designs, outcomes, t).to(self.device)           # [B, p]
            yi = self.outcome_likelihood(theta, xi).sample()                     # [B, 1]
            designs, outcomes = (
                torch.cat([designs,  xi.unsqueeze(1)], dim=1),
                torch.cat([outcomes, yi.unsqueeze(1)], dim=1),
            )
        return designs, outcomes