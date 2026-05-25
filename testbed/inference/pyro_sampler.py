from __future__ import annotations

from math import prod
from typing import Literal

import torch
import torch.distributions as td
import torch.distributions.constraints as C
import pyro
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.infer import MCMC, NUTS, HMC
from pyro.infer.autoguide.initialization import init_to_sample

from ..tasks.base_task import Task
from ..core.distributions import EmpiricalDistribution
from ..core.util import squeeze_history


class TaskPriorDist(TorchDistributionMixin, td.Distribution):
    """
    Wraps task.prior as a Pyro-compatible distribution for use with MCMC.

    batch_shape = []
    event_shape = task.event_shape_theta
    support     = independent(real, len(event_shape_theta))  — identity transform
    """

    arg_constraints: dict = {}
    has_rsample: bool = False

    def __init__(self, task: Task) -> None:
        self._task = task
        n_event = len(task.event_shape_theta)
        self._support = C.independent(C.real, n_event)
        super().__init__(
            batch_shape=torch.Size([]),
            event_shape=task.event_shape_theta,
            validate_args=False,
        )

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    @property
    def event_shape(self) -> torch.Size:
        return self._task.event_shape_theta

    @property
    def support(self) -> C.Constraint:
        return self._support

    def sample(self, sample_shape: torch.Size = torch.Size([])) -> torch.Tensor:
        n = prod(sample_shape) if len(sample_shape) > 0 else 1
        raw = self._task.prior.sample(n_samples=n)  # [n, *event_shape_theta]
        return raw.reshape(*sample_shape, *self._task.event_shape_theta)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        # theta: [*B, *event_shape_theta] — B may be empty when called by Pyro
        n_event = len(self._task.event_shape_theta)
        added = theta.ndim == n_event
        if added:
            theta = theta.unsqueeze(0)
        lp = self._task.prior.log_prob(theta)  # [*B] or [1]
        return lp.squeeze(0) if added else lp


def _make_model(
    task: Task,
    y_history: torch.Tensor,
    design_history: torch.Tensor,
) -> callable:
    prior_dist = TaskPriorDist(task)

    def model() -> None:
        theta = pyro.sample("theta", prior_dist)
        log_lik = task.history_logp(y_history, design_history, theta)
        pyro.factor("log_likelihood", log_lik)

    return model


def run_mcmc(
    task: Task,
    y_history: torch.Tensor,
    design_history: torch.Tensor,
    n_samples: int = 500,
    n_warmup: int = 500,
    num_chains: int = 1,
    kernel: Literal["nuts", "hmc"] = "nuts",
    disable_progbar: bool = True,
    **kernel_kwargs,
) -> EmpiricalDistribution:
    """
    Run MCMC to obtain posterior samples of theta given observed data.

    Parameters
    ----------
    task : Task
        Any Task subclass with continuous theta (discrete_theta must be False).
    y_history : Tensor, shape [T, *event_shape_y]
    design_history : Tensor, shape [T, *event_shape_design]
    n_samples : int
        Posterior samples to collect after warmup.
    n_warmup : int
        Warmup/burn-in steps (discarded).
    num_chains : int
        Independent MCMC chains (> 1 uses multiprocessing).
    kernel : {"nuts", "hmc"}
    disable_progbar : bool
        Suppress Pyro's progress bar.
    **kernel_kwargs
        Passed to the NUTS or HMC constructor.

    Returns
    -------
    EmpiricalDistribution
        locs has shape [n_samples * num_chains, *event_shape_theta].
    """
    if task.discrete_theta:
        raise ValueError(
            f"run_mcmc requires continuous theta; "
            f"{type(task).__name__} has discrete_theta=True."
        )

    y_history = squeeze_history(y_history, task.event_shape_y)
    design_history = squeeze_history(design_history, task.event_shape_design)

    model = _make_model(task, y_history, design_history)

    kernel_cls = {"nuts": NUTS, "hmc": HMC}.get(kernel.lower())
    if kernel_cls is None:
        raise ValueError(f"kernel must be 'nuts' or 'hmc', got {kernel!r}")

    mcmc_kernel = kernel_cls(model, init_strategy=init_to_sample, **kernel_kwargs)
    mcmc = MCMC(
        mcmc_kernel,
        num_samples=n_samples,
        warmup_steps=n_warmup,
        num_chains=num_chains,
        disable_progbar=disable_progbar,
    )
    mcmc.run()

    theta_samples = mcmc.get_samples()["theta"]  # [n_samples * num_chains, *event_shape_theta]
    return EmpiricalDistribution(locs=theta_samples)
