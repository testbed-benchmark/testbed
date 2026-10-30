"""Posterior inference with Pyro's MCMC."""

from functools import partial
from typing import Literal

from pyro.infer import HMC, MCMC, NUTS
from torch import Tensor

from ..core.distributions import EmpiricalDistribution
from .log_posterior import LogPosterior


def _potential_energy(log_post: LogPosterior, params: dict) -> Tensor:
    """Pyro's potential energy, the negative log posterior."""
    return -log_post(params["z"])


def run_mcmc(
    task,
    y_history: Tensor,
    design_history: Tensor,
    n_samples: int = 500,
    n_warmup: int = 500,
    num_chains: int = 1,
    kernel: Literal["nuts", "hmc"] = "nuts",
    disable_progbar: bool = True,
    **kernel_kwargs,
) -> EmpiricalDistribution:
    """Samples the posterior of ``task.prior`` given a history with Pyro's NUTS or HMC.

    The chains run in the raw coordinates of :class:`LogPosterior`, so a
    constrained prior needs no special handling. Each starts from its own draw
    of ``task.current_prior``, which during a sequential rollout is the
    previous step's posterior.

    Args:
        task: Task with continuous parameters and an explicit likelihood.
        y_history: Outcomes of one rollout, ``[t, *event_shape_y]``.
        design_history: Its designs, ``[t, *event_shape_design]``.
        n_samples: Posterior samples to keep per chain after warmup.
        n_warmup: Warmup steps per chain, discarded.
        num_chains: Independent chains; more than one runs them in parallel
            processes, as Pyro does.
        kernel: ``"nuts"`` or ``"hmc"``.
        disable_progbar: Suppress Pyro's progress bar.
        **kernel_kwargs: Passed to the NUTS or HMC constructor.

    Returns:
        The samples of every chain, as an :class:`EmpiricalDistribution` with
        ``n_samples * num_chains`` points.
    """
    kernel_cls = {"nuts": NUTS, "hmc": HMC}.get(kernel.lower())
    if kernel_cls is None:
        raise ValueError(f"kernel must be 'nuts' or 'hmc', got {kernel!r}")
    if num_chains < 1:
        raise ValueError(f"num_chains must be at least 1, got {num_chains}")

    log_post = LogPosterior(task, y_history, design_history)
    # Pyro wants one chain's parameters unbatched, and several chains' stacked.
    init_shape = () if num_chains == 1 else (num_chains,)
    z0 = log_post.to_raw(task.current_prior.sample(init_shape))
    mcmc = MCMC(
        kernel_cls(potential_fn=partial(_potential_energy, log_post), **kernel_kwargs),
        num_samples=n_samples,
        warmup_steps=n_warmup,
        num_chains=num_chains,
        initial_params={"z": z0},
        disable_progbar=disable_progbar,
    )
    mcmc.run()
    return EmpiricalDistribution(log_post.to_theta(mcmc.get_samples()["z"]))
