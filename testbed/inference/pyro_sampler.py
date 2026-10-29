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
    kernel: Literal["nuts", "hmc"] = "nuts",
    disable_progbar: bool = True,
    **kernel_kwargs,
) -> EmpiricalDistribution:
    """Samples the posterior of ``task.prior`` given a history with Pyro's NUTS or HMC.

    The chain runs in the raw coordinates of :class:`LogPosterior`, so a
    constrained prior needs no special handling. It starts from a draw of
    ``task.current_prior``, which during a sequential rollout is the previous
    step's posterior.

    Args:
        task: Task with continuous parameters.
        y_history: Outcomes of one rollout, ``[t, *event_shape_y]``.
        design_history: Its designs, ``[t, *event_shape_design]``.
        n_samples: Posterior samples to keep after warmup.
        n_warmup: Warmup steps, discarded.
        kernel: ``"nuts"`` or ``"hmc"``.
        disable_progbar: Suppress Pyro's progress bar.
        **kernel_kwargs: Passed to the NUTS or HMC constructor.

    Returns:
        The samples, as an :class:`EmpiricalDistribution` with ``n_samples`` points.
    """
    if task.discrete_theta:
        raise ValueError(
            f"run_mcmc requires continuous theta; {type(task).__name__}'s prior is discrete."
        )
    kernel_cls = {"nuts": NUTS, "hmc": HMC}.get(kernel.lower())
    if kernel_cls is None:
        raise ValueError(f"kernel must be 'nuts' or 'hmc', got {kernel!r}")

    log_post = LogPosterior(task, y_history, design_history)
    z0 = log_post.to_raw(task.current_prior.sample())
    mcmc = MCMC(
        kernel_cls(potential_fn=partial(_potential_energy, log_post), **kernel_kwargs),
        num_samples=n_samples,
        warmup_steps=n_warmup,
        initial_params={"z": z0},
        disable_progbar=disable_progbar,
    )
    mcmc.run()
    return EmpiricalDistribution(log_post.to_theta(mcmc.get_samples()["z"]))
