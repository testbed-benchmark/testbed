"""Posterior inference helpers."""

from .is_sampler import effective_sample_size, run_importance_sampling

try:
    from .pyro_sampler import run_mcmc
except ModuleNotFoundError as exc:
    if exc.name != "pyro":
        raise
    run_mcmc = None
