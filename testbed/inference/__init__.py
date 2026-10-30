"""Posterior inference: any function mapping a task and a history to a Distribution."""

from .importance_sampler import run_importance_sampling
from .log_posterior import LogPosterior


def __getattr__(name):
    """Imports the Pyro engine on first use, naming the extra if Pyro is missing."""
    if name == "run_mcmc":
        try:
            from .pyro_sampler import run_mcmc
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] != "pyro":
                raise
            raise ImportError(
                "run_mcmc needs Pyro; install it with pip install 'torch-testbed[pyro]'."
            ) from exc
        return run_mcmc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
