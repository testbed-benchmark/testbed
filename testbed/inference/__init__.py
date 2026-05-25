from .is_sampler import run_importance_sampling

try:
    from .pyro_sampler import run_mcmc
except ModuleNotFoundError as exc:
    if exc.name != "pyro":
        raise
    run_mcmc = None
