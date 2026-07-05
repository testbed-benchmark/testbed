"""Shared setup for the Hydra training entry points."""

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from testbed.core.manager import Manager


def setup(cfg):
    """Builds the run Manager and task common to every Hydra entry point.

    Hydra owns the run directory, so its resolved ``runtime.output_dir`` is
    handed straight to the Manager. The task is instantiated from ``cfg.task``
    and moved onto ``cfg.device``.

    Args:
        cfg: The composed Hydra/OmegaConf config.

    Returns:
        A ``(manager, task)`` tuple: the run manager and the device-placed task.
    """
    manager = Manager(cfg, run_dir=HydraConfig.get().runtime.output_dir)
    task = instantiate(cfg.task).to(cfg.device)
    return manager, task
