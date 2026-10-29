"""Shared setup for the Hydra training entry points."""

import inspect
import re
import sys
from math import prod
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_object, instantiate
from omegaconf import OmegaConf

from testbed.core.manager import Manager


def instantiate_design_net(config, task):
    """Instantiates a design net with the task arguments its target declares.

    Stochastic policy factories can opt into the task-owned design prior,
    while deterministic policies continue to receive only their shape inputs.
    """
    target = get_object(config._target_)
    parameters = inspect.signature(target).parameters
    task_arguments = {
        "event_shape_y": task.event_shape_y,
        "event_shape_theta": task.event_shape_theta,
        "event_shape_design": task.event_shape_design,
        "event_shape_out": task.event_shape_design,
        "design_prior": task.design_prior,
    }
    return instantiate(
        config,
        **{name: value for name, value in task_arguments.items() if name in parameters},
    )


def instantiate_action_net(config, task, T):
    """Instantiates an action net, which outputs an estimate of theta.

    A net configured with ``dims`` (an MLP) flattens the whole history, so its
    first and last widths follow from the task and T rather than the config,
    which supplies the hidden widths. Resolving them here rather than in the
    recipe is what lets one recipe serve whichever task it is pointed at --
    and, as with ``instantiate_design_net``, keeps shapes out of the config.
    """
    if "dims" in config:
        dim_in = T * (prod(task.event_shape_y) + prod(task.event_shape_design))
        config = OmegaConf.merge(
            config,
            {"dims": [dim_in, *config.dims[1:-1], prod(task.event_shape_theta)]},
        )

    return instantiate(
        config,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_theta,
    )


def setup(cfg):
    """Build the run manager and task common to every Hydra entry point.

    Under the Hydra CLI, Hydra owns the run directory, so its resolved
    ``runtime.output_dir`` is handed straight to the Manager. Called directly
    (from a script or notebook), the Manager creates a timestamped run under
    ``cfg.save_dir``. The task is instantiated from ``cfg.task`` and moved onto
    ``cfg.device``.

    Args:
        cfg: The composed Hydra/OmegaConf config.

    Returns:
        A ``(manager, task)`` tuple: the run manager and the device-placed task.
    """
    if HydraConfig.initialized():
        manager = Manager(cfg, run_dir=HydraConfig.get().runtime.output_dir)
    else:
        manager = Manager(cfg, root=cfg.get("save_dir", "runs"))
    task = instantiate(cfg.task).to(cfg.device)
    return manager, task


_RECIPES = Path(__file__).resolve().parent.parent / "recipes"


def _check_hydra_cli():
    """Fails early where the installed Hydra cannot run its command line."""
    match = re.match(r"(\d+)\.(\d+)", hydra.__version__)
    if (
        sys.version_info >= (3, 14)
        and match
        and (int(match[1]), int(match[2])) < (1, 4)
    ):
        raise RuntimeError(
            "hydra-core<1.4 cannot run its command line on Python 3.14 (argparse "
            "rejects its help strings). Launch training with Python 3.13 or earlier."
        )


def make_main(run, config_name):
    """Wraps a method's ``run(cfg)`` as a Hydra command-line entry point.

    The recipes path is absolute: a relative one would resolve against
    ``run``'s module as a ``pkg://`` path, and ``testbed/recipes`` is not a
    package. The Python 3.14 check runs only when ``main`` is called, so the
    method modules still import there.
    """
    hydra_main = hydra.main(
        version_base="1.3", config_path=str(_RECIPES), config_name=config_name
    )(run)

    def main():
        _check_hydra_cli()
        hydra_main()

    return main
