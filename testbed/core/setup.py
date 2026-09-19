"""Shared setup for the Hydra training entry points."""

import inspect
from math import prod

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_object, instantiate
from omegaconf import OmegaConf

from testbed.core.manager import Manager


def instantiate_design_net(config, task):
    """Instantiates a design net with the task arguments its target declares.

    Stochastic policy factories can opt into the task-owned design space,
    while deterministic policies continue to receive only their shape inputs.
    """
    target = get_object(config._target_)
    parameters = inspect.signature(target).parameters
    task_arguments = {
        "event_shape_y": task.event_shape_y,
        "event_shape_theta": task.event_shape_theta,
        "event_shape_design": task.event_shape_design,
        "event_shape_out": task.event_shape_design,
        "design_space": task.design_space,
    }
    return instantiate(
        config,
        **{name: value for name, value in task_arguments.items() if name in parameters},
    )


def instantiate_action_net(config, task, T):
    """Instantiates an action net, which outputs into the task's action space.

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
            {"dims": [dim_in, *config.dims[1:-1], prod(task.action_space.event_shape)]},
        )

    return instantiate(
        config,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.action_space.event_shape,
    )


def setup(cfg):
    """Build the run manager and task common to every Hydra entry point.

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
