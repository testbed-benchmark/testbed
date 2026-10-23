"""Rebuilding a trained designer from a finished run's config and checkpoint."""

import logging
import os
from pathlib import Path

import torch
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from testbed.core.setup import instantiate_design_net
from testbed.designers import PolicyDesigner, StaticDesigner


def run_config(run_dir):
    """Returns the config a finished run recorded for itself.

    The Manager's ``config.yaml`` is what the run actually built, which for
    ``methods.posthoc_action`` includes the task and action-net widths it adopted at
    startup; ``.hydra/config.yaml`` is what Hydra composed, and stands in for
    runs written before the Manager kept a copy.
    """
    run_dir = Path(run_dir)
    for path in (run_dir / "config.yaml", run_dir / ".hydra" / "config.yaml"):
        if path.exists():
            return OmegaConf.load(path)
    raise FileNotFoundError(f"{run_dir} holds no config.yaml")


def run_checkpoint_path(run_dir, checkpoint=None):
    """Returns the checkpoint to reload, defaulting to the run's own choice."""
    checkpoint = checkpoint or OmegaConf.select(
        run_config(run_dir), "final_eval.checkpoint"
    )
    return Path(run_dir) / (checkpoint or "final_step.pt")


def designer_from_run(run_dir, task, device="cpu", manager=None, checkpoint=None):
    """Rebuilds the designer a finished run trained, with its weights loaded.

    The run's own config owns the design net's architecture and the history
    convention it proposes in, so both come from there rather than from a
    caller's -- hand-copying them is how a frozen design net silently ends up
    mismatched. ``final_eval.designer`` names the designer to rebuild, as it
    does for ``scripts/eval_run.py``; a static run saves a design, not a net.
    """
    src = run_config(run_dir)
    path = run_checkpoint_path(run_dir, checkpoint)
    state = torch.load(path, map_location=device, weights_only=True)

    designer_config = OmegaConf.select(src, "final_eval.designer")
    if designer_config is None:
        # Runs predating configurable final_eval.designer all trained a policy.
        designer_config = OmegaConf.create(
            {"_target_": "testbed.designers.PolicyDesigner"}
        )
    target = get_class(designer_config._target_)

    if issubclass(target, StaticDesigner):
        designer = instantiate(
            designer_config,
            task,
            src.T,
            transform_designs=state.get("transform_designs", True),
        )
        load_static_design(designer, state, manager)
    else:
        # The whole designer config is instantiated, keeping settings of its own
        # (ALINE's candidate pool, say), and the net is loaded before it is
        # handed over, since designers hold it under different names.
        design_net = instantiate_design_net(src.design_net, task).to(device)
        key = "design_net_ema" if "design_net_ema" in state else "design_net"
        design_net.load_state_dict(state[key])
        kwargs = {}
        if issubclass(target, PolicyDesigner):
            # A policy proposes in the coordinates it was trained in.
            kwargs["use_raw_history"] = state.get(
                "use_raw_history", not design_net.produces_policy_log_probs
            )
        designer = instantiate(designer_config, task, src.T, design_net, **kwargs)
        _logger(manager).info(f"Loaded {key} from {path}")

    _logger(manager).info(f"Rebuilt {type(designer).__name__} from {path}")
    return designer.to(device)


def load_designer(designer, source, manager=None):
    """Loads a trained designer's frozen state from ``source``.

    Dispatches on what the designer holds: a design net, or a static design.
    """
    if hasattr(designer, "design_net"):
        return load_design_net(designer, source, manager)
    if hasattr(designer, "raw_design"):
        return load_static_design(designer, source, manager)
    raise ValueError(
        f"{type(designer).__name__} holds neither a design net nor a static "
        "design; there is nothing to load into it."
    )


def load_design_net(designer, source, manager=None):
    """Loads a trained design net into ``designer`` from another method's checkpoint.

    ``source`` is a loaded checkpoint or a path to one. Every method stores its
    design net under ``"design_net"``, and ``methods.var_post`` additionally
    stores an EMA copy, which evaluation prefers -- so we do too. A checkpoint
    recording ``use_raw_history`` pins down the coordinates that net proposes
    in, so a designer configured to read the other history is a silent
    mismatch worth shouting about.
    """
    state = source
    if isinstance(state, (str, os.PathLike)):
        state = torch.load(state, map_location="cpu", weights_only=True)

    key = "design_net_ema" if "design_net_ema" in state else "design_net"
    designer.design_net.load_state_dict(state[key])

    trained_raw = state.get("use_raw_history")
    if trained_raw is not None and bool(trained_raw) != bool(
        getattr(designer, "use_raw_history", False)
    ):
        _logger(manager).warning(
            f"Checkpoint was trained with use_raw_history={bool(trained_raw)}, but the "
            f"designer is configured with use_raw_history="
            f"{bool(getattr(designer, 'use_raw_history', False))}; the frozen design net "
            "will see a history in coordinates it was not trained on."
        )

    _logger(manager).info(f"Loaded {key} from {_describe(source)}")


def load_static_design(designer, source, manager=None):
    """Loads a fixed design sequence into a static ``designer``'s raw design.

    ``source`` is a raw (pre-transform) design ``[T, *event_shape_design]``, or
    a path to one saved with ``torch.save``. A ``methods.static`` checkpoint is
    a dict bundling the winning restart's design with everything else that run
    produced, so as a special case we take ``"raw_design"`` out of it.
    """
    state = source
    if isinstance(state, (str, os.PathLike)):
        state = torch.load(state, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        transform_designs = state.get("transform_designs", designer.transform_designs)
        if bool(transform_designs) != bool(designer.transform_designs):
            _logger(manager).warning(
                f"Checkpoint was trained with transform_designs={bool(transform_designs)}, "
                f"but the designer is configured with "
                f"transform_designs={bool(designer.transform_designs)}; the same raw "
                "values then stand for different designs."
            )
        state = state["raw_design"]

    if state.shape != designer.raw_design.shape:
        raise ValueError(
            f"Design has shape {tuple(state.shape)}, but the designer expects "
            f"{tuple(designer.raw_design.shape)}. Check T and the task."
        )

    with torch.no_grad():
        designer.raw_design.copy_(state)

    _logger(manager).info(f"Loaded static design from {_describe(source)}")


def _logger(manager):
    """The run's event logger, or a module logger when there is no manager."""
    return manager.event_logger if manager else logging.getLogger(__name__)


def _describe(source):
    """Names the source in a log line: a path, or what was handed over."""
    if isinstance(source, (str, os.PathLike)):
        return str(source)
    return "the given checkpoint" if isinstance(source, dict) else "the given tensor"
