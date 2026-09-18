"""Standalone loss evaluation for a trained designer/action-net pair.

The counterpart to `eval_policy.py`: where that scores a design policy by its
sPCE/sNMC bounds, this scores a designer together with the action net that
reads its rollouts, by the losses that action incurs. It is what a finished
`methods.action` run is evaluated with, though nothing here assumes the action
net was trained that way -- any designer plus any action net can be scored.

Every loss is evaluated on one shared set of rollouts, so the numbers are
directly comparable to each other and a second loss costs no extra sampling.
Losses are scored per rollout (`reduction="none"`, averaged over event
dimensions), which is what gives each one a standard error.

Usage:
    python -m testbed.eval.eval_loss <run_dir>

<run_dir> is a training run's output directory. Checkpoint, sample sizes,
device and the losses to score all come from the run's saved `final_eval`
block, so they are set in the recipe and overridden like any other Hydra value
(`final_eval.n_outer=32`).
"""

import inspect
import math
import sys
from math import prod
from pathlib import Path

import torch
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from testbed.core.load import designer_from_run, run_checkpoint_path, run_config
from testbed.core.setup import instantiate_action_net
from testbed.designers import ActionHook

# Scored when a run's config names no losses of its own. Deliberately just the
# one every task supports: which of the sorted/permutation-invariant variants
# are meaningful depends on the task's parameters.
DEFAULT_LOSSES = {"mse": {"_target_": "testbed.design_objectives.MSELoss"}}


def build_losses(config, task):
    """Instantiates the named losses, each scoring one rollout at a time.

    ``reduction="none"`` is forced so every rollout keeps its own value, and
    the task is supplied to losses that declare it (e.g. ``LogLoss``).
    """
    losses = {}
    for name, loss_config in config.items():
        # A recipe drops a loss it inherited, but cannot apply, by nulling it.
        if loss_config is None:
            continue
        parameters = inspect.signature(get_class(loss_config._target_)).parameters
        overrides = {"reduction": "none"} if "reduction" in parameters else {}
        if "task" in parameters:
            overrides["task"] = task
        losses[name] = instantiate(loss_config, **overrides)
    return losses


def per_rollout(loss_fn, action, theta):
    """Returns one loss value per rollout, on the scale ``reduction="mean"`` reports.

    A ``reduction="none"`` loss keeps whatever shape it scores over: element-wise
    for MSE, but already summed to ``[b]`` for the permutation-invariant losses,
    whose ``"mean"`` still normalizes by the event size. Dividing each rollout's
    total by that event size reproduces ``"mean"`` for both, so a loss reported
    here is directly comparable to the same loss reported during training.
    """
    value = loss_fn(action, theta)
    if value.ndim == 0:
        raise ValueError(
            f"{type(loss_fn).__name__} reduced over the batch; it needs to "
            "support reduction='none' to be scored per rollout."
        )
    return value.reshape(value.shape[0], -1).sum(-1) / prod(action.shape[1:])


def eval_losses(
    task,
    designer,
    action_net,
    losses,
    n_outer: int = 2_048,
    outer_batch_size: int = 256,
    use_raw_history: bool = False,
    verbose: bool = True,
):
    """Scores every loss on one shared set of rollouts.

    The action net is attached to ``designer`` as an ``ActionHook``, so each
    rollout produces the terminal action the losses are scored on.
    ``use_raw_history`` must match what the action net was trained with.

    Args:
        task: The task to draw parameters and outcomes from.
        designer: The designer producing rollouts; its hooks and the action
            net's mode are left as they were found.
        action_net: The action net scored on those rollouts.
        losses: Mapping of name to a per-rollout loss (see ``build_losses``).
        n_outer: Total rollouts to score.
        outer_batch_size: Rollouts per batch.
        use_raw_history: Condition the action net on raw designs.
        verbose: Show a progress bar over the batches.

    Returns:
        dict: ``<name>`` and ``<name>_se`` for each loss, plus ``n_outer``.
    """
    # Borrowed for the duration only: the designer is handed back as it came,
    # so one designer can score several action nets in turn.
    designer.hooks.append(ActionHook(action_net, use_raw_history=use_raw_history))
    was_training = action_net.training
    action_net.eval()

    values = {name: [] for name in losses}
    remaining = n_outer
    progress = tqdm(
        total=math.ceil(n_outer / outer_batch_size),
        desc="eval_losses",
        unit="batch",
        disable=None if verbose else True,
    )

    try:
        with torch.no_grad(), progress:
            while remaining > 0:
                b = min(outer_batch_size, remaining)
                theta = task.sample_prior(b)
                action = designer.rollout(theta).extra["action"]
                for name, loss_fn in losses.items():
                    values[name].append(per_rollout(loss_fn, action, theta))
                remaining -= b
                progress.update(1)
    finally:
        del designer.hooks[-1]
        action_net.train(was_training)

    out = {}
    for name, batches in values.items():
        scores = torch.cat(batches)
        out[name] = scores.mean().item()
        out[f"{name}_se"] = (scores.std(unbiased=True) / len(scores) ** 0.5).item()
    out["n_outer"] = n_outer
    return out


def evaluate_loss_checkpoint(
    run_dir: str,
    checkpoint: str = "final_step.pt",
    n_outer: int = 2_048,
    outer_batch_size: int = 256,
    device: str = "cpu",
    losses=None,
    verbose: bool = True,
):
    """Computes mean +/- SEM losses for a saved action-net checkpoint.

    The run's own config rebuilds both halves of the pair: the action net from
    ``cfg.action_net``, and the designer it was trained against -- the frozen
    one from ``cfg.designer_run_dir``, or ``cfg.designer`` when designs were
    drawn without a net. ``losses`` overrides the run's ``final_eval.losses``.
    """
    run_dir = Path(run_dir)
    cfg = run_config(run_dir)

    task = instantiate(cfg.task).to(device)
    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=True)

    action_net = instantiate_action_net(cfg.action_net, task, cfg.T).to(device)
    action_net.load_state_dict(state["action_net"])

    designer_run_dir = OmegaConf.select(cfg, "designer_run_dir")
    if designer_run_dir is not None:
        designer = designer_from_run(designer_run_dir, task, device)
    else:
        designer = instantiate(cfg.designer, task=task, T=cfg.T).to(device)

    losses = (
        losses if losses is not None else OmegaConf.select(cfg, "final_eval.losses")
    )
    if losses is None:
        losses = OmegaConf.create(DEFAULT_LOSSES)

    out = eval_losses(
        task,
        designer,
        action_net,
        build_losses(losses, task),
        n_outer=n_outer,
        outer_batch_size=outer_batch_size,
        use_raw_history=state.get("use_raw_history", False),
        verbose=verbose,
    )
    out["checkpoint"] = checkpoint
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m testbed.eval.eval_loss <run_dir>")

    run_dir = Path(sys.argv[1])
    cfg = run_config(run_dir)
    settings = OmegaConf.select(cfg, "final_eval")
    if settings is None:
        raise SystemExit(
            f"{run_dir} has no final_eval block; add one to the method recipe."
        )

    print(
        evaluate_loss_checkpoint(
            str(run_dir),
            checkpoint=run_checkpoint_path(run_dir).name,
            n_outer=settings.n_outer,
            outer_batch_size=settings.outer_batch_size,
            device=cfg.device,
        )
    )
