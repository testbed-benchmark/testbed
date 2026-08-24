"""Standalone sPCE/sNMC evaluation for any trained BED policy checkpoint.

General-purpose: only needs `cfg.task`, `cfg.design_net`, `cfg.T` in the run's
saved config and a `"design_net"` key in the checkpoint.

If the checkpoint has a `"design_net_ema"` key (currently only
`methods.var_post.Trainer` saves one), it's preferred over the raw
`"design_net"` weights, matching e.g. ScoreBED's evaluation on
`policy_train_state.params_ema` -- pass `--use_raw_weights` to force the raw
weights instead.

The bounds come from `testbed.design_objectives`: contrastive samples are
scored by `contrastive_logsumexps` in chunks (evaluation uses L=500_000, which
does not fit as one `[b*L, T, ...]` tensor), and combined by
`SPCE.gain_from_logsumexps` / `SNMC.gain_from_logsumexps`. Both bounds share
one set of draws, so the sampling is paid for once.

Usage:
    python -m testbed.eval.eval_policy <run_dir>

<run_dir> is a training run's output directory (containing .hydra/config.yaml
and the checkpoint file). Checkpoint, sample sizes and device all come from the
run's saved `final_eval` block, so they are set in the recipe and overridden
like any other Hydra value (`final_eval.n_outer=32`).
"""

import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.core.setup import instantiate_design_net
from testbed.design_objectives import SNMC, SPCE
from testbed.design_objectives.contrastive import contrastive_logsumexps
from testbed.designers import PolicyDesigner


def build_design_net(cfg, task):
    """Instantiates a design policy net from a saved run config."""
    return instantiate_design_net(cfg.design_net, task)


def chunked_spce_snmc(task, rollout, n_contrastive, chunk_size):
    """SPCE (lower) and SNMC (upper) per-sample gains, chunked over n_contrastive.

    Both bounds come from one set of contrastive draws, so the expensive
    sampling is paid for once rather than per bound.
    """
    logp_joint, chunk_logsumexps = contrastive_logsumexps(
        task, rollout, n_contrastive, chunk_size
    )
    lower = SPCE.gain_from_logsumexps(logp_joint, chunk_logsumexps, n_contrastive)
    upper = SNMC.gain_from_logsumexps(logp_joint, chunk_logsumexps, n_contrastive)
    return lower, upper


def evaluate_checkpoint(
    run_dir: str,
    checkpoint: str = "final_step.pt",
    n_outer: int = 2048,
    n_contrastive: int = 500_000,
    outer_batch_size: int = 256,
    contrastive_chunk_size: int = 2_000,
    use_raw_weights: bool = False,
    device: str = "cpu",
):
    """Computes mean +/- SEM sPCE/sNMC bounds for a saved policy checkpoint."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    task = instantiate(cfg.task).to(device)
    design_net = build_design_net(cfg, task).to(device)

    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    key = (
        "design_net"
        if use_raw_weights or "design_net_ema" not in state
        else "design_net_ema"
    )
    design_net.load_state_dict(state[key])
    design_net.eval()

    use_raw_history = state.get(
        "use_raw_history", not design_net.produces_policy_log_probs
    )
    designer = PolicyDesigner(
        task,
        cfg.T,
        design_net,
        use_raw_history=use_raw_history,
    )

    lower_gains, upper_gains = [], []
    remaining = n_outer
    with torch.no_grad():
        while remaining > 0:
            b = min(outer_batch_size, remaining)
            theta0 = task.sample_prior(b)
            rollout = designer.rollout(theta0)
            lower, upper = chunked_spce_snmc(
                task, rollout, n_contrastive, contrastive_chunk_size
            )
            lower_gains.append(lower)
            upper_gains.append(upper)
            remaining -= b

    lower_gains = torch.cat(lower_gains)
    upper_gains = torch.cat(upper_gains)
    n = lower_gains.shape[0]

    return {
        "lower_bound": lower_gains.mean().item(),
        "lower_bound_se": (lower_gains.std(unbiased=True) / n**0.5).item(),
        "upper_bound": upper_gains.mean().item(),
        "upper_bound_se": (upper_gains.std(unbiased=True) / n**0.5).item(),
        "weights_used": key,
        "n_outer": n,
        "n_contrastive": n_contrastive,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m testbed.eval.eval_policy <run_dir>")

    run_dir = Path(sys.argv[1])
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    settings = OmegaConf.select(cfg, "final_eval")
    if settings is None:
        raise SystemExit(
            f"{run_dir} has no final_eval block; add `- /final_eval: default` to "
            f"the method recipe's defaults."
        )

    print(
        evaluate_checkpoint(
            str(run_dir),
            checkpoint=settings.checkpoint,
            n_outer=settings.n_outer,
            n_contrastive=settings.n_contrastive,
            outer_batch_size=settings.outer_batch_size,
            contrastive_chunk_size=settings.contrastive_chunk_size,
            use_raw_weights=settings.use_raw_weights,
            device=cfg.device,
        )
    )
