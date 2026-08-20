"""Standalone sPCE/sNMC evaluation for any trained BED policy checkpoint.

General-purpose: only needs `cfg.task`, `cfg.design_net`, `cfg.T` in the run's
saved config and a `"design_net"` key in the checkpoint.

If the checkpoint has a `"design_net_ema"` key (currently only
`methods.var_post.Trainer` saves one), it's preferred over the raw
`"design_net"` weights, matching e.g. ScoreBED's evaluation on
`policy_train_state.params_ema` -- pass `--use_raw_weights` to force the raw
weights instead.

`testbed.design_objectives.SPCE`/`SNMC` materialize all `L` contrastive
samples in one tensor (`[b*(L+1), T, ...]`), which is fine for their usual
training-time L (tens) but not for large L (e.g. L=500_000): that would need.
This module reimplements the same bound formulas
(SPCE: `logp_joint - logsumexp([logp_joint] + logp_contrastive) + log(L+1)`;
SNMC: same but excluding logp_joint from the marginal, `+ log(L)`) with the
contrastive dimension processed in chunks, combined via the
logsumexp-associativity identity `logsumexp(all) ==
logsumexp(stack([logsumexp(chunk) for chunk in chunks]))`
 -- so results are identical to calling SPCE/SNMC directly,
just without ever materializing all L contrastive samples at once.

Usage:
    python -m testbed.eval.eval_policy <run_dir> [--checkpoint final_step.pt]
        [--n_outer 2048] [--n_contrastive 500000] [--outer_batch_size 256]
        [--contrastive_chunk_size 2000] [--use_raw_weights]

<run_dir> is a training run's output directory (containing .hydra/config.yaml
and the checkpoint file).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.designers import PolicyDesigner


def build_design_net(cfg, task):
    """Instantiates a design policy net from a saved run config.

    (Duplicated in dad.py/actionbed.py/var_post.py; kept local here so
    this script has no dependency on any one method's Trainer module.)
    """
    return instantiate(
        cfg.design_net,
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _chunked_contrastive_logsumexps(task, rollout, n_contrastive, chunk_size):
    """Per-chunk logsumexp of contrastive log-likelihoods, plus logp_joint.

    Returns (logp_joint: [b], chunk_logsumexps: [b, n_chunks]), where
    chunk_logsumexps holds log(sum_{i in chunk} p(y | theta_i)) for each
    chunk of the n_contrastive prior draws (theta_0/logp_joint is NOT
    included in chunk_logsumexps -- callers combine as needed for
    SPCE (include) vs SNMC (exclude), matching their exact conventions).
    """
    y_history, design_history, theta_0 = (
        rollout.y_history,
        rollout.design_history,
        rollout.theta,
    )
    b = theta_0.shape[0]
    logp_joint = task.history_logp(y_history, design_history, theta_0)  # [b]

    chunk_logsumexps = []
    remaining = n_contrastive
    while remaining > 0:
        c = min(chunk_size, remaining)
        theta_c = task.sample_current_prior(b * c).reshape(
            b, c, *task.event_shape_theta
        )
        theta_flat = theta_c.reshape(b * c, *task.event_shape_theta)
        y_rep = (
            y_history.unsqueeze(1)
            .expand(b, c, *y_history.shape[1:])
            .reshape(b * c, *y_history.shape[1:])
        )
        design_rep = (
            design_history.unsqueeze(1)
            .expand(b, c, *design_history.shape[1:])
            .reshape(b * c, *design_history.shape[1:])
        )
        logp_c = task.history_logp(y_rep, design_rep, theta_flat).reshape(b, c)
        chunk_logsumexps.append(torch.logsumexp(logp_c, dim=1))  # [b]
        remaining -= c

    return logp_joint, torch.stack(chunk_logsumexps, dim=1)  # [b], [b, n_chunks]


def chunked_spce_snmc(task, rollout, n_contrastive, chunk_size):
    """SPCE (lower) and SNMC (upper) per-sample gains, chunked over n_contrastive.

    Identical formulas to testbed.design_objectives.SPCE/SNMC.per_sample_gain,
    just computed without materializing all n_contrastive samples at once.
    """
    logp_joint, chunk_logsumexps = _chunked_contrastive_logsumexps(
        task, rollout, n_contrastive, chunk_size
    )

    lower_marginal = torch.logsumexp(
        torch.cat([logp_joint.unsqueeze(1), chunk_logsumexps], dim=1), dim=1
    ) - np.log(n_contrastive + 1)
    lower = logp_joint - lower_marginal  # sPCE

    upper_marginal = torch.logsumexp(chunk_logsumexps, dim=1) - np.log(n_contrastive)
    upper = logp_joint - upper_marginal  # sNMC

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

    # use_raw_history mirrors the default the training run was built with, so
    # evaluation conditions the net on the same history coordinates.
    designer = PolicyDesigner(
        task,
        cfg.T,
        design_net,
        use_raw_history=not design_net.produces_policy_log_probs,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--checkpoint", default="final_step.pt")
    parser.add_argument("--n_outer", type=int, default=2048)
    parser.add_argument("--n_contrastive", type=int, default=500_000)
    parser.add_argument("--outer_batch_size", type=int, default=256)
    parser.add_argument("--contrastive_chunk_size", type=int, default=2_000)
    parser.add_argument("--use_raw_weights", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    results = evaluate_checkpoint(
        args.run_dir,
        checkpoint=args.checkpoint,
        n_outer=args.n_outer,
        n_contrastive=args.n_contrastive,
        outer_batch_size=args.outer_batch_size,
        contrastive_chunk_size=args.contrastive_chunk_size,
        use_raw_weights=args.use_raw_weights,
        device=args.device,
    )
    print(results)
