"""Standalone sPCE/sNMC evaluation for any trained BED policy checkpoint.

The saved `cfg.design_net` constructs the network, and `cfg.final_eval.designer`
selects the rollout implementation and its evaluation settings. Network weights
are stored under `"design_net"`.

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
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from testbed.core.setup import instantiate_design_net
from testbed.designers import PolicyDesigner
from testbed.eval import eval_spce_snmc


def build_design_net(cfg, task):
    """Instantiates a design policy net from a saved run config."""
    return instantiate_design_net(cfg.design_net, task)


def evaluate_checkpoint(
    run_dir: str,
    checkpoint: str = "final_step.pt",
    n_outer: int = 2048,
    n_contrastive: int = 500_000,
    outer_batch_size: int = 256,
    contrastive_chunk_size: int = 2_000,
    use_raw_weights: bool = False,
    device: str = "cpu",
    fused: bool = True,
):
    """Computes mean +/- SEM sPCE/sNMC bounds for a saved policy checkpoint."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    task = instantiate(cfg.task).to(device)
    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    design_net = build_design_net(cfg, task).to(device)
    key = (
        "design_net"
        if use_raw_weights or "design_net_ema" not in state
        else "design_net_ema"
    )
    design_net.load_state_dict(state[key])
    design_net.eval()

    designer_config = OmegaConf.select(cfg, "final_eval.designer")
    if designer_config is None:
        # Run configs saved before final_eval.designer became configurable used
        # PolicyDesigner unconditionally. Preserve that behavior so finished
        # checkpoints remain re-evaluable with newer versions of TestBED.
        designer_config = OmegaConf.create(
            {"_target_": "testbed.designers.PolicyDesigner"}
        )
    designer_kwargs = {}
    if issubclass(get_class(designer_config._target_), PolicyDesigner):
        # Policy checkpoints preserve the coordinates used during training.
        designer_kwargs["use_raw_history"] = state.get(
            "use_raw_history", not design_net.produces_policy_log_probs
        )
    designer = instantiate(designer_config, task, cfg.T, design_net, **designer_kwargs)

    out = eval_spce_snmc(
        task,
        designer,
        n_outer=n_outer,
        n_contrastive=n_contrastive,
        outer_batch_size=outer_batch_size,
        contrastive_chunk_size=contrastive_chunk_size,
        fused=fused,
    )
    out["weights_used"] = key
    return out


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
            # run configs saved before final_eval.fused existed default to fused
            fused=settings.get("fused", True),
        )
    )
