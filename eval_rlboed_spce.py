"""Phase-0 diagnostic for the RL-BOED source-location reproduction.

Two jobs:
  1. Validate the eval pipeline with a RANDOM policy. The sPCE lower bound of a
     random design policy at T=30 should be ~1.62 (Blau et al. 2022, Table 1:
     Random = 1.624 +/- 0.053). If we reproduce that, the eval math is correct
     and the 11.73 gap is purely a training/selection issue.
  2. Re-evaluate a trained checkpoint (best.pt / best_eval.pt / final_step.pt)
     with a tighter estimate: more rollouts and/or several seeds pooled.

Scoring uses the canonical chunked ``SPCE`` estimator (testbed.design_objectives)
rather than the RL-BOED dense reward. The reward reuses a single contrastive set
sampled once per episode (capped by memory), whereas ``SPCE`` streams contrastive
particles in chunks, so ``--n-contrastive`` and ``--num-envs`` can be pushed far
higher than the reward allows. The rollout only needs the design/outcome history
and the true theta; the estimator draws its own contrastive particles afterwards.

Usage (from the repo root):
    # random-policy sanity check (small L is fine for the pipeline check)
    uv run python <this>.py --config testbed/recipes/location_finding/rlboed.yaml \
        --device cpu --num-envs 200 --n-contrastive 100000

    # evaluate a checkpoint with the paper protocol (needs a CUDA box for L=1e6)
    uv run python <this>.py --config testbed/recipes/location_finding/rlboed.yaml \
        --device cuda --checkpoint runs/.../best.pt \
        --num-envs 2000 --n-contrastive 1000000 --seeds 1,2,3 \
        --contrastive-chunk-size 50000
"""

from __future__ import annotations

import argparse
from math import sqrt

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from testbed.design_objectives import SPCE
from testbed.methods.rlboed import (
    build_design_space,
    build_env,
    make_networks,
)


def _pooled_stats(returns: torch.Tensor) -> tuple[float, float]:
    """Mean and standard error of a flat tensor of per-rollout sPCE bounds."""
    mean = returns.mean().item()
    se = returns.std(unbiased=True).item() / sqrt(returns.numel())
    return mean, se


@torch.no_grad()
def evaluate(
    cfg,
    task,
    design_space,
    actor,
    *,
    n_contrastive,
    contrastive_chunk_size,
    particle_budget,
    seed,
    device,
):
    """Rolls out `actor` (or a random policy if None) and scores it with sPCE.

    Scoring is the canonical chunked ``SPCE`` estimator — the sole evaluation
    metric. Returns the per-rollout sPCE lower bound as a ``[num_envs]`` tensor.
    """
    # The env reward is unused (we score the history directly), so build the env
    # with the cheapest possible contrastive count.
    eval_env = build_env(cfg, task, design_space, n_contrastive=0)
    obs, _ = eval_env.reset(seed=seed)

    num_envs = cfg.sac.num_envs
    low = design_space.low.to(device)
    high = design_space.high.to(device)

    for _ in range(cfg.T):
        if actor is None:  # random policy: uniform over the design box
            u = torch.rand(num_envs, *task.event_shape_design, device=device)
            designs = low + (high - low) * u
        else:
            designs = actor.sample_from_observation(obs, deterministic=True).design
        obs, _, _, _, _ = eval_env.step(designs)

    # Score the completed histories against fresh contrastive particles. SPCE
    # streams them in chunks, so L can far exceed the reward's fixed set.
    spce = SPCE(
        task,
        n_contrastive=n_contrastive,
        contrastive_chunk_size=contrastive_chunk_size,
        particle_budget=particle_budget,
    )
    history = eval_env.history
    return spce(
        history.outcomes,
        history.designs,
        eval_env.theta_0,
        reduce=False,
    )


def main() -> None:
    """Parses args, rolls out the policy, and reports pooled sPCE."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None, help="omit for a random policy")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-envs", type=int, default=200, help="rollouts per seed")
    p.add_argument("--n-contrastive", type=int, default=100_000)
    p.add_argument(
        "--contrastive-chunk-size",
        type=int,
        default=None,
        help="contrastive particles per env scored per chunk (bounds peak "
        "memory); omit to auto-size from --particle-budget",
    )
    p.add_argument(
        "--particle-budget",
        type=int,
        default=50_000_000,
        help="max contrastive particles (num_envs x per-env chunk) materialized "
        "at once when --contrastive-chunk-size is unset",
    )
    p.add_argument("--seeds", default="1", help="comma-separated eval seeds")
    p.add_argument("--design-low", type=float, default=None, help="override box low")
    p.add_argument("--design-high", type=float, default=None, help="override box high")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    cfg.device = args.device
    cfg.sac.num_envs = args.num_envs  # controls the number of parallel rollouts
    if args.design_low is not None:
        cfg.design_space.low = args.design_low
    if args.design_high is not None:
        cfg.design_space.high = args.design_high

    task = instantiate(cfg.task).to(device)
    design_space = build_design_space(cfg, task)

    actor = None
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        nets = make_networks(task, design_space, cfg, device)
        nets.actor.load_state_dict(ckpt["actor"])
        nets.actor.eval()
        actor = nets.actor

    seeds = [int(s) for s in args.seeds.split(",")]
    all_returns = []
    for seed in seeds:
        r = evaluate(
            cfg,
            task,
            design_space,
            actor,
            n_contrastive=args.n_contrastive,
            contrastive_chunk_size=args.contrastive_chunk_size,
            particle_budget=args.particle_budget,
            seed=seed,
            device=device,
        )
        mean, se = _pooled_stats(r)
        print(f"  seed={seed}: sPCE={mean:.4f} +/- {se:.4f}  (n={r.numel()})")
        all_returns.append(r)

    pooled = torch.cat(all_returns)
    mean, se = _pooled_stats(pooled)
    policy = "random" if actor is None else args.checkpoint
    print(
        f"\n[{policy}]  L={args.n_contrastive:,}  pooled sPCE = "
        f"{mean:.4f} +/- {se:.4f}  over {pooled.numel()} rollouts"
    )
    if actor is None:
        print(
            "NOTE: a UNIFORM random policy over [-4,4]^2 gives ~5.0 here; this is "
            "correct for this box and is NOT comparable to the paper's Random=1.624, "
            "which uses a different (boundary-clustered) random-design distribution. "
            "Random-policy sPCE is highly sensitive to the design-sampling range."
        )


if __name__ == "__main__":
    main()
