"""Step-DAD: semi-amortized policy-based BED (Hedman et al., ICML 2025).

Two phases. Phase one trains a DAD policy up front, as ``methods/dad.py`` does.
Phase two deploys it: for each of ``n_thetas`` ground truths it rolls the policy
out, and at each scheduled step infers the posterior from what has actually been
observed and fine-tunes the policy on the remaining horizon under that posterior
before carrying on.

Each trajectory gets its own copy of the pretrained weights, because refinement
mutates the policy in place and one experiment's refinement must not leak into
the next.

Run with ``python testbed/methods/stepdad.py [overrides...]``.
"""

import copy
import json
import math
from pathlib import Path

import hydra
import torch

from testbed.core.hooks import RefinementHook
from testbed.core.logging import flatten_metrics
from testbed.core.setup import instantiate_design_net, setup
from testbed.core.util import reset_prior
from testbed.designers import SemiAmortizedDesigner, make_refine_fn
from testbed.eval import (
    eval_end_to_end,
    eval_segment,
    eval_spce_snmc,
    sum_segments,
)
from testbed.methods.dad import Trainer, build_objective
from testbed.plotting import report_posterior_figures

# ============================================================================
# Phase 1: the pretrained policy
# ============================================================================


def pretrain(cfg, task, manager):
    """Trains a DAD policy from scratch."""
    design_net = instantiate_design_net(cfg.design_net, task).to(cfg.device)
    designer = SemiAmortizedDesigner(
        task,
        cfg.T,
        design_net,
        reparameterize_policy=cfg.objective.reparameterize_policy,
        use_raw_history=cfg.use_raw_history,
    )
    cfg.objective.requires_policy_log_probs = designer.requires_policy_log_probs

    optimizer = torch.optim.Adam(design_net.parameters(), lr=cfg.lr, betas=cfg.betas)
    scheduler = None
    if cfg.lr_anneal_every is not None and cfg.lr_gamma is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_anneal_every, gamma=cfg.lr_gamma
        )

    Trainer(
        task,
        designer,
        build_objective(cfg, task),
        optimizer,
        cfg,
        manager=manager,
        scheduler=scheduler,
    ).train()
    return design_net, cfg.use_raw_history


# ============================================================================
# Phase 2: deployment
# ============================================================================


def segment_boundaries(cfg):
    """The refinement steps that fall strictly inside the horizon."""
    return sorted({int(step) for step in cfg.refine_at if 0 < int(step) < cfg.T})


def segment_end(cfg, start):
    """Where the segment beginning at ``start`` ends: the next refinement, or T."""
    for boundary in segment_boundaries(cfg):
        if boundary > start:
            return boundary
    return cfg.T


def make_segment_eval_fn(cfg, task, designer, pretrained_designer):
    """Builds the ``RefinementHook`` eval_fn: the EIG of the segment just enabled.

    Runs at the same sample sizes as every other number in the run, from
    ``cfg.final_eval``, so the segments and the full-horizon rows are comparable.

    The posterior is whatever the hook installed before refining, so
    ``eval_segment`` picks it up from ``task.current_prior``.
    It is the same particle set the refinement itself optimized against.
    """
    arms = {"stepdad": designer, "no_finetune": pretrained_designer}

    def eval_fn(state, posterior):
        start = state.t + 1
        return {
            arm: eval_segment(
                task,
                arm_designer,
                state.y_history,
                state.design_history,
                n_steps=segment_end(cfg, start) - start,
                raw_design_history=state.raw_design_history,
                n_outer=cfg.final_eval.n_outer,
                n_contrastive=cfg.final_eval.n_contrastive,
                outer_batch_size=cfg.final_eval.outer_batch_size,
                contrastive_chunk_size=cfg.final_eval.contrastive_chunk_size,
            )
            for arm, arm_designer in arms.items()
        }

    return eval_fn


def build_refining_designer(
    cfg, task, design_net, use_raw_history, pretrained_designer, logger=None
):
    """Builds a deployment designer: its own copy of the weights, plus a refinement hook.

    Distinct from the plain designer :func:`pretrain` builds, which has no hook
    and shares the net it is training. This one carries a ``RefinementHook`` and
    a private copy of the weights, since refinement mutates them in place.

    The designer is constructed first and the hook appended to its ``ModuleList``,
    since the refinement closure needs the designer it is attached to.

    Args:
        cfg: The composed config.
        task: The task.
        design_net: Pretrained net, deep-copied so this designer owns its weights.
        use_raw_history: Whether the policy conditions on raw designs.
        pretrained_designer: The un-refined comparator each segment is measured
            against, under ``cfg.per_segment_eig``. Note that measuring consumes
            randomness inside the rollout, so a trajectory with per-segment EIG
            on is not step-for-step identical to one without at the same seed.
        logger: Optional logger for refinement progress.
    """
    designer = SemiAmortizedDesigner(
        task,
        cfg.T,
        copy.deepcopy(design_net),
        reparameterize_policy=cfg.objective.reparameterize_policy,
        use_raw_history=use_raw_history,
    )
    cfg.objective.requires_policy_log_probs = designer.requires_policy_log_probs
    objective = build_objective(cfg, task)

    designer.hooks.append(
        RefinementHook(
            cfg.refine_at,
            make_refine_fn(
                designer,
                objective,
                n_steps_opt=cfg.refine.n_steps_opt,
                lr=cfg.refine.lr,
                batch_size=cfg.refine.batch_size,
                grad_clip=cfg.refine.grad_clip,
                log_every=cfg.refine.log_every,
                logger=logger,
            ),
            eval_fn=(
                make_segment_eval_fn(cfg, task, designer, pretrained_designer)
                if cfg.per_segment_eig
                else None
            ),
            store_posterior_history=cfg.store_posterior_history,
            n_proposal=cfg.n_posterior_samples,
        )
    )
    return designer


def run_trajectory(task, designer):
    """Runs one deployment experiment against a fresh ground truth.

    The real steps need no graph; refinement re-enables grad for itself.
    """
    with reset_prior(task):
        theta_true = task.prior.sample(n_samples=1)
    with torch.no_grad():
        return designer.rollout(theta_true)


# ============================================================================
# Reporting
# ============================================================================


def combine_per_segment(rollouts, first, T):
    """Totals each arm's segments per trajectory, then averages over trajectories.

    The segments were already evaluated inside each rollout, at the refinement
    that produced them, and arrive on ``extra["refinement_evals"]``. All that is
    left is the chain-rule sum with the shared prior-measured opening segment,
    and the average across trajectories.

    The opening segment is the same number in every total: it is measured under
    the prior, with a policy no trajectory has refined yet, so it does not
    depend on which trajectory it is added to. That makes the spread across
    trajectories blind to how well it was estimated, so its own standard error
    is added back in quadrature rather than being dropped.
    """
    per_trajectory = []
    for rollout in rollouts:
        # Filled below into {"stepdad": [segment, ...], "no_finetune": [segment, ...]}
        arms = {}
        # record: {"step": 6, "stepdad": {...}, "no_finetune": {...}}
        for record in rollout.extra.get("refinement_evals", []):
            # key: "step" | "stepdad" | ...; value: 6 | {"lower_bound": ..., "start": 6, "end": 10}
            for key, value in record.items():
                # The step count sits alongside the arms; only the arms are segments.
                if key != "step":
                    # Seeded with the shared opening segment, copied so the arms do not alias.
                    arms.setdefault(key, [dict(first)]).append(value)
        per_trajectory.append(arms)

    results = {}
    # arm: "stepdad" | "no_finetune", assumed the same across trajectories
    for arm in per_trajectory[0]:
        # One chain-rule total per trajectory: [{"lower_bound": ..., "segments": [...]}, ...]
        totals = [sum_segments(trajectory[arm], T) for trajectory in per_trajectory]
        for bound in ("lower_bound", "upper_bound"):
            values = torch.tensor([total[bound] for total in totals])
            n = values.shape[0]
            across_trajectories = (
                (values.std(unbiased=True) / n**0.5).item() if n > 1 else 0.0
            )
            results.setdefault(arm, {})[bound] = values.mean().item()
            # Independent: the opening segment was estimated under the prior,
            # before any of these trajectories existed.
            results[arm][f"{bound}_se"] = math.hypot(
                across_trajectories, first[f"{bound}_se"]
            )
        results[arm]["n_thetas"] = len(per_trajectory)
    return results


def summarize_ess(rollouts):
    """Worst and mean ESS across every inference call, as a fraction."""
    fractions = [
        record["ess_fraction"]
        for rollout in rollouts
        for record in rollout.extra.get("refinement_ess", [])
    ]
    if not fractions:
        return {}
    return {
        "ess_fraction_min": min(fractions),
        "ess_fraction_mean": sum(fractions) / len(fractions),
        "n_inference_calls": len(fractions),
    }


# ============================================================================
# Entry point
# ============================================================================


@hydra.main(version_base="1.3", config_path="../recipes", config_name="method/stepdad")
def main(cfg):
    """Pretrains or loads a policy, deploys it with refinement, and scores it."""
    # Checked before the pretraining rather than at the plot, which is the last
    # thing a run does: the posteriors have to have been kept as they were
    # inferred, and there is no recovering them afterwards.
    if cfg.plot_posteriors and not cfg.store_posterior_history:
        raise ValueError(
            "plot_posteriors=true needs store_posterior_history=true; the plot "
            "draws the posterior particles, which are otherwise discarded as "
            "each refinement finishes."
        )

    manager, task = setup(cfg)
    torch.manual_seed(cfg.seed)

    design_net, use_raw_history = pretrain(cfg, task, manager)

    results = {}

    def pretrained_designer_over(horizon):
        """The un-refined policy, for a given horizon."""
        return SemiAmortizedDesigner(
            task,
            horizon,
            copy.deepcopy(design_net),
            reparameterize_policy=cfg.objective.reparameterize_policy,
            use_raw_history=use_raw_history,
        )

    pretrained_designer = pretrained_designer_over(cfg.T)


    # 1. Plain DAD over the full horizon
    with reset_prior(task):
        results["dad_unrefined"] = eval_spce_snmc(
            task,
            pretrained_designer,
            n_outer=cfg.final_eval.n_outer,
            n_contrastive=cfg.final_eval.n_contrastive,
            outer_batch_size=cfg.final_eval.outer_batch_size,
            contrastive_chunk_size=cfg.final_eval.contrastive_chunk_size,
        )

    # 2. Plain DAD up to the first refinement. Nothing has
    #    been observed at t=0, so this is the same eval as above over a shorter
    #    horizon, measured under the prior.
    #    Kept here instead of moving to after propose block like greedy because
    #    it reduces compute as each rollout can share the eval from up [0, tau[0]).
    #    Also it reduces se as across-trajectory variance reflects only genuine 
    #    trajectory-to-trajectory variation as opposed to double counting this initial
    #    segement
    first = None
    if cfg.per_segment_eig:
        opening = segment_boundaries(cfg)[0]
        with reset_prior(task):
            first = eval_spce_snmc(
                task,
                pretrained_designer_over(opening),
                n_outer=cfg.final_eval.n_outer,
                n_contrastive=cfg.final_eval.n_contrastive,
                outer_batch_size=cfg.final_eval.outer_batch_size,
                contrastive_chunk_size=cfg.final_eval.contrastive_chunk_size,
            )
        first = {**first, "start": 0, "end": opening}

    # 3. The experiments. Everything that depends on a realized trajectory comes
    #    from in here, including each segment's EIG, which is taken inside the
    #    refinement that produced it: under the posterior just inferred, and with
    #    the policy in the state that will actually design those steps.
    rollouts = []
    for index in range(cfg.n_thetas):
        designer = build_refining_designer(
            cfg,
            task,
            design_net,
            use_raw_history,
            pretrained_designer=pretrained_designer,
            logger=manager.event_logger,
        )
        rollouts.append(run_trajectory(task, designer))
        manager.event_logger.info(f"trajectory {index + 1}/{cfg.n_thetas} done")

    results.update(summarize_ess(rollouts))

    if cfg.plot_posteriors:
        report_posterior_figures(task, rollouts, manager)

    # 4. Combine, as the Step-DAD paper does: the opening segment plus each
    #    trajectory's tau-to-tau segments, averaged over trajectories. Nothing is
    #    evaluated here; the segments came from what was already ran in the refinement hook.
    if cfg.per_segment_eig:
        results["per_segment"] = {
            "first_segment": first,
            **combine_per_segment(rollouts, first, cfg.T),
        }

    # And the alternative total: one full-T sPCE over the refined trajectories
    # stacked together, which is directly comparable to the "dad" row above.
    if cfg.end_to_end_eig:
        results["end_to_end"] = eval_end_to_end(
            task,
            rollouts,
            n_contrastive=cfg.final_eval.n_contrastive,
            contrastive_chunk_size=cfg.final_eval.contrastive_chunk_size,
        )

    Path(manager.path("eval.json")).write_text(json.dumps(results, indent=2))
    if manager.metric_logger is not None:
        manager.metric_logger.log(flatten_metrics(results))
        manager.metric_logger.close()

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
