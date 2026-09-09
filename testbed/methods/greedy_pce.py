"""Greedy (myopic) PCE: the non-amortized baseline.

At each step, infer the posterior from what has been observed and pick the design
maximizing the one-step EIG under it, by gradient ascent on ``PCE``. No policy
and nothing to train, so this method only evaluates.

Everything it needs already exists: ``SequentialDesigner`` composes a
``PosteriorUpdateHook`` and runs the per-step Adam loop, and ``PCE`` is the
one-step-ahead sPCE bound under ``task.current_prior``. This module is here to
give the baseline a run directory, W&B logging and a sweep entry point on the
same footing as the trained methods.

Run with ``python testbed/methods/greedy_pce.py [overrides...]``.
"""

import json
from pathlib import Path

import hydra
import torch

from testbed.core.setup import setup
from testbed.core.util import reset_prior
from testbed.design_objectives import PCE
from testbed.designers import SequentialDesigner
from testbed.eval import eval_end_to_end
from testbed.plotting import report_posterior_figures


def build_designer(cfg, task):
    """Builds the greedy designer: per-step Adam on the myopic PCE bound."""
    return SequentialDesigner(
        task,
        cfg.T,
        PCE(task, n_outer=cfg.pce.n_outer, n_inner=cfg.pce.n_inner),
        cfg.pce.lr,
        cfg.pce.n_step_opt,
        init_design_dist=cfg.pce.init_design_dist,
        store_posterior_history=cfg.store_posterior_history,
        n_proposal=cfg.n_posterior_samples,
    )


def posteriors_to_plot(cfg, rollout):
    """The posteriors a figure should show: every step's, or just the last.

    Greedy infers one after every step, so keeping them all costs T particle
    sets per trajectory. The final posterior is on the rollout either way, and
    conditions on the whole horizon, so the cheap case is still the informative
    one.
    """
    return rollout.extra.get("posterior_history") or {cfg.T: rollout.extra["posterior"]}


def run_trajectory(task, designer):
    """Runs one experiment against a fresh ground truth.

    Batch 1, because each step's design comes from its own Adam loop:
    ``SequentialDesigner`` cannot vectorize independent per-example
    optimizations the way a policy net can.
    """
    with reset_prior(task):
        theta_true = task.prior.sample(n_samples=1)
    return designer.rollout(theta_true)


def summarize_ess(rollouts):
    """Worst, mean and final ESS across the per-step posterior updates.

    Greedy infers a posterior at every step, and the later steps are where
    importance sampling degenerates, so the minimum over the trajectory matters
    as much as the mean.
    """
    per_rollout = [rollout.extra.get("ess_history", []) for rollout in rollouts]
    fractions = [
        record["ess_fraction"] for history in per_rollout for record in history
    ]
    if not fractions:
        return {}
    finals = [history[-1]["ess_fraction"] for history in per_rollout if history]
    return {
        "ess_fraction_min": min(fractions),
        "ess_fraction_mean": sum(fractions) / len(fractions),
        "ess_fraction_final_mean": sum(finals) / len(finals),
        "n_inference_calls": len(fractions),
    }


@hydra.main(
    version_base="1.3", config_path="../recipes", config_name="method/greedy_pce"
)
def main(cfg):
    """Runs greedy PCE experiments and scores them."""
    manager, task = setup(cfg)
    torch.manual_seed(cfg.seed)

    rollouts = []
    for index in range(cfg.n_thetas):
        rollouts.append(run_trajectory(task, build_designer(cfg, task)))
        manager.event_logger.info(f"trajectory {index + 1}/{cfg.n_thetas} done")

    if cfg.plot_posteriors:
        report_posterior_figures(
            task,
            rollouts,
            manager,
            posteriors_for=lambda rollout: posteriors_to_plot(cfg, rollout),
        )

    results = {
        "end_to_end": eval_end_to_end(
            task,
            rollouts,
            n_contrastive=cfg.final_eval.n_contrastive,
            contrastive_chunk_size=cfg.final_eval.contrastive_chunk_size,
        ),
        **summarize_ess(rollouts),
    }

    Path(manager.path("eval.json")).write_text(json.dumps(results, indent=2))
    if manager.metric_logger is not None:
        manager.metric_logger.log(
            {
                f"eval/{key}": value
                for key, value in {
                    **{f"end_to_end/{k}": v for k, v in results["end_to_end"].items()},
                    **{
                        key: value
                        for key, value in results.items()
                        if isinstance(value, (int, float))
                    },
                }.items()
            }
        )
        manager.metric_logger.close()

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
