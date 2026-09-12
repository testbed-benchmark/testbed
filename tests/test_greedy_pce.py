"""Tests for the greedy (myopic) PCE baseline method."""

import torch

from testbed.design_objectives import PCE
from testbed.designers import RandomDesigner, SequentialDesigner
from testbed.eval import eval_end_to_end
from testbed.methods.greedy_pce import run_trajectory, summarize_ess
from testbed.tasks import LocationFinding

SIZES = {"n_contrastive": 2_000, "contrastive_chunk_size": 1_000}


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _greedy(task, T):
    return SequentialDesigner(
        task,
        T,
        PCE(task, n_outer=64, n_inner=63),
        lr=0.05,
        n_step_opt=15,
        n_proposal=2_000,
    )


def test_greedy_beats_random():
    # The point of the baseline: choosing designs by myopic EIG under the
    # running posterior should gather more information than choosing at random.
    T, n_thetas = 3, 6
    torch.manual_seed(0)

    task = _make_task()
    greedy_rollouts = [run_trajectory(task, _greedy(task, T)) for _ in range(n_thetas)]
    greedy = eval_end_to_end(task, greedy_rollouts, **SIZES)

    random_task = _make_task()
    random_designer = RandomDesigner(random_task, T)
    random_rollouts = [
        random_designer.rollout(random_task.prior.sample(n_samples=1))
        for _ in range(n_thetas)
    ]
    random = eval_end_to_end(random_task, random_rollouts, **SIZES)

    assert greedy["lower_bound"] > random["lower_bound"]


def test_scoring_is_not_measured_against_the_last_trajectory_posterior():
    # SequentialDesigner infers a posterior every step and leaves the last one
    # installed, and the contrastive draws come from current_prior. Scoring
    # without a reset would measure the bound against that posterior.
    torch.manual_seed(0)
    task = _make_task()
    rollouts = [run_trajectory(task, _greedy(task, T=2)) for _ in range(2)]
    installed = task.current_prior
    assert installed is not task.prior

    eval_end_to_end(task, rollouts, **SIZES)

    assert task.current_prior is installed


def test_ess_is_tracked_for_every_step_of_every_trajectory():
    # Greedy infers a posterior at every step, and the late steps are where
    # importance sampling degenerates, so the worst step matters as much as the
    # average.
    T, n_thetas = 3, 2
    torch.manual_seed(0)
    task = _make_task()
    rollouts = [run_trajectory(task, _greedy(task, T)) for _ in range(n_thetas)]

    summary = summarize_ess(rollouts)

    assert summary["n_inference_calls"] == T * n_thetas
    assert 0.0 < summary["ess_fraction_min"] <= summary["ess_fraction_mean"] <= 1.0
    assert 0.0 < summary["ess_fraction_final_mean"] <= 1.0
