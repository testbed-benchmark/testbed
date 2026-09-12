"""Tests for per-step evaluation on PosteriorUpdateHook and eval_fixed_design."""

import torch

from testbed.core.hooks import PosteriorUpdateHook
from testbed.design_objectives import PCE
from testbed.designers import SequentialDesigner
from testbed.eval import eval_fixed_design, sum_segments
from testbed.tasks import LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _designer(task, T, step_eval_fn, n_proposal=200):
    return SequentialDesigner(
        task,
        T,
        PCE(task, n_outer=8, n_inner=7),
        lr=0.05,
        n_step_opt=2,
        step_eval_fn=step_eval_fn,
        n_proposal=n_proposal,
    )


def test_step_evals_cover_every_step_of_the_horizon():
    # One segment per step, so they tile [0, T) with no separate opening term.
    torch.manual_seed(0)
    task = _make_task()
    designer = _designer(task, T=3, step_eval_fn=lambda state, prior: {"marker": 1})

    rollout = designer.rollout(task.sample_prior(1))

    assert [record["step"] for record in rollout.extra["step_evals"]] == [1, 2, 3]


def test_the_first_step_is_measured_under_the_original_prior():
    # The chain rule opens with a prior-measured term, and after_propose at t=0
    # runs before any posterior has been installed.
    torch.manual_seed(0)
    task = _make_task()
    seen = []

    def eval_fn(state, prior):
        seen.append(prior)
        return None

    _designer(task, T=2, step_eval_fn=eval_fn).rollout(task.sample_prior(1))

    assert seen[0] is task.prior
    assert seen[1] is not task.prior


def test_the_design_scored_is_the_one_the_trajectory_used():
    # The reason this runs in after_propose: a designer that optimizes each
    # design in turn cannot be re-asked for it later without landing elsewhere.
    torch.manual_seed(0)
    task = _make_task()
    scored = []

    def eval_fn(state, prior):
        scored.append(state.design.clone())
        return None

    rollout = _designer(task, T=3, step_eval_fn=eval_fn).rollout(task.sample_prior(1))

    torch.testing.assert_close(torch.cat(scored, dim=0), rollout.design_history[0])


def test_no_eval_fn_records_nothing():
    torch.manual_seed(0)
    task = _make_task()
    rollout = _designer(task, T=2, step_eval_fn=None).rollout(task.sample_prior(1))
    assert "step_evals" not in rollout.extra


def test_eval_fixed_design_labels_the_step_it_covers():
    torch.manual_seed(0)
    task = _make_task()
    y_history = torch.randn(1, 2, *task.event_shape_y)
    design_history = torch.randn(1, 2, *task.event_shape_design)

    result = eval_fixed_design(
        task,
        torch.randn(1, *task.event_shape_design),
        y_history,
        design_history,
        n_outer=16,
        n_contrastive=32,
        outer_batch_size=8,
        contrastive_chunk_size=16,
    )

    assert (result["start"], result["end"]) == (2, 3)
    assert result["lower_bound"] <= result["upper_bound"]


def test_per_step_evals_sum_as_a_tiling():
    # What combine_per_step relies on: one-step segments add up over the horizon.
    torch.manual_seed(0)
    task = _make_task()

    def eval_fn(state, prior):
        return eval_fixed_design(
            task,
            state.design,
            state.y_history,
            state.design_history,
            n_outer=8,
            n_contrastive=16,
            outer_batch_size=8,
            contrastive_chunk_size=16,
        )

    rollout = _designer(task, T=3, step_eval_fn=eval_fn).rollout(task.sample_prior(1))
    segments = [
        {key: value for key, value in record.items() if key != "step"}
        for record in rollout.extra["step_evals"]
    ]

    total = sum_segments(segments, T=3)
    assert total["lower_bound"] == sum(s["lower_bound"] for s in segments)
