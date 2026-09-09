"""Tests for the Step-DAD scoring functions."""

import copy

import pytest
import torch

from testbed.core.hooks import RefinementHook
from testbed.core.util import reset_prior
from testbed.design_nets import DeepSetPolicyNet
from testbed.design_objectives import SPCE
from testbed.designers import SemiAmortizedDesigner, make_refine_fn
from testbed.eval import eval_end_to_end, eval_segment, eval_spce_snmc, sum_segments
from testbed.inference import run_importance_sampling
from testbed.tasks import LocationFinding

T = 4
BOUNDARY = 2
SIZES = {
    "n_outer": 16,
    "n_contrastive": 64,
    "outer_batch_size": 8,
    "contrastive_chunk_size": 32,
}


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _run(task, net, n_thetas=2, horizon=T, refine_at=(BOUNDARY,)):
    """Runs independent deployment trajectories, one fresh designer each."""
    rollouts, designers = [], []
    for _ in range(n_thetas):
        designer = SemiAmortizedDesigner(task, T=horizon, design_net=copy.deepcopy(net))
        designer.hooks.append(
            RefinementHook(
                refine_at,
                make_refine_fn(
                    designer,
                    SPCE(task, n_contrastive=8),
                    n_steps_opt=3,
                    lr=1e-2,
                    batch_size=8,
                ),
                n_proposal=400,
            )
        )
        with torch.no_grad():
            rollouts.append(designer.rollout(task.prior.sample(n_samples=1)))
        designers.append(designer)
    return rollouts, designers


def test_total_scores_every_trajectory_together():
    torch.manual_seed(0)
    task = _make_task()
    rollouts, _ = _run(task, _make_net(task), n_thetas=3)

    result = eval_end_to_end(
        task, rollouts, n_contrastive=64, contrastive_chunk_size=32
    )

    assert result["n_thetas"] == 3
    assert result["T"] == T
    # sPCE is a lower bound and sNMC an upper bound on the same quantity.
    assert result["lower_bound"] <= result["upper_bound"]


def test_total_leaves_the_task_as_it_found_it():
    # Every trajectory leaves a posterior installed, and the contrastive draws
    # come from current_prior, so without the reset the bound would be measured
    # against the last trajectory's posterior.
    torch.manual_seed(0)
    task = _make_task()
    rollouts, _ = _run(task, _make_net(task))
    installed = task.current_prior
    assert installed is not task.prior

    eval_end_to_end(task, rollouts, n_contrastive=64, contrastive_chunk_size=32)

    assert task.current_prior is installed


def test_segment_is_measured_under_the_installed_posterior():
    # eval_segment takes whatever current_prior holds, so the caller controls the
    # measure. A posterior of identical particles makes that visible.
    torch.manual_seed(0)
    task = _make_task()
    rollouts, designers = _run(task, _make_net(task), n_thetas=1)
    rollout, designer = rollouts[0], designers[0]

    y_context = rollout.y_history[:, :BOUNDARY]
    design_context = rollout.design_history[:, :BOUNDARY]

    seen = []

    class _RecordingDesigner(type(designer)):
        def continue_from(self, theta, *args, **kwargs):
            seen.append(theta)
            return super().continue_from(theta, *args, **kwargs)

    designer.__class__ = _RecordingDesigner
    posterior = run_importance_sampling(task, y_context, design_context, n_proposal=400)
    task.update_prior(posterior)

    eval_segment(
        task,
        designer,
        y_context,
        design_context,
        n_steps=T - BOUNDARY,
        **SIZES,
    )

    # Drawn from the installed posterior's particles, not from the prior.
    assert torch.isin(seen[0], posterior.locs).all()


def test_segment_reports_the_span_it_covered():
    torch.manual_seed(0)
    task = _make_task()
    rollouts, designers = _run(task, _make_net(task), n_thetas=1)
    rollout = rollouts[0]

    result = eval_segment(
        task,
        designers[0],
        rollout.y_history[:, :BOUNDARY],
        rollout.design_history[:, :BOUNDARY],
        n_steps=T - BOUNDARY,
        **SIZES,
    )

    assert (result["start"], result["end"]) == (BOUNDARY, T)


def test_sum_segments_adds_a_tiling():
    first = {"lower_bound": 1.0, "upper_bound": 2.0, "start": 0, "end": 2}
    second = {"lower_bound": 0.5, "upper_bound": 1.5, "start": 2, "end": 5}

    # Out of order on purpose: the sort is what makes the tiling check meaningful.
    total = sum_segments([second, first], T=5)

    assert total["lower_bound"] == pytest.approx(1.5)
    assert total["upper_bound"] == pytest.approx(3.5)
    assert [segment["start"] for segment in total["segments"]] == [0, 2]


def test_sum_segments_rejects_a_gap():
    # A gap drops information and an overlap double-counts it, and either still
    # produces a plausible-looking number.
    first = {"lower_bound": 1.0, "upper_bound": 2.0, "start": 0, "end": 2}
    third = {"lower_bound": 0.5, "upper_bound": 1.5, "start": 3, "end": 5}

    with pytest.raises(ValueError, match="no gap or overlap"):
        sum_segments([first, third], T=5)


def test_sum_segments_rejects_an_overlap():
    first = {"lower_bound": 1.0, "upper_bound": 2.0, "start": 0, "end": 3}
    second = {"lower_bound": 0.5, "upper_bound": 1.5, "start": 2, "end": 5}

    with pytest.raises(ValueError, match="no gap or overlap"):
        sum_segments([first, second], T=5)


def test_sum_segments_rejects_a_short_horizon():
    first = {"lower_bound": 1.0, "upper_bound": 2.0, "start": 0, "end": 2}

    with pytest.raises(ValueError, match="but the horizon is 5"):
        sum_segments([first], T=5)


@pytest.mark.parametrize("horizon,boundaries", [(T, [BOUNDARY]), (6, [2, 4])])
def test_the_decomposition_tiles_for_one_and_for_many_boundaries(horizon, boundaries):
    # The single-intervention case and the multi-step case go through exactly the
    # same pieces; only the caller's loop differs.
    torch.manual_seed(0)
    task = _make_task()
    net = _make_net(task)
    rollouts, designers = _run(
        task, net, n_thetas=1, horizon=horizon, refine_at=tuple(boundaries)
    )
    rollout, designer = rollouts[0], designers[0]

    upto = SemiAmortizedDesigner(task, T=boundaries[0], design_net=copy.deepcopy(net))
    with reset_prior(task):
        opening = eval_spce_snmc(task, upto, verbose=False, **SIZES)
    segments = [{**opening, "start": 0, "end": boundaries[0]}]

    for start, end in zip(boundaries, boundaries[1:] + [horizon]):
        y_context = rollout.y_history[:, :start]
        design_context = rollout.design_history[:, :start]
        with torch.no_grad():
            posterior = run_importance_sampling(
                task, y_context, design_context, n_proposal=400
            )
        task.update_prior(posterior)
        segments.append(
            eval_segment(
                task,
                designer,
                y_context,
                design_context,
                n_steps=end - start,
                **SIZES,
            )
        )

    total = sum_segments(segments, T=horizon)

    expected_spans = [(0, boundaries[0])] + list(
        zip(boundaries, boundaries[1:] + [horizon])
    )
    assert [
        (segment["start"], segment["end"]) for segment in total["segments"]
    ] == expected_spans
    assert total["lower_bound"] == pytest.approx(
        sum(segment["lower_bound"] for segment in segments)
    )
