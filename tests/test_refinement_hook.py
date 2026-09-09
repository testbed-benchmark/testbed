"""Tests for RefinementHook and the refine_designer / make_refine_fn pair."""

import copy

import pytest
import torch

from testbed.core.hooks import RefinementHook
from testbed.design_nets import DeepSetPolicyNet
from testbed.design_objectives import SPCE
from testbed.designers import PolicyDesigner, SemiAmortizedDesigner, make_refine_fn
from testbed.tasks import LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _designer(task, T, refine_at, refine_fn, n_proposal=500):
    designer = SemiAmortizedDesigner(task, T=T, design_net=_make_net(task))
    designer.hooks.append(RefinementHook(refine_at, refine_fn, n_proposal=n_proposal))
    return designer


def test_fires_only_at_the_scheduled_steps():
    torch.manual_seed(0)
    task = _make_task()
    fired = []
    designer = _designer(
        task,
        T=5,
        refine_at=[2, 4],
        refine_fn=lambda state, posterior: fired.append(state.t),
    )

    designer.rollout(task.sample_prior(1))

    # 1-indexed step counts: refining "after 2 steps" is state.t == 1.
    assert fired == [1, 3]


def test_never_fires_on_the_last_step():
    # The horizon a refinement would optimize is empty there.
    torch.manual_seed(0)
    task = _make_task()
    fired = []
    designer = _designer(
        task,
        T=3,
        refine_at=[3],
        refine_fn=lambda state, posterior: fired.append(state.t),
    )

    designer.rollout(task.sample_prior(1))

    assert fired == []


def test_posterior_is_installed_before_refine_fn_runs():
    # refine_designer draws theta from current_prior, so the ordering is what
    # makes the fine-tune a posterior objective rather than a prior one.
    torch.manual_seed(0)
    task = _make_task()
    seen = []

    def refine_fn(state, posterior):
        seen.append(
            (state.task.current_prior is posterior, posterior is not state.task.prior)
        )

    designer = _designer(task, T=4, refine_at=[2], refine_fn=refine_fn)
    designer.rollout(task.sample_prior(1))

    assert seen == [(True, True)]


def test_refine_at_rejects_non_positive_steps():
    with pytest.raises(ValueError, match="1-indexed"):
        RefinementHook([0], lambda state, posterior: None)


def test_a_noop_refinement_leaves_the_rollout_unchanged():
    # Isolates the hook's own effect on the rollout: with refine_fn doing
    # nothing, the only difference from a plain PolicyDesigner rollout is the
    # installed prior, which a deterministic policy's rollout never reads.
    #
    # The inference stub consumes no randomness, deliberately. Real importance
    # sampling draws n_proposal prior samples, which shifts the RNG stream and
    # makes every later outcome differ -- worth knowing before comparing any two
    # seeded rollouts across an inference call.
    task = _make_task()

    torch.manual_seed(0)
    net = _make_net(task)
    plain = PolicyDesigner(task, T=4, design_net=net)
    torch.manual_seed(1)
    plain_rollout = plain.rollout(task.sample_prior(1))

    task_refined = _make_task()
    torch.manual_seed(0)
    semi = SemiAmortizedDesigner(task_refined, T=4, design_net=_make_net(task_refined))
    semi.hooks.append(
        RefinementHook(
            [2],
            lambda state, posterior: None,
            inference_fn=lambda task, y_history, design_history: task.prior,
        )
    )
    torch.manual_seed(1)
    refined_rollout = semi.rollout(task_refined.sample_prior(1))

    torch.testing.assert_close(refined_rollout.y_history, plain_rollout.y_history)
    torch.testing.assert_close(
        refined_rollout.design_history, plain_rollout.design_history
    )


def test_refinement_records_ess_and_metrics():
    torch.manual_seed(0)
    task = _make_task()
    designer = _designer(
        task, T=4, refine_at=[2], refine_fn=lambda state, posterior: {"marker": 7}
    )

    rollout = designer.rollout(task.sample_prior(1))

    assert len(rollout.extra["refinement_ess"]) == 1
    ess = rollout.extra["refinement_ess"][0]
    assert 1.0 <= ess["ess"] <= ess["n_particles"]
    # Recorded against the step count it conditions on, matching refine_at.
    assert rollout.extra["refinement_metrics"] == [{"step": 2, "marker": 7}]
    assert sorted(rollout.extra["posterior_history"]) == [2]


def test_refinement_updates_the_policy_and_improves_its_own_objective():
    torch.manual_seed(0)
    task = _make_task()
    designer = SemiAmortizedDesigner(task, T=4, design_net=_make_net(task))
    before = copy.deepcopy(designer.design_net.state_dict())

    designer.hooks.append(
        RefinementHook(
            [2],
            make_refine_fn(
                designer,
                SPCE(task, n_contrastive=32),
                n_steps_opt=15,
                lr=1e-2,
                batch_size=32,
            ),
            n_proposal=500,
        )
    )
    rollout = designer.rollout(task.sample_prior(1))

    after = designer.design_net.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)

    # Its own training objective, so a working fine-tune should not send it down.
    metrics = rollout.extra["refinement_metrics"][0]
    assert metrics["gain_after"] > metrics["gain_before"]


def test_multiple_refinements_are_cumulative():
    # The second refinement starts from the first one's weights, not the
    # pretrained ones.
    torch.manual_seed(0)
    task = _make_task()
    designer = SemiAmortizedDesigner(task, T=6, design_net=_make_net(task))
    snapshots = []

    refine = make_refine_fn(
        designer, SPCE(task, n_contrastive=16), n_steps_opt=5, lr=1e-2, batch_size=16
    )

    def refine_fn(state, posterior):
        metrics = refine(state, posterior)
        snapshots.append(copy.deepcopy(designer.design_net.state_dict()))
        return metrics

    designer.hooks.append(RefinementHook([2, 4], refine_fn, n_proposal=500))
    designer.rollout(task.sample_prior(1))

    assert len(snapshots) == 2
    first, second = snapshots
    assert any(not torch.equal(first[k], second[k]) for k in first)
