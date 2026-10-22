"""Characterization test: SequentialDesigner's default posterior-update
composition reproduces the pre-refactor behavior (propose -> sample -> infer
-> update_prior, inline, every step)."""

import pytest
import torch

from testbed.design_objectives import NMC, PCE
from testbed.designers import SequentialDesigner
from testbed.inference.is_sampler import run_importance_sampling
from testbed.tasks import LocationFinding, TemporalDiscounting


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _reference_rollout(task, theta_true, T, objective, lr, n_step_opt, n_proposal):
    """Reimplements the pre-refactor SequentialDesigner.rollout inline, for comparison.

    theta_true: [*event_shape_theta] (unbatched).
    """
    y_list, design_list = [], []
    for _ in range(T):
        design = torch.randn(task.event_shape_design, device=task.device)
        design.requires_grad_(True)
        optimizer = torch.optim.Adam([design], lr=lr)
        for _ in range(n_step_opt):
            optimizer.zero_grad()
            gain = objective.estimate_gain(design)
            (-gain).backward()
            optimizer.step()
        design = design.detach()

        y = task.sample_outcome(theta_true, design)
        y_list.append(y)
        design_list.append(design)

        y_history = torch.stack(y_list)
        design_history = torch.stack(design_list)
        posterior = run_importance_sampling(
            task, y_history, design_history, n_proposal=n_proposal
        )
        task.update_prior(posterior)

    return torch.stack(y_list), torch.stack(design_list)


def test_sequential_designer_matches_pre_refactor_reference():
    T, lr, n_step_opt, n_proposal = 3, 0.1, 5, 2_000
    seed = 0

    ref_task = _make_task()
    torch.manual_seed(seed)
    theta_batched = ref_task.prior.sample(n_samples=1)  # [1, *event_shape_theta]
    theta_true = theta_batched.reshape(*ref_task.event_shape_theta)

    ref_objective = NMC(ref_task, n_outer=8, n_inner=8)
    ref_y_history, ref_design_history = _reference_rollout(
        ref_task, theta_true, T, ref_objective, lr, n_step_opt, n_proposal
    )

    new_task = _make_task()
    torch.manual_seed(seed)
    theta_batched_new = new_task.prior.sample(n_samples=1)  # [1, *event_shape_theta]
    torch.testing.assert_close(theta_batched_new, theta_batched)

    new_objective = NMC(new_task, n_outer=8, n_inner=8)
    designer = SequentialDesigner(
        new_task,
        T,
        new_objective,
        lr,
        n_step_opt,
        inference_fn=run_importance_sampling,
        n_proposal=n_proposal,
    )
    rollout = designer.rollout(theta_batched_new)

    torch.testing.assert_close(rollout.y_history.squeeze(0), ref_y_history)
    torch.testing.assert_close(rollout.design_history.squeeze(0), ref_design_history)
    assert rollout.extra["posterior"] is new_task.current_prior
    # Posterior updates ran online, so the two tasks' priors both contracted
    # away from the (identical) original prior.
    assert new_task.current_prior is not new_task.prior


def test_raw_design_history_matches_forward_transform():
    task = TemporalDiscounting()
    objective = PCE(task, n_outer=4, n_inner=4)
    designer = SequentialDesigner(task, T=2, objective=objective, lr=1e-2, n_step_opt=3)

    rollout = designer.rollout(task.sample_prior(1))

    assert "raw_design_history" in rollout.extra
    raw_design_history = rollout.extra["raw_design_history"]
    assert raw_design_history.shape == rollout.design_history.shape
    torch.testing.assert_close(
        task.design_space.to_design(raw_design_history), rollout.design_history
    )


def test_transform_designs_false_opts_out():
    task = TemporalDiscounting()
    # With transform_designs=False, the raw optimized tensor (init ~N(0,1))
    # is used directly as the design, which won't satisfy TDDesignSpace's
    # real domain constraints (D1 > 0, 0 < R0 < 100) -- disable validation
    # since this test is only about the designer-level opt-out.
    task.validate_designs = False
    objective = PCE(task, n_outer=4, n_inner=4)
    designer = SequentialDesigner(
        task,
        T=2,
        objective=objective,
        lr=1e-2,
        n_step_opt=3,
        transform_designs=False,
    )

    rollout = designer.rollout(task.sample_prior(1))

    # No transform applied: raw and constrained designs coincide exactly.
    torch.testing.assert_close(
        rollout.extra["raw_design_history"], rollout.design_history
    )


def test_store_posterior_history_off_by_default():
    task = _make_task()
    objective = NMC(task, n_outer=4, n_inner=4)
    designer = SequentialDesigner(task, T=2, objective=objective, lr=1e-2, n_step_opt=3)

    rollout = designer.rollout(task.sample_prior(1))

    assert "posterior_history" not in rollout.extra


def test_store_posterior_history_tracks_one_per_step():
    T = 3
    task = _make_task()
    objective = NMC(task, n_outer=4, n_inner=4)
    designer = SequentialDesigner(
        task,
        T=T,
        objective=objective,
        lr=1e-2,
        n_step_opt=3,
        store_posterior_history=True,
    )

    rollout = designer.rollout(task.sample_prior(1))

    history = rollout.extra["posterior_history"]
    assert len(history) == T
    # The last entry in the history is the same posterior surfaced as
    # extra["posterior"] (the final task.current_prior).
    assert history[-1] is rollout.extra["posterior"]


def test_store_posterior_history_requires_update_posterior():
    task = _make_task()
    objective = NMC(task, n_outer=4, n_inner=4)

    with pytest.raises(ValueError, match="update_posterior"):
        SequentialDesigner(
            task,
            T=2,
            objective=objective,
            lr=1e-2,
            n_step_opt=3,
            update_posterior=False,
            store_posterior_history=True,
        )


def test_transform_designs_false_current_prior_init_is_identity():
    # initialize_design()'s "current_prior" branch inverts a constrained
    # sample through self._to_raw before optimizing; with transform_designs
    # =False that inverse is the identity, so the raw init must equal the
    # constrained sample unchanged -- no separate override needed to keep
    # the forward/inverse pair consistent.
    task = _make_task()
    objective = NMC(task, n_outer=4, n_inner=4)
    designer = SequentialDesigner(
        task,
        T=1,
        objective=objective,
        lr=1e-2,
        n_step_opt=1,
        init_design_dist="current_prior",
        update_posterior=False,
        transform_designs=False,
    )

    torch.manual_seed(0)
    design_constrained = task.sample_current_prior().to(task.device)

    torch.manual_seed(0)
    design_raw = designer.initialize_design()

    torch.testing.assert_close(design_raw.detach(), design_constrained)
