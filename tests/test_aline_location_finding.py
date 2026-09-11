"""Tests for the BED-focused ALINE implementation."""

import json
from types import SimpleNamespace

import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace
from testbed.core.manager import Manager
from testbed.core.rollout import Rollout
from testbed.design_nets import AlineModel, AlineOutput
from testbed.designers import AlineDesigner
from testbed.eval import evaluate_checkpoint
from testbed.methods.aline import Trainer
from testbed.tasks import LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_cfg(**overrides):
    values = {
        "T": 3,
        "pool_size": 4,
        "eval_pool_size": 5,
        "eval_batch_size": 2,
        "n_outer": 4,
        "n_outer_eval": 4,
        "n_inner_eval": 2,
        "n_train_iter": 2,
        "n_warmup_iter": 1,
        "eval_every": 0,
        "eval_deterministic": False,
        "gamma": 1.0,
        "alpha": 1.0,
        "grad_clip": 1.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_model(task, space):
    return AlineModel(
        event_shape_y=task.event_shape_y,
        event_shape_theta=task.event_shape_theta,
        event_shape_design=task.event_shape_design,
        design_low=space.low,
        design_high=space.high,
        embedding_dim=8,
        feedforward_dim=16,
        n_heads=2,
        n_layers=1,
        n_components=3,
    )


def _make_trainer(cfg=None, manager=None, task=None):
    task = _make_task() if task is None else task
    space = BoxDesignSpace(-4.0, 4.0, event_shape=task.event_shape_design)
    model = _make_model(task, space)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = Trainer(
        task,
        model,
        space,
        optimizer,
        cfg or _make_cfg(),
        manager=manager,
    )
    return trainer, model


def test_model_outputs_valid_policy_and_factorized_gmm():
    task = _make_task()
    space = BoxDesignSpace(-4.0, 4.0, event_shape=task.event_shape_design)
    model = _make_model(task, space)
    context_designs = space.sample((2, 2))
    context_y = task.sample_outcome(
        task.sample_prior(2).unsqueeze(1).expand(-1, 2, -1, -1),
        context_designs,
    )
    candidates = space.sample((2, 5))
    available = torch.tensor(
        [[True, True, False, True, True], [True, False, True, True, True]]
    )

    output = model(
        context_y,
        context_designs,
        candidates,
        available,
    )

    assert output.acquisition_logits.shape == (2, 5)
    assert output.mixture_means.shape == (2, 2, 3)
    assert output.posterior.batch_shape == torch.Size([2, 2])
    assert torch.all(output.mixture_scales > 0)
    assert torch.isneginf(output.acquisition_logits[~available]).all()
    assert output.posterior.log_prob(task.sample_prior(2).reshape(2, 2)).shape == (
        2,
        2,
    )


def test_fractional_outcome_scale_is_not_truncated_by_integer_center():
    task = _make_task()
    space = BoxDesignSpace(-4.0, 4.0, event_shape=task.event_shape_design)
    model = AlineModel(
        event_shape_y=task.event_shape_y,
        event_shape_theta=task.event_shape_theta,
        event_shape_design=task.event_shape_design,
        design_low=space.low,
        design_high=space.high,
        embedding_dim=8,
        feedforward_dim=16,
        n_heads=2,
        n_layers=1,
        n_components=3,
        outcome_center=0,
        outcome_scale=0.5,
    )

    assert model.outcome_center.is_floating_point()
    torch.testing.assert_close(model.outcome_scale, torch.tensor([0.5]))
    torch.testing.assert_close(
        model._normalize_outcomes(torch.tensor([[[1.0]]])),
        torch.tensor([[[2.0]]]),
    )


def test_fractional_design_bounds_are_not_truncated_by_integer_partners():
    task = _make_task()
    model = AlineModel(
        event_shape_y=task.event_shape_y,
        event_shape_theta=task.event_shape_theta,
        event_shape_design=task.event_shape_design,
        design_low=0,
        design_high=0.5,
        embedding_dim=8,
        feedforward_dim=16,
        n_heads=2,
        n_layers=1,
        n_components=3,
        design_center=0,
        design_scale=0.5,
    )

    assert model.design_low.is_floating_point()
    assert model.design_center.is_floating_point()
    torch.testing.assert_close(model.design_high, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(model.design_scale, torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(
        model._normalize_designs(torch.tensor([[[1.0, 1.0]]])),
        torch.tensor([[[2.0, 2.0]]]),
    )


def test_attention_prevents_candidate_leakage():
    torch.manual_seed(0)
    task = _make_task()
    space = BoxDesignSpace(-4.0, 4.0, event_shape=task.event_shape_design)
    model = _make_model(task, space).eval()
    theta = task.sample_prior(2)[:1]
    context_designs = space.sample((1, 2))
    context_y = task.sample_outcome(
        theta.unsqueeze(1).expand(-1, 2, -1, -1), context_designs
    )
    candidates = space.sample((1, 3))
    changed_candidates = candidates.clone()
    changed_candidates[:, 1:] = -changed_candidates[:, 1:]

    output = model(context_y, context_designs, candidates)
    changed = model(context_y, context_designs, changed_candidates)

    torch.testing.assert_close(
        output.acquisition_logits[:, 0], changed.acquisition_logits[:, 0]
    )
    torch.testing.assert_close(output.mixture_logits, changed.mixture_logits)
    torch.testing.assert_close(output.mixture_means, changed.mixture_means)
    torch.testing.assert_close(output.mixture_scales, changed.mixture_scales)


def test_rollout_samples_in_bounds_without_replacement():
    torch.manual_seed(1)
    trainer, _ = _make_trainer()
    theta = trainer.task.sample_prior(4)

    rollout = trainer.designer.rollout(
        theta, deterministic=True, return_posterior_scores=True
    )

    assert rollout.design_history.shape == (4, 3, 2)
    assert rollout.y_history.shape == (4, 3, 1)
    assert rollout.extra["posterior_log_probs"].shape == (4, 4)
    assert rollout.extra["policy_log_probs"].shape == (4, 3)
    assert isinstance(rollout, Rollout)
    assert trainer.design_space.contains(rollout.design_history)
    for designs in rollout.design_history:
        assert torch.unique(designs, dim=0).shape[0] == trainer.cfg.T


@pytest.mark.parametrize("random_policy", [False, True])
def test_shared_rollout_keeps_one_pool_and_masks_selected_candidates(random_policy):
    trainer, model = _make_trainer()
    designer = trainer.designer
    seen = []
    handle = model.register_forward_pre_hook(
        lambda module, inputs: seen.append((inputs[2], inputs[3].clone()))
    )
    theta = trainer.task.sample_prior(2).requires_grad_()
    rollout = designer.rollout(
        theta,
        random_policy=random_policy,
        pool_size=designer.T,
        return_posterior_scores=True,
    )
    handle.remove()

    assert len(seen) == designer.T + 1
    candidates = seen[0][0]
    for t, (pool, mask) in enumerate(seen):
        assert pool is candidates
        assert torch.all(mask.sum(dim=1) == designer.T - t)
        for b in range(2):
            for design in rollout.design_history[b, :t]:
                selected = (pool[b] == design).all(dim=-1)
                assert selected.sum() == 1
                assert not mask[b, selected].any()
    assert not rollout.y_history.requires_grad
    assert rollout.extra["posterior_log_probs"].shape == (2, designer.T + 1)
    assert rollout.extra["posterior_log_probs"].requires_grad
    assert ("policy_log_probs" in rollout.extra) is (not random_policy)
    assert designer.pool_size == trainer.cfg.pool_size
    assert designer.deterministic is False


def test_posterior_collection_preserves_deterministic_evaluation_trajectory():
    trainer, model = _make_trainer()
    model.eval()
    theta = trainer.task.sample_prior(2)
    torch.manual_seed(12)
    ordinary = trainer.designer.rollout(theta, deterministic=True)
    torch.manual_seed(12)
    scored = trainer.designer.rollout(
        theta, deterministic=True, return_posterior_scores=True
    )
    torch.testing.assert_close(ordinary.design_history, scored.design_history)
    torch.testing.assert_close(ordinary.y_history, scored.y_history)
    torch.testing.assert_close(
        ordinary.extra["policy_log_probs"], scored.extra["policy_log_probs"]
    )
    assert "posterior_log_probs" not in ordinary.extra


def test_designer_accepts_unbatched_theta_and_rejects_invalid_inputs():
    trainer, _ = _make_trainer()
    designer = trainer.designer
    theta = trainer.task.sample_prior(2)[0]
    assert designer.rollout(theta).theta.shape == (1, 1, 2)
    with pytest.raises(ValueError, match="pool_size must be at least T"):
        designer.rollout(theta, pool_size=designer.T - 1)
    with pytest.raises(ValueError, match="theta must have one batch dimension"):
        designer.rollout(torch.zeros(2, 2))


@pytest.mark.parametrize("n_sources, dim_theta", [(1, 2), (2, 3)])
def test_mean_log_prob_includes_every_parameter(n_sources, dim_theta):
    task = LocationFinding(n_sources=n_sources, dim_theta=dim_theta)
    space = BoxDesignSpace(-4.0, 4.0, event_shape=task.event_shape_design)
    model = _make_model(task, space)
    trainer = Trainer(task, model, space, None, _make_cfg())
    n_parameters = n_sources * dim_theta
    output = AlineOutput(
        acquisition_logits=torch.zeros(2, 1),
        mixture_logits=torch.zeros(2, n_parameters, 1),
        mixture_means=torch.zeros(2, n_parameters, 1),
        mixture_scales=torch.ones(2, n_parameters, 1),
    )
    theta = torch.ones(2, n_sources, dim_theta, requires_grad=True)
    actual = trainer.designer._mean_log_prob(output, theta)
    expected = torch.distributions.Normal(0, 1).log_prob(theta).mean(dim=(1, 2))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(theta.grad, torch.full_like(theta, -1 / n_parameters))


def test_single_outer_sample_has_finite_losses_and_gradients():
    torch.manual_seed(4)
    trainer, model = _make_trainer(cfg=_make_cfg(n_outer=1, n_warmup_iter=0))
    metrics = trainer.train_step(step=0)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_policy_reward_matches_reference_clipping_and_standardization():
    trainer, _ = _make_trainer(cfg=_make_cfg(n_outer=2, n_warmup_iter=0, gamma=0.5))
    rollout = Rollout(
        theta=trainer.task.sample_prior(2),
        y_history=torch.empty(2, 0, 1),
        design_history=torch.empty(2, 0, 2),
        extra={
            "posterior_log_probs": torch.tensor(
                [[0.0, -2.0, 1.0], [0.0, -1.0, -2.0]], requires_grad=True
            ),
            "policy_log_probs": torch.tensor(
                [[1.0, 2.0], [4.0, 4.0]], requires_grad=True
            ),
        },
    )
    trainer.designer.rollout = lambda theta, **kwargs: rollout

    metrics = trainer.train_step(step=0)

    assert metrics["policy_loss"] == pytest.approx(1 / (2 * 2**0.5))
    torch.testing.assert_close(
        rollout.extra["posterior_log_probs"].grad,
        torch.full_like(rollout.extra["posterior_log_probs"], -1 / 6),
    )


@pytest.mark.parametrize("training", [True, False])
def test_empty_context_has_finite_outputs_and_gradients(training):
    trainer, model = _make_trainer()
    model.train(training)
    candidates = trainer.design_space.sample((2, 3))
    output = model(
        torch.empty(2, 0, 1),
        torch.empty(2, 0, 2),
        candidates,
    )
    loss = output.mixture_means.sum() + output.acquisition_logits.sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_rollout_passes_history_and_trains_every_posterior_state():
    trainer, model = _make_trainer()
    sample_outcome = trainer.task.sample_outcome
    lengths = []

    def sample(theta, design, y_history=None, design_history=None):
        lengths.append(y_history.shape[1])
        assert y_history.shape[1] == design_history.shape[1]
        return sample_outcome(
            theta, design, y_history=y_history, design_history=design_history
        )

    trainer.task.sample_outcome = sample
    outputs = []
    handle = model.register_forward_hook(
        lambda module, inputs, output: outputs.append(output)
    )
    theta = trainer.task.sample_prior(4)
    rollout = trainer.designer.rollout(theta, return_posterior_scores=True)
    handle.remove()
    assert lengths == [0, 1, 2]
    expected = torch.stack(
        [out.posterior.log_prob(theta.reshape(4, 2)).mean(-1) for out in outputs],
        dim=1,
    )
    torch.testing.assert_close(rollout.extra["posterior_log_probs"], expected)


def test_joint_boundary_resets_optimizer_with_reference_learning_rates():
    trainer, model = _make_trainer()
    trainer.train_step(0)
    old = trainer.optimizer
    trainer.train_step(1)
    assert trainer.optimizer is not old
    assert [g["lr"] for g in trainer.optimizer.param_groups] == pytest.approx(
        [2e-4, 1e-3]
    )
    assert {id(p) for p in trainer.optimizer.param_groups[1]["params"]} == {
        id(p) for p in model.acquisition_head.parameters()
    }
