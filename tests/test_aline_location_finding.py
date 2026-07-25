"""Tests for the BED-focused ALINE implementation."""

import json
from types import SimpleNamespace

import pytest
import torch

from testbed.core.manager import Manager
from testbed.design_nets import AlineModel, AlineOutput
from testbed.envs import BoxDesignSpace
from testbed.methods.aline import AlineRollout, Trainer, validate_config
from testbed.tasks import LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_cfg(**overrides):
    values = {
        "T": 3,
        "pool_size": 4,
        "n_initial": 1,
        "target_subsets": [[0, 1]],
        "target_weights": None,
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
        outcome_scale=10.0,
    )


def _make_trainer(cfg=None, manager=None):
    task = _make_task()
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
        torch.tensor([True, False]),
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
    target_mask = torch.tensor([True, True])

    output = model(context_y, context_designs, candidates, target_mask)
    changed = model(context_y, context_designs, changed_candidates, target_mask)

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

    rollout = trainer.rollout(theta, deterministic=True)

    assert rollout.designs.shape == (4, 3, 2)
    assert rollout.outcomes.shape == (4, 3, 1)
    assert rollout.posterior_log_probs.shape == (4, 3)
    assert rollout.action_log_probs.shape == (4, 2)
    assert rollout.rewards.shape == (4, 2)
    assert trainer.design_space.contains(rollout.designs)
    for designs in rollout.designs:
        assert torch.unique(designs, dim=0).shape[0] == trainer.cfg.T


def test_selected_log_prob_ignores_nontarget_parameters():
    trainer, _ = _make_trainer()
    output = AlineOutput(
        acquisition_logits=torch.zeros(2, 1),
        mixture_logits=torch.zeros(2, 2, 1),
        mixture_means=torch.zeros(2, 2, 1),
        mixture_scales=torch.ones(2, 2, 1),
    )
    theta = torch.tensor([[[0.0, 1.0]], [[0.5, 2.0]]])
    changed_theta = theta.clone()
    changed_theta[..., 1] = 100.0
    target_mask = torch.tensor([True, False])

    actual = trainer._selected_log_prob(output, theta, target_mask)
    changed = trainer._selected_log_prob(output, changed_theta, target_mask)

    torch.testing.assert_close(actual, changed)


def test_warmup_only_updates_inference_then_joint_updates_policy():
    torch.manual_seed(2)
    trainer, model = _make_trainer()
    before_warmup = [p.detach().clone() for p in model.acquisition_head.parameters()]

    warmup_metrics = trainer.train_step(step=0)

    assert warmup_metrics["policy_loss"] == 0.0
    for before, after in zip(before_warmup, model.acquisition_head.parameters()):
        torch.testing.assert_close(before, after)
    assert all(p.grad is None for p in model.acquisition_head.parameters())

    joint_metrics = trainer.train_step(step=1)

    assert all(torch.isfinite(torch.tensor(value)) for value in joint_metrics.values())
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.acquisition_head.parameters()
    )


def test_single_outer_sample_keeps_policy_gradient():
    torch.manual_seed(4)
    trainer, model = _make_trainer(
        cfg=_make_cfg(n_outer=1, n_warmup_iter=0)
    )

    metrics = trainer.train_step(step=0)

    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.acquisition_head.parameters()
    )


def test_policy_discount_starts_at_zero_power():
    trainer, _ = _make_trainer(
        cfg=_make_cfg(n_outer=2, n_warmup_iter=0, gamma=0.5)
    )
    rollout = AlineRollout(
        outcomes=torch.empty(2, 0, 1),
        designs=torch.empty(2, 0, 2),
        posterior_log_probs=torch.zeros(2, 2, requires_grad=True),
        action_log_probs=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        ),
        rewards=torch.tensor([[1.0, 3.0], [3.0, 1.0]]),
        target_mask=torch.tensor([True, True]),
        final_output=None,
    )
    trainer.rollout = lambda theta, random_policy=False: rollout

    metrics = trainer.train_step(step=0)

    assert metrics["policy_loss"] == pytest.approx(-0.25)


def test_validation_rejects_invalid_target_subset():
    task = _make_task()
    space = BoxDesignSpace(-4.0, 4.0, event_shape=task.event_shape_design)
    cfg = _make_cfg(target_subsets=[[0, 2]])

    with pytest.raises(ValueError, match="outside flattened theta"):
        validate_config(task, cfg, space)


@pytest.mark.parametrize(
    "weights",
    ([1.0, float("nan")], [1.0, float("inf")]),
)
def test_validation_rejects_nonfinite_target_weights(weights):
    task = _make_task()
    space = BoxDesignSpace(-4.0, 4.0, event_shape=task.event_shape_design)
    cfg = _make_cfg(target_subsets=[[0], [1]], target_weights=weights)

    with pytest.raises(ValueError, match="finite, non-negative"):
        validate_config(task, cfg, space)


def test_train_logs_metrics_and_saves_checkpoints(tmp_path):
    torch.manual_seed(3)
    manager = Manager(
        SimpleNamespace(),
        root=str(tmp_path),
        name="aline_test",
        timestamp=False,
        console=False,
    )
    cfg = _make_cfg(eval_every=1)
    trainer, _ = _make_trainer(cfg=cfg, manager=manager)

    trainer.train()

    run_dir = tmp_path / "aline_test"
    final = torch.load(run_dir / "final_step.pt", weights_only=True)
    best = torch.load(run_dir / "best.pt", weights_only=True)
    assert final["step"] == cfg.n_train_iter - 1
    assert set(best) == {"model", "step", "metric"}
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("inference_loss" in record for record in records)
    assert any("eval/posterior_nll" in record for record in records)
    assert any("eval/spce" in record for record in records)
