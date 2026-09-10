"""Tests for the BED-focused ALINE implementation."""

import json
from types import SimpleNamespace

import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace
from testbed.core.manager import Manager
from testbed.design_nets import AlineModel, AlineOutput
from testbed.methods.aline import AlineRollout, Trainer
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

    rollout = trainer.rollout(theta, deterministic=True)

    assert rollout.designs.shape == (4, 3, 2)
    assert rollout.outcomes.shape == (4, 3, 1)
    assert rollout.posterior_log_probs.shape == (4, 4)
    assert rollout.action_log_probs.shape == (4, 3)
    assert rollout.rewards.shape == (4, 3)
    assert trainer.design_space.contains(rollout.designs)
    for designs in rollout.designs:
        assert torch.unique(designs, dim=0).shape[0] == trainer.cfg.T


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
    actual = trainer._mean_log_prob(output, theta)
    expected = torch.distributions.Normal(0, 1).log_prob(theta).mean(dim=(1, 2))
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(theta.grad, torch.full_like(theta, -1 / n_parameters))


def test_warmup_only_updates_inference_then_joint_updates_policy():
    torch.manual_seed(2)
    trainer, model = _make_trainer()
    calls = {"random": 0, "policy": 0}
    random_action = trainer._random_action
    policy_action = trainer._policy_action

    def tracked_random_action(available_mask):
        calls["random"] += 1
        return random_action(available_mask)

    def tracked_policy_action(output, *, deterministic):
        calls["policy"] += 1
        return policy_action(output, deterministic=deterministic)

    trainer._random_action = tracked_random_action
    trainer._policy_action = tracked_policy_action
    before_warmup = [p.detach().clone() for p in model.acquisition_head.parameters()]

    warmup_metrics = trainer.train_step(step=0)

    assert warmup_metrics["policy_loss"] == 0.0
    assert calls == {"random": trainer.cfg.T, "policy": 0}
    for before, after in zip(before_warmup, model.acquisition_head.parameters()):
        torch.testing.assert_close(before, after)
    assert all(p.grad is None for p in model.acquisition_head.parameters())

    calls = {"random": 0, "policy": 0}
    joint_metrics = trainer.train_step(step=1)

    assert calls == {"random": 0, "policy": trainer.cfg.T}
    assert all(torch.isfinite(torch.tensor(value)) for value in joint_metrics.values())
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.acquisition_head.parameters()
    )


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
    rollout = AlineRollout(
        outcomes=torch.empty(2, 0, 1),
        designs=torch.empty(2, 0, 2),
        posterior_log_probs=torch.zeros(2, 3, requires_grad=True),
        action_log_probs=torch.tensor([[1.0, 2.0], [4.0, 4.0]], requires_grad=True),
        rewards=torch.tensor([[-2.0, 3.0], [-1.0, -1.0]]),
        final_output=None,
    )
    trainer.rollout = lambda theta, **kwargs: rollout

    metrics = trainer.train_step(step=0)

    assert metrics["policy_loss"] == pytest.approx(1 / (2 * 2**0.5))
    torch.testing.assert_close(
        rollout.posterior_log_probs.grad,
        torch.full_like(rollout.posterior_log_probs, -1 / 6),
    )


def test_evaluate_full_parameter_eig_without_target_configuration():
    torch.manual_seed(5)
    trainer, model = _make_trainer(task=LocationFinding(n_sources=2, dim_theta=3))
    rollout = trainer.rollout
    calls = []

    def tracked_rollout(theta, **kwargs):
        calls.append((theta.shape[0], kwargs.get("pool_size")))
        return rollout(theta, **kwargs)

    trainer.rollout = tracked_rollout
    metrics = trainer.evaluate()
    assert calls == [(2, trainer.cfg.eval_pool_size)] * 2
    assert set(metrics) == {"eval/posterior_nll", "eval/spce", "eval/snmc"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert model.training


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
    assert any("eval/snmc" in record for record in records)


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
    rollout = trainer.rollout(theta)
    handle.remove()
    assert lengths == [0, 1, 2]
    expected = torch.stack(
        [out.posterior.log_prob(theta.reshape(4, 2)).mean(-1) for out in outputs],
        dim=1,
    )
    torch.testing.assert_close(rollout.posterior_log_probs, expected)
    full_log_probs = [
        out.posterior.log_prob(theta.reshape(4, 2)).mean(-1) for out in outputs
    ]
    expected_rewards = torch.stack(
        [
            (next_log_prob - current_log_prob).detach()
            for current_log_prob, next_log_prob in zip(
                full_log_probs[:-1], full_log_probs[1:]
            )
        ],
        dim=1,
    )
    torch.testing.assert_close(
        rollout.rewards,
        expected_rewards,
    )


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


@pytest.mark.parametrize("training", [True, False])
def test_forward_matches_authors_code(training):
    # Optional integration check against the unmodified sibling checkout.
    # Load just the neural classes to avoid importing its training-only deps.
    import ast
    import os
    from pathlib import Path
    from typing import Any, List, Optional, Tuple

    reference = Path(
        os.environ.get(
            "ALINE_REFERENCE_DIR", Path(__file__).resolve().parents[2] / "ALINE"
        )
    )
    if not (reference / "model/encoder.py").exists():
        pytest.skip(
            "Set ALINE_REFERENCE_DIR to the authors' checkout for parity checks"
        )
    namespace = dict(
        torch=torch,
        nn=torch.nn,
        F=torch.nn.functional,
        Tensor=torch.Tensor,
        TransformerEncoderLayer=torch.nn.TransformerEncoderLayer,
        Any=Any,
        List=List,
        Optional=Optional,
        Tuple=Tuple,
        AttrDict=SimpleNamespace,
    )
    for filename, names in [
        ("embedder.py", {"Embedder"}),
        ("encoder.py", {"EfficientTransformerEncoderLayer", "Encoder"}),
        ("head.py", {"GMMTargetHead", "AcquisitionHead"}),
    ]:
        path = reference / "model" / filename
        tree = ast.parse(path.read_text())
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name in names
        ]
        exec(compile(tree, str(path), "exec"), namespace)
    torch.manual_seed(42)
    trainer, model = _make_trainer()
    model.train(training)
    embedder = namespace["Embedder"](
        2, 1, 8, 16, n_target_theta=2, embedding_type="theta"
    )
    encoder = namespace["Encoder"](8, 16, 2, 0.0, 1)
    encoder.device = "cpu"
    head = namespace["GMMTargetHead"](1, 8, 16, 3)
    acquisition = namespace["AcquisitionHead"](8, 16, False)
    embedder.x_embedder.load_state_dict(model.design_embedder.state_dict())
    embedder.y_embedder.load_state_dict(model.outcome_embedder.state_dict())
    embedder.theta_tokens.data.copy_(model.target_tokens.data)
    encoder.encoder.load_state_dict(model.transformer.state_dict())
    head.heads.load_state_dict(model.inference_heads.state_dict())
    acquisition.predictor.load_state_dict(model.acquisition_head.state_dict())
    for module in [embedder, encoder, head, acquisition]:
        module.train(training)
    context_x = torch.randn(2, 3, 2)
    context_y = torch.randn(2, 3, 1)
    candidates = torch.randn(2, 4, 2)
    mask = torch.ones(model.dim_theta, dtype=torch.bool)
    batch = SimpleNamespace(
        context_x=context_x,
        context_y=context_y,
        query_x=candidates,
        target_all=torch.zeros(2, 2, 1),
        target_mask=mask,
    )
    # Use no_grad in eval to exercise the native eval paths too.
    with torch.set_grad_enabled(training):
        encoded = encoder(batch, embedder(batch))
        expected = head(batch, encoded[:, 7:])
        expected_policy = acquisition(encoded[:, 3:7])
        actual = model(context_y, context_x, candidates)
        torch.testing.assert_close(actual.mixture_means, expected.mixture_means)
        torch.testing.assert_close(actual.mixture_scales, expected.mixture_stds)
        torch.testing.assert_close(
            actual.mixture_logits.softmax(-1), expected.mixture_weights
        )
        torch.testing.assert_close(
            actual.acquisition_logits.softmax(-1), expected_policy
        )
        if training:
            actual_loss = -actual.posterior.log_prob(torch.ones(2, 2)).mean()
            actual_loss = (
                actual_loss - actual.acquisition_logits.log_softmax(-1)[:, 0].mean()
            )
            expected_posterior = torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(probs=expected.mixture_weights),
                torch.distributions.Normal(
                    expected.mixture_means, expected.mixture_stds
                ),
            )
            expected_loss = -expected_posterior.log_prob(torch.ones(2, 2)).mean()
            expected_loss = expected_loss - expected_policy[:, 0].log().mean()
            actual_loss.backward()
            expected_loss.backward()
            torch.testing.assert_close(
                model.design_embedder[0].weight.grad, embedder.x_embedder[0].weight.grad
            )
            torch.testing.assert_close(
                model.target_tokens.grad, embedder.theta_tokens.grad
            )
            torch.testing.assert_close(
                model.acquisition_head[0].weight.grad,
                acquisition.predictor[0].weight.grad,
            )


def test_hydra_entry_point_trains_evaluates_and_saves(tmp_path):
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "testbed.methods.aline",
            "device=cpu",
            "T=3",
            "pool_size=4",
            "eval_pool_size=5",
            "eval_batch_size=2",
            "n_train_iter=2",
            "n_warmup_iter=1",
            "n_outer=4",
            "n_outer_eval=4",
            "n_inner_eval=2",
            "eval_every=1",
            "model.embedding_dim=8",
            "model.feedforward_dim=16",
            "model.n_heads=2",
            "model.n_layers=1",
            "model.n_components=3",
            f"hydra.run.dir={tmp_path}",
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "final_step.pt").exists()
