"""Tests for the BED-focused ALINE implementation."""

import json
from types import SimpleNamespace

import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace
from testbed.core.manager import Manager
from testbed.design_nets import AlineModel, AlineOutput
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
    assert rollout.action_log_probs.shape == (4, 1)
    assert rollout.rewards.shape == (4, 1)
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
        posterior_log_probs=torch.zeros(2, 2, requires_grad=True),
        action_log_probs=torch.tensor([[1.0, 2.0], [4.0, 4.0]], requires_grad=True),
        rewards=torch.tensor([[-2.0, 3.0], [-1.0, -1.0]]),
        target_mask=torch.tensor([True, True]),
        final_output=None,
        inference_log_probs=torch.zeros(2, 2, requires_grad=True),
    )
    trainer.rollout = lambda theta, **kwargs: rollout

    metrics = trainer.train_step(step=0)

    assert metrics["policy_loss"] == pytest.approx(1 / (2 * 2**0.5))


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


@pytest.mark.parametrize("training", [True, False])
def test_empty_context_has_finite_outputs_and_gradients(training):
    trainer, model = _make_trainer(cfg=_make_cfg(n_initial=0))
    model.train(training)
    candidates = trainer.design_space.sample((2, 3))
    output = model(
        torch.empty(2, 0, 1),
        torch.empty(2, 0, 2),
        candidates,
        torch.tensor([True, False]),
    )
    loss = output.mixture_means.sum() + output.acquisition_logits.sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_rollout_passes_history_and_trains_all_parameters_before_acquisition():
    trainer, model = _make_trainer(cfg=_make_cfg(target_subsets=[[0]]))
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
        [out.posterior.log_prob(theta.reshape(4, 2)).mean(-1) for out in outputs[1:-1]],
        dim=1,
    )
    torch.testing.assert_close(rollout.inference_log_probs, expected)
    targeted = [out.posterior.log_prob(theta.reshape(4, 2))[:, 0] for out in outputs]
    torch.testing.assert_close(
        rollout.rewards[:, 0], (targeted[2] - targeted[1]).detach()
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
    mask = torch.tensor([True, False])
    batch = SimpleNamespace(
        context_x=context_x,
        context_y=context_y / 10,
        query_x=candidates,
        target_all=torch.zeros(2, 2, 1),
        target_mask=mask,
    )
    # Use no_grad in eval to exercise the native eval paths too.
    with torch.set_grad_enabled(training):
        encoded = encoder(batch, embedder(batch))
        expected = head(batch, encoded[:, 7:])
        expected_policy = acquisition(encoded[:, 3:7])
        actual = model(context_y, context_x, candidates, mask)
        torch.testing.assert_close(actual.mixture_means, expected.mixture_means)
        torch.testing.assert_close(actual.mixture_scales, expected.mixture_stds)
        torch.testing.assert_close(
            actual.mixture_logits.softmax(-1), expected.mixture_weights
        )
        torch.testing.assert_close(
            actual.acquisition_logits.softmax(-1), expected_policy
        )
        if training:
            actual.mixture_means.sum().backward()
            expected.mixture_means.sum().backward()
            torch.testing.assert_close(
                model.design_embedder[0].weight.grad, embedder.x_embedder[0].weight.grad
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
