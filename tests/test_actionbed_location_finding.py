"""Tests for the ActionBED training components."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from testbed.core.distributions import TorchDistribution
from testbed.core.manager import Manager
from testbed.design_nets import CategoricalPolicyNet, DeepSetPolicyNet, MLPPolicyNet
from testbed.design_objectives import LotDecisionLoss, MSELoss, SortedMSELoss
from testbed.designers import PolicyDesigner, RandomDesigner
from testbed.methods.actionbed import Trainer, resolve_event_shape_action
from testbed.tasks import DestructiveLotInspection, LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_design_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _make_trainer(task=None, cfg=None, manager=None):
    task = task or _make_task()
    cfg = cfg or SimpleNamespace(n_outer=4, n_outer_eval=4, grad_clip=None)
    policy_net = _make_design_net(task)
    action_net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_theta,
    )
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    loss_fn = MSELoss()
    all_params = list(policy_net.parameters()) + list(action_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    return (
        Trainer(task, designer, action_net, loss_fn, optimizer, cfg, manager=manager),
        policy_net,
        action_net,
    )


def test_trainer_requires_policy_designer():
    task = _make_task()
    cfg = SimpleNamespace(n_outer=4, n_outer_eval=4, grad_clip=None)
    action_net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_theta,
    )
    optimizer = torch.optim.Adam(action_net.parameters(), lr=1e-3)

    with pytest.raises(AssertionError):
        Trainer(task, object(), action_net, MSELoss(), optimizer, cfg)


def test_train_step_returns_loss_key():
    trainer, _, _ = _make_trainer()
    metrics = trainer.train_step()
    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)


def test_train_step_updates_both_networks():
    trainer, policy_net, action_net = _make_trainer()
    params_before_design = [p.clone() for p in policy_net.parameters()]
    params_before_action = [p.clone() for p in action_net.parameters()]

    trainer.train_step()

    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before_design, policy_net.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before_action, action_net.parameters())
    )


def test_train_step_updates_categorical_design_and_action_networks():
    task = DestructiveLotInspection(num_zones=3)
    cfg = SimpleNamespace(n_outer=32, n_outer_eval=32, grad_clip=None)
    policy_net = CategoricalPolicyNet(
        task.design_space, dim_y=1, outcome_bounds=(0.0, 4.0)
    )
    action_net = MLPPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_action,
        dims=[6, 16, 1],
    )
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    optimizer = torch.optim.Adam(
        list(policy_net.parameters()) + list(action_net.parameters()), lr=1e-3
    )
    trainer = Trainer(
        task,
        designer,
        action_net,
        LotDecisionLoss(reduction="none"),
        optimizer,
        cfg,
    )
    params_before_design = [p.clone() for p in policy_net.parameters()]
    params_before_action = [p.clone() for p in action_net.parameters()]

    trainer.train_step()

    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before_design, policy_net.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before_action, action_net.parameters())
    )


def test_categorical_train_step_with_separate_optimizers_and_entropy():
    task = DestructiveLotInspection(num_zones=3)
    cfg = SimpleNamespace(
        n_outer=32,
        n_outer_eval=32,
        grad_clip=None,
        design_grad_clip=0.5,
        action_grad_clip=1.0,
        entropy_coef=1e-3,
    )
    policy_net = CategoricalPolicyNet(
        task.design_space, dim_y=1, outcome_bounds=(0.0, 4.0)
    )
    action_net = MLPPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_action,
        dims=[6, 16, 1],
    )
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    trainer = Trainer(
        task,
        designer,
        action_net,
        LotDecisionLoss(reduction="none"),
        torch.optim.Adam(policy_net.parameters(), lr=1e-3),
        cfg,
        action_optimizer=torch.optim.Adam(action_net.parameters(), lr=1e-3),
    )

    metrics = trainer.train_step()

    assert metrics["entropy"] > 0.0
    assert 0.0 <= metrics["loss"] <= 1.0


def test_categorical_action_warmup_uses_unreduced_loss():
    task = DestructiveLotInspection(num_zones=3)
    cfg = SimpleNamespace(
        n_outer=32,
        n_outer_eval=32,
        grad_clip=None,
        action_grad_clip=1.0,
    )
    policy_net = CategoricalPolicyNet(
        task.design_space, dim_y=1, outcome_bounds=(0.0, 4.0)
    )
    action_net = MLPPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_action,
        dims=[6, 16, 1],
    )
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    uniform = TorchDistribution(
        torch.distributions.Categorical(logits=torch.zeros(task.num_zones))
    )
    random_designer = RandomDesigner(task, T=3, dist=uniform)
    trainer = Trainer(
        task,
        designer,
        action_net,
        LotDecisionLoss(reduction="none"),
        torch.optim.Adam(policy_net.parameters(), lr=1e-3),
        cfg,
        action_optimizer=torch.optim.Adam(action_net.parameters(), lr=1e-3),
        random_designer=random_designer,
    )
    design_params_before = [parameter.clone() for parameter in policy_net.parameters()]

    metrics = trainer.action_warmup_step()

    assert isinstance(metrics["action_warmup/loss"], float)
    for before, after in zip(design_params_before, policy_net.parameters()):
        torch.testing.assert_close(before, after)


def test_train_step_with_grad_clip():
    cfg = SimpleNamespace(n_outer=4, n_outer_eval=4, grad_clip=1.0)
    trainer, _, _ = _make_trainer(cfg=cfg)
    metrics = trainer.train_step()
    assert "loss" in metrics


def test_evaluate_returns_eval_loss_key():
    trainer, _, _ = _make_trainer()
    metrics = trainer.evaluate()
    assert "eval/loss" in metrics
    assert isinstance(metrics["eval/loss"], float)


def test_evaluate_does_not_update_parameters():
    trainer, policy_net, action_net = _make_trainer()
    all_params_before = [
        p.clone() for net in (policy_net, action_net) for p in net.parameters()
    ]
    trainer.evaluate()
    all_params_after = [p for net in (policy_net, action_net) for p in net.parameters()]

    for before, after in zip(all_params_before, all_params_after):
        torch.testing.assert_close(before, after)


def test_resolve_event_shape_action_uses_custom_attribute():
    task = _make_task()
    task.event_shape_action = torch.Size([3])
    manager = MagicMock()

    shape = resolve_event_shape_action(task, manager)

    assert shape == torch.Size([3])
    manager.event_logger.warning.assert_not_called()


def test_resolve_event_shape_action_falls_back_to_theta():
    task = _make_task()
    manager = MagicMock()

    shape = resolve_event_shape_action(task, manager)

    assert shape == task.event_shape_theta
    manager.event_logger.warning.assert_called_once()


def test_maybe_save_best_saves_checkpoint_on_improvement(tmp_path):
    manager = Manager(
        SimpleNamespace(),
        root=str(tmp_path),
        name="test",
        timestamp=False,
        console=False,
    )
    trainer, _, _ = _make_trainer(manager=manager)

    trainer._maybe_save_best(step=0, metrics={"step": 0, "eval/loss": 0.5})

    best_path = tmp_path / "test" / "best.pt"
    assert best_path.exists()
    checkpoint = torch.load(best_path, weights_only=True)
    assert set(checkpoint.keys()) >= {"design_net", "action_net", "step", "metric"}
    assert checkpoint["metric"] == pytest.approx(0.5)
    assert checkpoint["step"] == 0

    # Worse metric (higher loss) should not overwrite the checkpoint
    trainer._maybe_save_best(step=1, metrics={"step": 1, "eval/loss": 1.0})
    checkpoint_after = torch.load(best_path, weights_only=True)
    assert checkpoint_after["metric"] == pytest.approx(0.5)


def test_sorted_mse_loss_is_compatible_with_train_step():
    task = _make_task()
    cfg = SimpleNamespace(n_outer=4, n_outer_eval=4, grad_clip=None)
    policy_net = _make_design_net(task)
    action_net = DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_theta,
    )
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    all_params = list(policy_net.parameters()) + list(action_net.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    trainer = Trainer(
        task, designer, action_net, SortedMSELoss(), optimizer, cfg, manager=None
    )

    metrics = trainer.train_step()
    assert "loss" in metrics
