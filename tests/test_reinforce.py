"""Tests for the REINFORCE gradient estimator."""

from types import SimpleNamespace

import pytest
import torch

from testbed.design_nets import DeepSetPolicyNet
from testbed.design_objectives import SPCE, Reinforce
from testbed.design_objectives.reinforce import leave_one_out_baseline
from testbed.designers import PolicyDesigner
from testbed.methods.dad import Trainer, build_objective
from testbed.tasks import HyperbolicTemporalDiscounting


def _make_task():
    # unconstrained_design keeps the likelihood non-saturated for an untrained
    # policy, so design gradients are non-zero.
    return HyperbolicTemporalDiscounting(unconstrained_design=True)


def _make_design_net(task):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _make_rollout(task, policy_net, b=8, T=3):
    designer = PolicyDesigner(task, T=T, policy_net=policy_net)
    theta0 = task.sample_prior(b)
    y_obs, design_obs = designer.rollout(theta0)
    return y_obs, design_obs, theta0


def _spce_reward(task, n_contrastive=4):
    return SPCE(task, n_contrastive=n_contrastive).per_sample_gain


def _policy_grad(policy_net, gain):
    grads = torch.autograd.grad(gain, list(policy_net.parameters()), retain_graph=True)
    return torch.cat([g.flatten() for g in grads])


def test_build_objective_creates_reinforce():
    task = _make_task()
    cfg = SimpleNamespace(objective=SimpleNamespace(name="reinforce", n_inner=4))
    obj = build_objective(cfg, task)
    assert isinstance(obj, Reinforce)
    assert obj.baseline is None
    assert obj.differentiable_reward


def test_value_equals_mean_spce_reward():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    y_obs, design_obs, theta0 = _make_rollout(task, policy_net)

    spce = SPCE(task, n_contrastive=16)
    reinforce = Reinforce(
        task, reward_fn=spce.per_sample_gain, differentiable_reward=True
    )

    torch.manual_seed(1)  # contrastive draws
    gain_reinforce = reinforce.estimate_gain(y_obs, design_obs, theta0)
    torch.manual_seed(1)
    gain_spce = spce.estimate_gain(y_obs, design_obs, theta0)

    assert torch.allclose(gain_reinforce, gain_spce)


def test_score_function_term_with_black_box_reward():
    # With a detached (black-box) reward and differentiable_reward=False, any
    # policy gradient must come from the score-function term.
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    y_obs, design_obs, theta0 = _make_rollout(task, policy_net)

    def black_box_reward(y, d, t):
        return _spce_reward(task)(y, d, t).detach()

    reinforce = Reinforce(task, reward_fn=black_box_reward, differentiable_reward=False)
    gain = reinforce.estimate_gain(y_obs, design_obs, theta0)
    grad = _policy_grad(policy_net, gain)

    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_differentiable_reward_toggle_changes_gradient():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    y_obs, design_obs, theta0 = _make_rollout(task, policy_net)
    reward_fn = _spce_reward(task, n_contrastive=16)

    grads, gains = [], []
    for differentiable_reward in (True, False):
        reinforce = Reinforce(
            task, reward_fn=reward_fn, differentiable_reward=differentiable_reward
        )
        torch.manual_seed(1)  # identical contrastive draws
        gain = reinforce.estimate_gain(y_obs, design_obs, theta0)
        gains.append(gain)
        grads.append(_policy_grad(policy_net, gain))

    assert torch.allclose(gains[0], gains[1])  # same value...
    assert not torch.allclose(grads[0], grads[1])  # ...different gradient


def test_custom_reward_changes_value():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    y_obs, design_obs, theta0 = _make_rollout(task, policy_net)

    def constant_reward(y, d, t):
        return torch.full((y.shape[0],), 3.0)

    reinforce = Reinforce(task, reward_fn=constant_reward, differentiable_reward=False)
    gain = reinforce.estimate_gain(y_obs, design_obs, theta0)
    assert gain.item() == pytest.approx(3.0)


def test_leave_one_out_baseline():
    reward = torch.tensor([1.0, 2.0, 6.0])
    expected = torch.tensor([4.0, 3.5, 1.5])
    assert torch.allclose(leave_one_out_baseline(reward), expected)


def test_baseline_does_not_change_value():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    y_obs, design_obs, theta0 = _make_rollout(task, policy_net)
    reward_fn = _spce_reward(task, n_contrastive=16)

    gains = []
    for baseline in (None, "leave_one_out"):
        reinforce = Reinforce(
            task, reward_fn=reward_fn, baseline=baseline, differentiable_reward=True
        )
        torch.manual_seed(1)
        gains.append(reinforce.estimate_gain(y_obs, design_obs, theta0))

    assert torch.allclose(gains[0], gains[1])


def test_rejects_unknown_baseline():
    task = _make_task()
    with pytest.raises(ValueError):
        Reinforce(task, reward_fn=_spce_reward(task), baseline="unknown")


def test_spce_per_sample_gain_matches_estimate_gain():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    y_obs, design_obs, theta0 = _make_rollout(task, policy_net)

    spce = SPCE(task, n_contrastive=16)
    torch.manual_seed(1)
    per_sample = spce.per_sample_gain(y_obs, design_obs, theta0)
    torch.manual_seed(1)
    gain = spce.estimate_gain(y_obs, design_obs, theta0)

    assert per_sample.shape == (theta0.shape[0],)
    assert torch.allclose(per_sample.mean(), gain)


def test_train_step_with_reinforce():
    torch.manual_seed(0)
    task = _make_task()
    cfg = SimpleNamespace(
        n_outer=4, grad_clip=1.0, objective=SimpleNamespace(name="reinforce", n_inner=4)
    )
    policy_net = _make_design_net(task)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    objective = build_objective(cfg, task)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    trainer = Trainer(task, designer, objective, optimizer, cfg)

    params_before = [p.clone() for p in policy_net.parameters()]
    metrics = trainer.train_step()

    assert torch.isfinite(torch.tensor(metrics["gain"]))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before, policy_net.parameters())
    )
