"""Tests for the REINFORCE gradient estimator."""

from types import SimpleNamespace

import pytest
import torch

from testbed.design_nets import DeepSetPolicyNet, TanhGaussianPolicyNet
from testbed.design_objectives import SPCE, Reinforce
from testbed.design_objectives.reinforce import leave_one_out_baseline
from testbed.designers import PolicyDesigner
from testbed.envs import BoxDesignSpace
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


def _make_stochastic_net(task):
    space = BoxDesignSpace(-1.0, 1.0, event_shape=tuple(task.event_shape_design))
    return TanhGaussianPolicyNet(space)


def _make_rollout(task, policy_net, b=8, T=3, return_policy_log_probs=False, **kwargs):
    designer = PolicyDesigner(task, T=T, policy_net=policy_net, **kwargs)
    theta0 = task.sample_prior(b)
    return designer.rollout(theta0, return_policy_log_probs=return_policy_log_probs)


def _policy_grad(policy_net, gain):
    grads = torch.autograd.grad(
        gain, list(policy_net.parameters()), retain_graph=True, allow_unused=True
    )
    return torch.cat(
        [
            (grad if grad is not None else torch.zeros_like(param)).flatten()
            for grad, param in zip(grads, policy_net.parameters())
        ]
    )


# ============================================================================
# Construction and validation
# ============================================================================


def test_build_objective_creates_reinforce():
    task = _make_task()
    cfg = SimpleNamespace(objective=SimpleNamespace(name="reinforce", n_inner=4))

    det_designer = PolicyDesigner(task, T=3, policy_net=_make_design_net(task))
    deterministic_obj = build_objective(cfg, task, det_designer)
    assert isinstance(deterministic_obj, Reinforce)
    assert not deterministic_obj.requires_policy_log_probs

    stoch_designer = PolicyDesigner(task, T=3, policy_net=_make_stochastic_net(task))
    stochastic_obj = build_objective(cfg, task, stoch_designer)
    assert stochastic_obj.requires_policy_log_probs


def test_rejects_unknown_baseline():
    task = _make_task()
    spce = SPCE(task, n_contrastive=4)
    designer = PolicyDesigner(task, T=3, policy_net=_make_stochastic_net(task))
    with pytest.raises(ValueError):
        Reinforce(task, spce.per_sample_gain, designer, baseline="unknown")


# ============================================================================
# Deterministic policy (pathwise / DAD Eq. 16)
# ============================================================================


def test_deterministic_value_equals_mean_spce_gain():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    theta0 = task.sample_prior(8)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    y_obs, design_obs = designer.rollout(theta0)

    spce = SPCE(task, n_contrastive=16)
    reinforce = Reinforce(task, spce.per_sample_gain, designer)

    torch.manual_seed(1)  # contrastive draws
    gain_reinforce = reinforce.estimate_gain(y_obs, design_obs, theta0)
    torch.manual_seed(1)
    gain_spce = spce.estimate_gain(y_obs, design_obs, theta0)

    assert torch.allclose(gain_reinforce, gain_spce)


@pytest.mark.parametrize("baseline", [None, "leave_one_out"])
def test_deterministic_matches_dad_eq16(baseline):
    # Regression against the previous estimator: value == mean SPCE gain and
    # gradient == pathwise gain gradient + score-function term (DAD Eq. 16).
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    theta0 = task.sample_prior(8)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    y_obs, design_obs = designer.rollout(theta0)

    spce = SPCE(task, n_contrastive=16)
    reinforce = Reinforce(task, spce.per_sample_gain, designer, baseline=baseline)

    torch.manual_seed(1)
    gain = reinforce.estimate_gain(y_obs, design_obs, theta0)

    torch.manual_seed(1)
    reward = spce.per_sample_gain(y_obs, design_obs, theta0)
    if baseline == "leave_one_out":
        baseline_value = leave_one_out_baseline(reward.detach())
    else:
        baseline_value = 0.0
    score = task.history_logp(y_obs, design_obs, theta0)
    reference = (
        reward + (reward.detach() - baseline_value) * (score - score.detach())
    ).mean()

    assert torch.allclose(gain, reference)
    assert torch.allclose(
        _policy_grad(policy_net, gain), _policy_grad(policy_net, reference)
    )


def test_deterministic_constant_value():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    theta0 = task.sample_prior(8)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    y_obs, design_obs = designer.rollout(theta0)

    def constant_value(y_history, design_history, theta_0):
        return y_history.new_full((y_history.shape[0],), 3.0)

    reinforce = Reinforce(task, constant_value, designer)
    gain = reinforce.estimate_gain(y_obs, design_obs, theta0)
    assert gain.item() == pytest.approx(3.0)


# ============================================================================
# Stochastic policy (score-function REINFORCE)
# ============================================================================


def test_stochastic_rollout_returns_policy_log_probs():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    y_obs, design_obs, policy_log_probs = _make_rollout(
        task, policy_net, b=8, T=3, return_policy_log_probs=True
    )

    assert policy_log_probs.shape == (8, 3)
    assert policy_log_probs.requires_grad
    # score-function sampling: designs carry no graph
    assert not design_obs.requires_grad


def test_deterministic_rollout_rejects_policy_log_probs():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    with pytest.raises(RuntimeError):
        _make_rollout(task, policy_net, return_policy_log_probs=True)


def test_stochastic_requires_policy_log_probs():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    y_obs, design_obs, _ = _make_rollout(task, policy_net, return_policy_log_probs=True)
    theta0 = task.sample_prior(y_obs.shape[0])

    spce = SPCE(task, n_contrastive=4)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    reinforce = Reinforce(task, spce.per_sample_gain, designer)
    with pytest.raises(ValueError):
        reinforce.estimate_gain(y_obs, design_obs, theta0)


def test_stochastic_value_equals_mean_spce_gain():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    theta0 = task.sample_prior(8)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    y_obs, design_obs, policy_log_probs = designer.rollout(
        theta0, return_policy_log_probs=True
    )

    spce = SPCE(task, n_contrastive=16)
    reinforce = Reinforce(task, spce.per_sample_gain, designer)

    torch.manual_seed(1)
    gain = reinforce.estimate_gain(y_obs, design_obs, theta0, policy_log_probs)
    torch.manual_seed(1)
    gain_spce = spce.estimate_gain(y_obs, design_obs, theta0)

    assert torch.allclose(gain, gain_spce)


def test_stochastic_grad_comes_from_score_term():
    # The value is detached, so any policy gradient must come from the
    # score-function term.
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    theta0 = task.sample_prior(8)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    y_obs, design_obs, policy_log_probs = designer.rollout(
        theta0, return_policy_log_probs=True
    )

    spce = SPCE(task, n_contrastive=4)
    reinforce = Reinforce(task, spce.per_sample_gain, designer)
    gain = reinforce.estimate_gain(y_obs, design_obs, theta0, policy_log_probs)
    grad = _policy_grad(policy_net, gain)

    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_stochastic_constant_value_loo_baseline_zero_grad():
    # With a constant value the leave-one-out baseline cancels the score term
    # exactly; nonzero gradients here would indicate a reparameterization leak.
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    theta0 = task.sample_prior(8)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    y_obs, design_obs, policy_log_probs = designer.rollout(
        theta0, return_policy_log_probs=True
    )

    def constant_value(y_history, design_history, theta_0):
        return y_history.new_full((y_history.shape[0],), 3.0)

    reinforce = Reinforce(task, constant_value, designer, baseline="leave_one_out")
    gain = reinforce.estimate_gain(y_obs, design_obs, theta0, policy_log_probs)
    grad = _policy_grad(policy_net, gain)

    assert gain.item() == pytest.approx(3.0)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_stochastic_baseline_does_not_change_value():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    theta0 = task.sample_prior(8)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    y_obs, design_obs, policy_log_probs = designer.rollout(
        theta0, return_policy_log_probs=True
    )

    spce = SPCE(task, n_contrastive=16)
    gains = []
    for baseline in (None, "leave_one_out"):
        reinforce = Reinforce(task, spce.per_sample_gain, designer, baseline=baseline)
        torch.manual_seed(1)
        gains.append(
            reinforce.estimate_gain(y_obs, design_obs, theta0, policy_log_probs)
        )

    assert torch.allclose(gains[0], gains[1])


def test_stochastic_reparameterized_pathwise_grad():
    # reparameterize_policy=True: designs carry the graph and the estimator is
    # pathwise, so no policy log probs are needed.
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    theta0 = task.sample_prior(8)
    designer = PolicyDesigner(
        task, T=3, policy_net=policy_net, reparameterize_policy=True
    )
    y_obs, design_obs = designer.rollout(theta0)
    assert design_obs.requires_grad

    spce = SPCE(task, n_contrastive=4)
    reinforce = Reinforce(task, spce.per_sample_gain, designer)
    assert not reinforce.requires_policy_log_probs

    gain = reinforce.estimate_gain(y_obs, design_obs, theta0)
    grad = _policy_grad(policy_net, gain)

    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


# ============================================================================
# Shared pieces and training
# ============================================================================


def test_leave_one_out_baseline():
    reward = torch.tensor([1.0, 2.0, 6.0])
    expected = torch.tensor([4.0, 3.5, 1.5])
    assert torch.allclose(leave_one_out_baseline(reward), expected)


def test_spce_per_sample_gain_matches_estimate_gain():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    theta0 = task.sample_prior(8)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    y_obs, design_obs = designer.rollout(theta0)

    spce = SPCE(task, n_contrastive=16)
    torch.manual_seed(1)
    per_sample = spce.per_sample_gain(y_obs, design_obs, theta0)
    torch.manual_seed(1)
    gain = spce.estimate_gain(y_obs, design_obs, theta0)

    assert per_sample.shape == (theta0.shape[0],)
    assert torch.allclose(per_sample.mean(), gain)


@pytest.mark.parametrize("net_factory", [_make_design_net, _make_stochastic_net])
def test_train_step_with_reinforce(net_factory):
    torch.manual_seed(0)
    task = _make_task()
    cfg = SimpleNamespace(
        n_outer=4, grad_clip=1.0, objective=SimpleNamespace(name="reinforce", n_inner=4)
    )
    policy_net = net_factory(task)
    designer = PolicyDesigner(task, T=3, policy_net=policy_net)
    objective = build_objective(cfg, task, designer)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    trainer = Trainer(task, designer, objective, optimizer, cfg)

    params_before = [param.clone() for param in policy_net.parameters()]
    metrics = trainer.train_step()

    assert torch.isfinite(torch.tensor(metrics["gain"]))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before, policy_net.parameters())
    )
