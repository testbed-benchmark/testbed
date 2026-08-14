"""Tests for the REINFORCE gradient estimator."""

from types import SimpleNamespace

import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace
from testbed.design_nets import DeepSetPolicyNet, GaussianPolicyNet
from testbed.design_objectives import SPCE, Reinforce, SPCEReinforce
from testbed.design_objectives.reinforce import leave_one_out_baseline
from testbed.designers import Rollout, build_policy_designer
from testbed.methods.dad import Trainer, build_objective
from testbed.tasks import HyperbolicTemporalDiscounting


class _ValueFnReinforce(Reinforce):
    """Test-only Reinforce wrapping an arbitrary calculate_value(rollout) callable."""

    def __init__(self, task, value_fn, stochastic_design_policy, baseline=None):
        super().__init__(task, stochastic_design_policy, baseline)
        self._value_fn = value_fn

    def calculate_value(self, rollout):
        return self._value_fn(rollout)


def _make_task():
    return HyperbolicTemporalDiscounting()


def _make_design_net(task):
    # task.design_space (an HTDDesignSpace) maps raw (near-zero at init)
    # outputs to non-saturating (D_1, R_0); no wrapper needed.
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=task.event_shape_design,
    )


def _make_stochastic_net(task):
    # Box over valid (D_1, R_0); centered near k * D_1 ~ 1 so it is non-saturating.
    space = BoxDesignSpace(low=[5.0, 20.0], high=[40.0, 80.0], event_shape=(2,))
    return GaussianPolicyNet(space, event_shape_y=task.event_shape_y)


def _make_rollout(task, policy_net, b=8, T=3, **kwargs):
    designer = build_policy_designer(task, T=T, design_net=policy_net, **kwargs)
    theta0 = task.sample_prior(b)
    return designer.rollout(theta0)


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
    cfg = SimpleNamespace(
        objective=SimpleNamespace(
            name="reinforce", n_inner=4, baseline=None, requires_policy_log_probs=False
        )
    )

    deterministic_obj = build_objective(cfg, task)
    assert isinstance(deterministic_obj, Reinforce)
    assert not deterministic_obj.stochastic_design_policy

    cfg.objective.requires_policy_log_probs = True
    stochastic_obj = build_objective(cfg, task)
    assert stochastic_obj.stochastic_design_policy


def test_rejects_unknown_baseline():
    task = _make_task()
    spce = SPCE(task, n_contrastive=4)
    with pytest.raises(ValueError):
        _ValueFnReinforce(task, spce.per_sample_gain, True, baseline="unknown")


# ============================================================================
# Deterministic policy (pathwise / DAD Eq. 16)
# ============================================================================


def test_deterministic_value_equals_mean_spce_gain():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    theta0 = task.sample_prior(8)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(theta0)

    spce = SPCE(task, n_contrastive=16)
    reinforce = _ValueFnReinforce(task, spce.per_sample_gain, False)

    torch.manual_seed(1)  # contrastive draws
    gain_reinforce = reinforce.estimate_gain(rollout)
    torch.manual_seed(1)
    gain_spce = spce.estimate_gain(rollout)

    assert torch.allclose(gain_reinforce, gain_spce)


@pytest.mark.parametrize("baseline", [None, "leave_one_out"])
def test_deterministic_matches_dad_eq16(baseline):
    # Regression against the previous estimator: value == mean SPCE gain and
    # gradient == pathwise gain gradient + score-function term (DAD Eq. 16).
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    theta0 = task.sample_prior(8)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(theta0)

    spce = SPCE(task, n_contrastive=16)
    reinforce = _ValueFnReinforce(
        task,
        spce.per_sample_gain,
        False,
        baseline=baseline,
    )

    torch.manual_seed(1)
    gain = reinforce.estimate_gain(rollout)

    torch.manual_seed(1)
    reward = spce.per_sample_gain(rollout)
    if baseline == "leave_one_out":
        baseline_value = leave_one_out_baseline(reward.detach())
    else:
        baseline_value = 0.0
    score = task.history_logp(rollout.y_history, rollout.design_history, theta0)
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
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(theta0)

    def constant_value(rollout):
        return rollout.y_history.new_full((rollout.y_history.shape[0],), 3.0)

    reinforce = _ValueFnReinforce(task, constant_value, False)
    gain = reinforce.estimate_gain(rollout)
    assert gain.item() == pytest.approx(3.0)


# ============================================================================
# Stochastic policy (score-function REINFORCE)
# ============================================================================


def test_stochastic_rollout_returns_policy_log_probs():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    rollout = _make_rollout(task, policy_net, b=8, T=3)

    assert rollout.extra["policy_log_probs"].shape == (8, 3)
    assert rollout.extra["policy_log_probs"].requires_grad
    # score-function sampling: designs carry no graph
    assert not rollout.design_history.requires_grad


def test_deterministic_rollout_has_no_policy_log_probs():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_design_net(task)
    rollout = _make_rollout(task, policy_net)
    assert "policy_log_probs" not in rollout.extra


def test_stochastic_requires_policy_log_probs():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    rollout = _make_rollout(task, policy_net)
    rollout_without_log_probs = Rollout(
        theta=rollout.theta,
        y_history=rollout.y_history,
        design_history=rollout.design_history,
    )

    reinforce = SPCEReinforce(task, n_contrastive=4, stochastic_design_policy=True)
    with pytest.raises(ValueError):
        reinforce.estimate_gain(rollout_without_log_probs)


def test_stochastic_value_equals_mean_spce_gain():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    theta0 = task.sample_prior(8)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(theta0)

    spce = SPCE(task, n_contrastive=16)
    reinforce = _ValueFnReinforce(task, spce.per_sample_gain, True)

    torch.manual_seed(1)
    gain = reinforce.estimate_gain(rollout)
    torch.manual_seed(1)
    gain_spce = spce.estimate_gain(rollout)

    assert torch.allclose(gain, gain_spce)


def test_stochastic_grad_comes_from_score_term():
    # The value is detached, so any policy gradient must come from the
    # score-function term.
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    theta0 = task.sample_prior(8)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(theta0)

    spce = SPCE(task, n_contrastive=4)
    reinforce = _ValueFnReinforce(task, spce.per_sample_gain, True)
    gain = reinforce.estimate_gain(rollout)
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
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(theta0)

    def constant_value(rollout):
        return rollout.y_history.new_full((rollout.y_history.shape[0],), 3.0)

    reinforce = _ValueFnReinforce(
        task,
        constant_value,
        True,
        baseline="leave_one_out",
    )
    gain = reinforce.estimate_gain(rollout)
    grad = _policy_grad(policy_net, gain)

    assert gain.item() == pytest.approx(3.0)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_stochastic_baseline_does_not_change_value():
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    theta0 = task.sample_prior(8)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(theta0)

    spce = SPCE(task, n_contrastive=16)
    gains = []
    for baseline in (None, "leave_one_out"):
        reinforce = _ValueFnReinforce(
            task,
            spce.per_sample_gain,
            True,
            baseline=baseline,
        )
        torch.manual_seed(1)
        gains.append(reinforce.estimate_gain(rollout))

    assert torch.allclose(gains[0], gains[1])


def test_stochastic_reparameterized_pathwise_grad():
    # reparameterize_policy=True: designs carry the graph and the estimator is
    # pathwise, so no policy log probs are needed.
    torch.manual_seed(0)
    task = _make_task()
    policy_net = _make_stochastic_net(task)
    theta0 = task.sample_prior(8)
    designer = build_policy_designer(
        task, T=3, design_net=policy_net, reparameterize_policy=True
    )
    rollout = designer.rollout(theta0)
    assert rollout.design_history.requires_grad

    spce = SPCE(task, n_contrastive=4)
    reinforce = _ValueFnReinforce(task, spce.per_sample_gain, False)
    assert not reinforce.stochastic_design_policy

    gain = reinforce.estimate_gain(rollout)
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
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    rollout = designer.rollout(theta0)

    spce = SPCE(task, n_contrastive=16)
    torch.manual_seed(1)
    per_sample = spce.per_sample_gain(rollout)
    torch.manual_seed(1)
    gain = spce.estimate_gain(rollout)

    assert per_sample.shape == (theta0.shape[0],)
    assert torch.allclose(per_sample.mean(), gain)


@pytest.mark.parametrize(
    ("net_factory", "requires_policy_log_probs"),
    [(_make_design_net, False), (_make_stochastic_net, True)],
)
def test_train_step_with_reinforce(net_factory, requires_policy_log_probs):
    torch.manual_seed(0)
    task = _make_task()
    policy_net = net_factory(task)
    designer = build_policy_designer(task, T=3, design_net=policy_net)
    cfg = SimpleNamespace(
        n_outer=4,
        grad_clip=1.0,
        objective=SimpleNamespace(
            name="reinforce",
            n_inner=4,
            baseline=None,
            requires_policy_log_probs=requires_policy_log_probs,
        ),
    )
    objective = build_objective(cfg, task)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    trainer = Trainer(task, designer, objective, optimizer, cfg)

    params_before = [param.clone() for param in policy_net.parameters()]
    metrics = trainer.train_step()

    assert torch.isfinite(torch.tensor(metrics["gain"]))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(params_before, policy_net.parameters())
    )
