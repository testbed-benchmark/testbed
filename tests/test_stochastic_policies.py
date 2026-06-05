import pytest
import torch

from testbed.design_nets import PolicySample, StochasticDesignNet, TanhGaussianPolicyNet
from testbed.envs import (
    BEDEnv,
    BoxDesignSpace,
    CategoricalDesignSpace,
    PaddedHistoryObservation,
    ZeroReward,
)
from testbed.tasks import LocationFinding


def make_space():
    return BoxDesignSpace(-1.0, 1.0, event_shape=(2,))


def make_policy():
    return TanhGaussianPolicyNet(make_space(), dim_y=1, dim_h=32, dim_emb=16)


def test_tanh_gaussian_policy_samples_bounded_actions_and_log_probs():
    policy = make_policy()
    y_history = torch.randn(5, 3, 1)
    design_history = torch.randn(5, 3, 2)

    sample = policy.sample(y_history, design_history)

    assert isinstance(sample, PolicySample)
    assert sample.action.shape == (5, 2)
    assert sample.deterministic_action.shape == (5, 2)
    assert sample.raw_action.shape == (5, 2)
    assert sample.log_prob.shape == (5,)
    assert make_space().contains(sample.action)
    assert make_space().contains(sample.deterministic_action)


def test_tanh_gaussian_forward_returns_deterministic_action():
    policy = make_policy()
    y_history = torch.randn(4, 2, 1)
    design_history = torch.randn(4, 2, 2)

    sample = policy.sample(y_history, design_history, deterministic=True)
    forward_action = policy(y_history, design_history)

    assert torch.equal(forward_action, sample.action)
    assert torch.equal(forward_action, sample.deterministic_action)


def test_tanh_gaussian_policy_accepts_padded_observation_dict():
    task = LocationFinding(n_sources=1, dim_theta=2, noise_scale=0.2)
    space = make_space()
    env = BEDEnv(
        task,
        horizon=3,
        design_space=space,
        reward_fn=ZeroReward(),
        observation_fn=PaddedHistoryObservation(),
        batch_shape=(4,),
    )
    policy = TanhGaussianPolicyNet(space, dim_y=1, dim_h=32, dim_emb=16)

    obs, _ = env.reset(seed=0)
    sample0 = policy.sample_from_observation(obs)
    obs, *_ = env.step(torch.zeros(4, 2))
    sample1 = policy.sample_from_observation(obs)

    assert sample0.action.shape == (4, 2)
    assert sample0.log_prob.shape == (4,)
    assert sample1.action.shape == (4, 2)
    assert sample1.log_prob.shape == (4,)
    assert space.contains(sample0.action)
    assert space.contains(sample1.action)


def test_tanh_gaussian_policy_supports_scalar_and_multidim_batch_shapes():
    policy = make_policy()

    scalar_sample = policy.sample(torch.empty(0, 1), torch.empty(0, 2))
    assert scalar_sample.action.shape == (2,)
    assert scalar_sample.log_prob.shape == ()

    y_history = torch.randn(2, 3, 4, 1)
    design_history = torch.randn(2, 3, 4, 2)
    sample = policy.sample(y_history, design_history)

    assert sample.action.shape == (2, 3, 2)
    assert sample.log_prob.shape == (2, 3)


def test_tanh_gaussian_policy_allows_gradient_flow():
    policy = make_policy()
    y_history = torch.randn(6, 2, 1)
    design_history = torch.randn(6, 2, 2)

    sample = policy.sample(y_history, design_history)
    loss = sample.action.sum() + sample.log_prob.sum()
    loss.backward()

    grads = [p.grad for p in policy.parameters() if p.requires_grad]
    assert any(grad is not None and torch.any(grad != 0) for grad in grads)


def test_tanh_gaussian_policy_rejects_non_box_design_space():
    with pytest.raises(TypeError, match="BoxDesignSpace"):
        TanhGaussianPolicyNet(CategoricalDesignSpace(3))


def test_stochastic_design_net_sample_is_abstract_extension():
    assert issubclass(TanhGaussianPolicyNet, StochasticDesignNet)
