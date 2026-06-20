import pytest
import torch

from testbed.design_nets import (
    DeepSetGaussianActor,
    GaussianPolicyActor,
    PolicySample,
    TanhGaussianPolicyNet,
)
from testbed.envs import (
    BEDEnv,
    BoxDesignSpace,
    CategoricalDesignSpace,
    ZeroReward,
)
from testbed.tasks import LocationFinding


def make_space():
    return BoxDesignSpace(-1.0, 1.0, event_shape=(2,))


def make_policy():
    return TanhGaussianPolicyNet(make_space())


def set_actor_output(policy, mean, log_std):
    mean = torch.as_tensor(mean, dtype=policy.action_scale.dtype)
    log_std = torch.as_tensor(log_std, dtype=policy.action_scale.dtype)
    actor = policy.actor
    scaled_log_std = (
        2 * (log_std - actor.log_std_min) / (actor.log_std_max - actor.log_std_min) - 1
    )

    with torch.no_grad():
        actor.mean_head.weight.zero_()
        actor.mean_head.bias.copy_(mean)
        actor.log_std_head.weight.zero_()
        actor.log_std_head.bias.copy_(torch.atanh(scaled_log_std))


class RecordingGaussianActor(GaussianPolicyActor):
    def __init__(self, mean, log_std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("log_std", torch.as_tensor(log_std, dtype=torch.float32))
        self.y_shape = None
        self.design_shape = None
        self.y_history = None
        self.design_history = None
        self.mask = None

    def forward(self, y_history, design_history, *, mask=None):
        self.y_shape = y_history.shape
        self.design_shape = design_history.shape
        self.y_history = y_history.detach().clone()
        self.design_history = design_history.detach().clone()
        self.mask = mask
        batch_size = design_history.shape[0]
        return (
            self.mean.expand(batch_size, -1),
            self.log_std.expand(batch_size, -1),
        )


class BadShapeGaussianActor(GaussianPolicyActor):
    def forward(self, y_history, design_history, *, mask=None):
        batch_size = design_history.shape[0]
        return (
            torch.zeros(batch_size, 1, device=design_history.device),
            torch.zeros(batch_size, 1, device=design_history.device),
        )


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


def test_tanh_gaussian_policy_raw_actions_are_approximately_normal():
    torch.manual_seed(0)
    policy = make_policy()
    mean = torch.tensor([0.25, -0.5])
    log_std = torch.tensor([-0.7, -0.1])
    set_actor_output(policy, mean, log_std)

    n = 20_000
    sample = policy.sample(torch.empty(n, 0, 1), torch.empty(n, 0, 2))
    z = (sample.raw_action - mean) / log_std.exp()

    assert torch.allclose(sample.raw_action.mean(dim=0), mean, atol=0.03)
    assert torch.allclose(sample.raw_action.std(dim=0), log_std.exp(), atol=0.03)
    assert torch.allclose(
        (z.abs() < 1).float().mean(dim=0),
        torch.full((2,), 0.6827),
        atol=0.025,
    )
    assert torch.allclose(
        (z.abs() < 2).float().mean(dim=0),
        torch.full((2,), 0.9545),
        atol=0.015,
    )


def test_tanh_gaussian_policy_transformed_actions_follow_box_scale_and_shift():
    torch.manual_seed(0)
    space = BoxDesignSpace(torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 7.0]))
    policy = TanhGaussianPolicyNet(space)
    mean = torch.tensor([0.25, -0.5])
    log_std = torch.tensor([-0.7, -0.1])
    set_actor_output(policy, mean, log_std)

    n = 2_000
    sample = policy.sample(torch.empty(n, 0, 1), torch.empty(n, 0, 2))
    expected_deterministic_action = torch.tanh(mean) * torch.tensor(
        [3.0, 2.0]
    ) + torch.tensor([1.0, 5.0])

    assert torch.allclose(policy.action_scale, torch.tensor([3.0, 2.0]))
    assert torch.allclose(policy.action_bias, torch.tensor([1.0, 5.0]))
    assert space.contains(sample.action)
    assert space.contains(sample.deterministic_action)
    assert torch.all(sample.action > space.low)
    assert torch.all(sample.action < space.high)
    assert torch.allclose(
        sample.deterministic_action[0],
        expected_deterministic_action,
    )


def test_tanh_gaussian_policy_normalizes_physical_histories_for_actor():
    actor = RecordingGaussianActor(mean=[0.0, 0.0], log_std=[-1.0, -1.0])
    policy = TanhGaussianPolicyNet(
        BoxDesignSpace(-4.0, 4.0, event_shape=(2,)),
        actor=actor,
        normalize_design_history=True,
        outcome_bounds=(-3.0, 10.0),
    )
    y_history = torch.tensor([[[-3.0], [3.5], [10.0]]])
    design_history = torch.tensor([[[-4.0, 4.0], [0.0, 0.0], [4.0, -4.0]]])

    policy.sample(y_history, design_history, deterministic=True)

    torch.testing.assert_close(
        actor.y_history,
        torch.tensor([[[0.0], [0.5], [1.0]]]),
    )
    torch.testing.assert_close(
        actor.design_history,
        torch.tensor([[[-1.0, 1.0], [0.0, 0.0], [1.0, -1.0]]]),
    )


def test_tanh_gaussian_forward_returns_deterministic_action():
    policy = make_policy()
    y_history = torch.randn(4, 2, 1)
    design_history = torch.randn(4, 2, 2)

    sample = policy.sample(y_history, design_history, deterministic=True)
    forward_action = policy(y_history, design_history)

    assert torch.equal(forward_action, sample.action)
    assert torch.equal(forward_action, sample.deterministic_action)


def test_tanh_gaussian_policy_accepts_history_observation():
    task = LocationFinding(n_sources=1, dim_theta=2, noise_scale=0.2)
    space = make_space()
    env = BEDEnv(
        task,
        horizon=3,
        design_space=space,
        reward_fn=ZeroReward(),
        batch_shape=(4,),
    )
    policy = TanhGaussianPolicyNet(space)

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


def test_tanh_gaussian_policy_uses_deepset_actor_by_default():
    policy = make_policy()

    assert isinstance(policy.actor, DeepSetGaussianActor)
    assert not hasattr(policy, "encoder_net")
    assert not hasattr(policy, "actor_net")
    assert not hasattr(policy, "mean_head")
    assert not hasattr(policy, "log_std_head")


def test_tanh_gaussian_policy_accepts_custom_actor_and_forwards_mask():
    actor = RecordingGaussianActor(mean=[0.2, -0.1], log_std=[-0.5, -0.25])
    policy = TanhGaussianPolicyNet(make_space(), actor=actor)
    y_history = torch.randn(2, 3, 4, 1)
    design_history = torch.randn(2, 3, 4, 2)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[:, :, -1] = False

    sample = policy.sample(
        y_history,
        design_history,
        mask=mask,
        deterministic=True,
    )

    assert actor.y_shape == (6, 4, 1)
    assert actor.design_shape == (6, 4, 2)
    assert actor.mask.shape == (6, 4)
    assert torch.equal(actor.mask, mask.reshape(6, 4))
    assert sample.action.shape == (2, 3, 2)
    assert sample.log_prob.shape == (2, 3)
    assert torch.allclose(
        sample.action[0, 0],
        torch.tanh(torch.tensor([0.2, -0.1])),
    )


def test_tanh_gaussian_policy_rejects_invalid_mask_shape():
    policy = make_policy()

    with pytest.raises(ValueError, match="mask must have shape"):
        policy.sample(
            torch.randn(2, 3, 1),
            torch.randn(2, 3, 2),
            mask=torch.ones(2, 2, dtype=torch.bool),
        )


def test_tanh_gaussian_policy_rejects_invalid_actor_output_shape():
    policy = TanhGaussianPolicyNet(make_space(), actor=BadShapeGaussianActor())

    with pytest.raises(ValueError, match="actor mean"):
        policy.sample(torch.randn(2, 3, 1), torch.randn(2, 3, 2))
