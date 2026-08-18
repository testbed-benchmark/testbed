import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace, FiniteDesignSpace
from testbed.design_nets import (
    CategoricalPolicyNet,
    DeepSetGaussianActor,
    DesignProposal,
    GaussianPolicyNet,
)
from testbed.envs import BEDEnv, ZeroReward
from testbed.tasks import LocationFinding


def make_space():
    return BoxDesignSpace(-1.0, 1.0, event_shape=(2,))


def make_policy():
    return GaussianPolicyNet(make_space(), event_shape_y=(1,))


def set_actor_output(policy, mean, log_std):
    mean = torch.as_tensor(mean, dtype=policy.design_low.dtype)
    log_std = torch.as_tensor(log_std, dtype=policy.design_low.dtype)
    actor = policy.actor
    scaled_log_std = (
        2
        * (log_std - actor.head.log_std_min)
        / (actor.head.log_std_max - actor.head.log_std_min)
        - 1
    )

    with torch.no_grad():
        actor.head.mean_head.weight.zero_()
        actor.head.mean_head.bias.copy_(mean)
        actor.head.log_std_head.weight.zero_()
        actor.head.log_std_head.bias.copy_(torch.atanh(scaled_log_std))


class RecordingGaussianActor(torch.nn.Module):
    def __init__(self, mean, log_std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("log_std", torch.as_tensor(log_std, dtype=torch.float32))
        self.y_shape = None
        self.design_shape = None
        self.mask = None

    def forward(self, y_history, design_history, *, mask=None):
        self.y_shape = y_history.shape
        self.design_shape = design_history.shape
        self.mask = mask
        batch_size = design_history.shape[0]
        return (
            self.mean.expand(batch_size, -1),
            self.log_std.expand(batch_size, -1),
        )


class BadShapeGaussianActor(torch.nn.Module):
    def forward(self, y_history, design_history, *, mask=None):
        batch_size = design_history.shape[0]
        return (
            torch.zeros(batch_size, 1, device=design_history.device),
            torch.zeros(batch_size, 1, device=design_history.device),
        )


def test_tanh_gaussian_policy_proposes_raw_designs_and_log_probs():
    policy = make_policy()
    y_history = torch.randn(5, 3, 1)
    design_history = torch.randn(5, 3, 2)

    proposal = policy.propose(y_history, design_history)

    assert isinstance(proposal, DesignProposal)
    assert proposal.raw_design.shape == (5, 2)
    assert proposal.deterministic_raw_design.shape == (5, 2)
    assert proposal.log_prob.shape == (5,)


def test_tanh_gaussian_policy_non_reparameterized_sample():
    torch.manual_seed(0)
    policy = make_policy()
    y_history = torch.randn(5, 3, 1)
    design_history = torch.randn(5, 3, 2)

    proposal = policy.propose(y_history, design_history, reparameterize_policy=False)

    # Raw design carries no graph; the log prob is the pure score.
    assert not proposal.raw_design.requires_grad
    assert proposal.log_prob.requires_grad


def test_tanh_gaussian_policy_raw_designs_are_approximately_normal():
    torch.manual_seed(0)
    policy = make_policy()
    mean = torch.tensor([0.25, -0.5])
    log_std = torch.tensor([-0.7, -0.1])
    set_actor_output(policy, mean, log_std)

    n = 20_000
    proposal = policy.propose(
        torch.empty(n, 0, 1),
        torch.empty(n, 0, 2),
        reparameterize_policy=True,
    )
    z = (proposal.raw_design - mean) / log_std.exp()

    assert torch.allclose(proposal.raw_design.mean(dim=0), mean, atol=0.03)
    assert torch.allclose(proposal.raw_design.std(dim=0), log_std.exp(), atol=0.03)
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


def test_tanh_gaussian_policy_transformed_designs_follow_box_scale_and_shift():
    torch.manual_seed(0)
    space = BoxDesignSpace(
        torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 7.0]), event_shape=(2,)
    )
    policy = GaussianPolicyNet(space, event_shape_y=(1,))
    mean = torch.tensor([0.25, -0.5])
    log_std = torch.tensor([-0.7, -0.1])
    set_actor_output(policy, mean, log_std)

    n = 2_000
    proposal = policy.propose(torch.empty(n, 0, 1), torch.empty(n, 0, 2))
    design, _ = space.to_design(proposal.raw_design, proposal.log_prob)
    deterministic_design = space.to_design(proposal.deterministic_raw_design)
    expected_deterministic_design = torch.tanh(mean) * torch.tensor(
        [3.0, 2.0]
    ) + torch.tensor([1.0, 5.0])

    assert space.contains(design)
    assert space.contains(deterministic_design)
    assert torch.all(design > space.low)
    assert torch.all(design < space.high)
    assert torch.allclose(
        deterministic_design[0],
        expected_deterministic_design,
    )


def test_tanh_gaussian_policy_normalizes_physical_histories_for_actor():
    actor = DeepSetGaussianActor(event_shape_y=(1,), event_shape_design=(2,))
    encoder_inputs = {}
    actor.encoder.encoder_net.register_forward_pre_hook(
        lambda _module, args: encoder_inputs.setdefault(
            "value", args[0].detach().clone()
        )
    )
    policy = GaussianPolicyNet(
        BoxDesignSpace(-4.0, 4.0, event_shape=(2,)),
        actor=actor,
        normalize_design_history=True,
        outcome_bounds=(-3.0, 10.0),
    )
    y_history = torch.tensor([[[-3.0], [3.5], [10.0]]])
    design_history = torch.tensor([[[-4.0, 4.0], [0.0, 0.0], [4.0, -4.0]]])

    policy(y_history, design_history)

    torch.testing.assert_close(
        encoder_inputs["value"],
        torch.tensor([[[0.0, -1.0, 1.0], [0.5, 0.0, 0.0], [1.0, 1.0, -1.0]]]),
    )


def test_tanh_gaussian_proposal_has_raw_density_and_deterministic_value():
    torch.manual_seed(0)
    policy = make_policy()
    set_actor_output(policy, mean=[0.25, -0.5], log_std=[-0.7, -0.1])
    y_history = torch.empty(4, 0, 1)
    design_history = torch.empty(4, 0, 2)

    proposal = policy.propose(y_history, design_history)
    expected_raw_log_prob = (
        torch.distributions.Normal(
            torch.tensor([0.25, -0.5]), torch.tensor([-0.7, -0.1]).exp()
        )
        .log_prob(proposal.raw_design)
        .sum(-1)
    )

    assert isinstance(proposal, DesignProposal)
    assert proposal.raw_design.shape == (4, 2)
    assert proposal.deterministic_raw_design.shape == (4, 2)
    torch.testing.assert_close(proposal.log_prob, expected_raw_log_prob)
    torch.testing.assert_close(
        proposal.deterministic_raw_design,
        torch.tensor([0.25, -0.5]).expand(4, -1),
    )

    expected_design, expected_log_prob = policy.design_space.to_design(
        proposal.raw_design, proposal.log_prob
    )

    assert policy.design_space.contains(expected_design)
    torch.testing.assert_close(
        expected_log_prob,
        proposal.log_prob - policy.design_space._log_det_jacobian(proposal.raw_design),
    )


def test_tanh_gaussian_policy_accepts_history_tensors_from_observation():
    task = LocationFinding(n_sources=1, dim_theta=2, noise_scale=0.2)
    space = make_space()
    env = BEDEnv(
        task,
        horizon=3,
        design_space=space,
        reward_fn=ZeroReward(),
        batch_shape=(4,),
    )
    policy = GaussianPolicyNet(space, event_shape_y=task.event_shape_y)

    obs, _ = env.reset(seed=0)
    proposal0 = policy.propose(obs.outcomes, obs.designs, mask=obs.mask)
    obs, *_ = env.step(torch.zeros(4, 2))
    proposal1 = policy.propose(obs.outcomes, obs.designs, mask=obs.mask)
    design0 = space.to_design(proposal0.raw_design)
    design1 = space.to_design(proposal1.raw_design)

    assert proposal0.raw_design.shape == (4, 2)
    assert proposal0.log_prob.shape == (4,)
    assert proposal1.raw_design.shape == (4, 2)
    assert proposal1.log_prob.shape == (4,)
    assert space.contains(design0)
    assert space.contains(design1)


def test_tanh_gaussian_policy_supports_scalar_and_multidim_batch_shapes():
    policy = make_policy()

    scalar_proposal = policy.propose(torch.empty(0, 1), torch.empty(0, 2))
    assert scalar_proposal.raw_design.shape == (2,)
    assert scalar_proposal.log_prob.shape == ()

    y_history = torch.randn(2, 3, 4, 1)
    design_history = torch.randn(2, 3, 4, 2)
    proposal = policy.propose(y_history, design_history)

    assert proposal.raw_design.shape == (2, 3, 2)
    assert proposal.log_prob.shape == (2, 3)


def test_tanh_gaussian_policy_allows_gradient_flow():
    policy = make_policy()
    y_history = torch.randn(6, 2, 1)
    design_history = torch.randn(6, 2, 2)

    proposal = policy.propose(y_history, design_history, reparameterize_policy=True)
    design, log_prob = policy.design_space.to_design(
        proposal.raw_design, proposal.log_prob
    )
    loss = design.sum() + log_prob.sum()
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


@pytest.mark.parametrize(
    ("policy_cls", "design_space"),
    [
        (GaussianPolicyNet, BoxDesignSpace(-1.0, 1.0, event_shape=(2, 2))),
        (
            CategoricalPolicyNet,
            FiniteDesignSpace(torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)),
        ),
    ],
)
def test_default_stochastic_actors_support_multidimensional_outcome_events(
    policy_cls, design_space
):
    event_shape_y = (2, 3)
    policy = policy_cls(design_space, event_shape_y=event_shape_y)
    outcomes = torch.randn(4, 5, *event_shape_y)
    designs = torch.randn(4, 5, 2, 2)

    proposal = policy.propose(outcomes, designs)

    assert proposal.raw_design.shape == (4, 2, 2)


def test_tanh_gaussian_policy_rejects_invalid_mask_shape():
    policy = make_policy()

    with pytest.raises(ValueError, match="mask must have shape"):
        policy.propose(
            torch.randn(2, 3, 1),
            torch.randn(2, 3, 2),
            mask=torch.ones(2, 2, dtype=torch.bool),
        )


def test_tanh_gaussian_policy_rejects_invalid_actor_output_shape():
    policy = GaussianPolicyNet(make_space(), actor=BadShapeGaussianActor())

    with pytest.raises(ValueError, match="actor mean"):
        policy.propose(torch.randn(2, 3, 1), torch.randn(2, 3, 2))
