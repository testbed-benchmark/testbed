"""Tests for the SAC location-finding example."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

from testbed.design_nets import (
    DeepSetGaussianActor,
    RLBOEDQNetwork,
    TanhGaussianPolicyNet,
)
from testbed.designers import PolicyDesigner
from testbed.envs import BoxDesignSpace
from testbed.methods import RLBOED, FitResult, RLBOEDConfig
from testbed.methods.rlboed import _CompactEpisodeReplayBuffer
from testbed.tasks import LocationFinding

EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "sac_location_finding.py"
SPEC = importlib.util.spec_from_file_location("sac_location_finding", EXAMPLE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
sac_location_finding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sac_location_finding)
SoftQNetwork = RLBOEDQNetwork


def test_example_exposes_released_source_location_seeds():
    assert sac_location_finding.PAPER_SEEDS == (
        373693,
        943929,
        675273,
        79387,
        508137,
        557390,
        756177,
        155183,
        262598,
        572185,
    )
    assert sac_location_finding.Args.seed == sac_location_finding.PAPER_SEEDS[0]


def _linear_shapes(module):
    return [
        (layer.in_features, layer.out_features)
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]


def test_actor_defaults_match_released_rl_boed_architecture():
    """The default actor reproduces the released source-location policy."""
    actor = DeepSetGaussianActor(dim_y=1, dim_design=2)

    assert _linear_shapes(actor.encoder_net) == [(3, 128), (128, 128), (128, 64)]
    assert _linear_shapes(actor.actor_net) == [(64, 128), (128, 128)]
    assert actor.mean_head.in_features == 128
    assert actor.log_std_head.in_features == 128
    assert actor.log_std_min == -20.0
    assert actor.log_std_max == 0.0
    assert actor.init_std == pytest.approx(3**-0.5)
    assert not hasattr(actor, "null_enc")
    for module in (actor.encoder_net, actor.actor_net, actor.mean_head):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))


def test_actor_initializes_requested_standard_deviation():
    actor = DeepSetGaussianActor(dim_y=1, dim_design=2, init_std=0.4)
    y_history = torch.randn(3, 4, 1)
    design_history = torch.randn(3, 4, 2)

    _, log_std = actor(y_history, design_history)

    torch.testing.assert_close(log_std.exp(), torch.full_like(log_std, 0.4))
    assert torch.count_nonzero(actor.log_std_head.weight) == 0
    assert actor.log_std_head.weight.requires_grad


@pytest.mark.parametrize("init_std", [0.0, -0.1, 1.0, 1e-10])
def test_actor_rejects_initial_std_outside_log_std_bounds(init_std):
    with pytest.raises(ValueError, match="init_std"):
        DeepSetGaussianActor(dim_y=1, dim_design=2, init_std=init_std)


def test_critic_defaults_match_released_rl_boed_architecture():
    """The default critic uses the paper's summary and emitter dimensions."""
    critic = SoftQNetwork(
        dim_y=1,
        dim_design=2,
        design_low=-4.0,
        design_high=4.0,
        outcome_low=-3.0,
        outcome_high=10.0,
    )

    assert _linear_shapes(critic.encoder_net) == [(3, 128), (128, 128), (128, 64)]
    assert _linear_shapes(critic.emitter_net) == [
        (66, 128),
        (128, 128),
        (128, 1),
    ]
    for layer in critic.modules():
        if isinstance(layer, nn.Linear):
            torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))


def test_soft_q_network_is_permutation_invariant():
    """Permuting valid history entries does not change the Q estimate."""
    critic = SoftQNetwork(
        dim_y=1,
        dim_design=2,
        design_low=-4.0,
        design_high=4.0,
        outcome_low=-3.0,
        outcome_high=10.0,
    )
    observation = {
        "designs": torch.randn(3, 5, 2),
        "outcomes": torch.randn(3, 5, 1),
        "mask": torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
                [True, False, False, False, False],
            ]
        ),
        "t": torch.tensor([3, 5, 1]),
    }
    action = torch.randn(3, 2)
    permutation = torch.tensor([2, 0, 4, 1, 3])
    permuted_observation = {
        "designs": observation["designs"][:, permutation],
        "outcomes": observation["outcomes"][:, permutation],
        "mask": observation["mask"][:, permutation],
        "t": observation["t"],
    }

    actual = critic(observation, action)
    permuted = critic(permuted_observation, action)

    torch.testing.assert_close(actual, permuted)


def test_soft_q_network_ignores_masked_history_entries():
    """Changing padded history entries does not change the Q estimate."""
    critic = SoftQNetwork(
        dim_y=1,
        dim_design=2,
        design_low=-4.0,
        design_high=4.0,
        outcome_low=-3.0,
        outcome_high=10.0,
    )
    observation = {
        "designs": torch.randn(2, 4, 2),
        "outcomes": torch.randn(2, 4, 1),
        "mask": torch.tensor([[True, True, False, False], [True, False, False, False]]),
    }
    modified_observation = {key: value.clone() for key, value in observation.items()}
    modified_observation["designs"][~observation["mask"]] = 1000.0
    modified_observation["outcomes"][~observation["mask"]] = -1000.0
    action = torch.randn(2, 2)

    actual = critic(observation, action)
    modified = critic(modified_observation, action)

    assert actual.shape == (2, 1)
    torch.testing.assert_close(actual, modified)


def test_soft_q_network_uses_reference_history_and_action_coordinates():
    critic = SoftQNetwork(
        dim_y=1,
        dim_design=2,
        design_low=torch.tensor([-2.0, 2.0]),
        design_high=torch.tensor([4.0, 6.0]),
        outcome_low=-3.0,
        outcome_high=10.0,
    )
    inputs = {}

    def record_encoder_input(module, args):
        inputs["encoder"] = args[0].detach().clone()

    def record_emitter_input(module, args):
        inputs["emitter"] = args[0].detach().clone()

    critic.encoder_net.register_forward_pre_hook(record_encoder_input)
    critic.emitter_net.register_forward_pre_hook(record_emitter_input)
    physical_design = torch.tensor([[1.0, 4.0]])
    observation = {
        "designs": physical_design.unsqueeze(1),
        "outcomes": torch.tensor([[[3.5]]]),
        "mask": torch.tensor([[True]]),
    }

    critic(observation, physical_design)

    torch.testing.assert_close(
        inputs["encoder"][..., -2:],
        torch.full((1, 1, 2), 0.5),
    )
    torch.testing.assert_close(
        inputs["emitter"][..., -2:],
        torch.zeros(1, 2),
    )


def test_compact_replay_buffer_reconstructs_padded_transitions(monkeypatch):
    buffer = _CompactEpisodeReplayBuffer(
        buffer_size=3,
        batch_size=3,
        horizon=3,
        event_shape_design=torch.Size([2]),
        event_shape_y=torch.Size([1]),
        device="cpu",
    )
    designs = torch.tensor(
        [[[-2.0, 2.0], [-1.0, 1.0], [0.0, 0.5]]],
    )
    outcomes = torch.tensor([[[3.5], [4.5], [5.5]]])
    actions = torch.tensor(
        [[[-1.5, 1.5], [-0.5, 0.5], [0.25, -0.25]]],
    )

    buffer.extend_episode_batch(
        designs=designs,
        outcomes=outcomes,
        actions=actions,
        rewards=torch.tensor([[1.0, 2.0, 3.0]]),
        dones=torch.tensor([[False, False, True]]),
    )

    monkeypatch.setattr(
        torch,
        "randint",
        lambda high, size, device=None: torch.tensor([0, 1, 2], device=device),
    )
    sample = buffer.sample()

    assert sample.device == torch.device("cpu")
    assert sample.batch_size == torch.Size([3])
    assert sample["observation", "designs"].shape == (3, 3, 2)
    assert sample["observation", "mask"].dtype == torch.bool
    assert sample["observation", "t"].dtype == torch.long
    assert sample["done"].dtype == torch.bool
    torch.testing.assert_close(sample["action"], actions[0])
    torch.testing.assert_close(sample["reward"], torch.tensor([1.0, 2.0, 3.0]))
    assert sample["done"].tolist() == [False, False, True]
    assert sample["observation", "t"].tolist() == [0, 1, 2]
    assert sample["next_observation", "t"].tolist() == [1, 2, 3]
    assert sample["observation", "mask"].tolist() == [
        [False, False, False],
        [True, False, False],
        [True, True, False],
    ]
    assert sample["next_observation", "mask"].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    torch.testing.assert_close(
        sample["observation", "designs"][1],
        torch.tensor([[-2.0, 2.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        sample["next_observation", "outcomes"][1],
        torch.tensor([[3.5], [4.5], [0.0]]),
    )
    assert torch.all(sample["action"][0].abs() <= 4.0)


def test_compact_replay_buffer_rollover_retains_newest_transitions(monkeypatch):
    buffer = _CompactEpisodeReplayBuffer(
        buffer_size=5,
        batch_size=5,
        horizon=3,
        event_shape_design=torch.Size([1]),
        event_shape_y=torch.Size([1]),
        device="cpu",
    )

    for episode in range(3):
        transition_ids = torch.arange(
            episode * 3,
            episode * 3 + 3,
            dtype=torch.float32,
        )
        values = transition_ids.reshape(1, 3, 1)
        buffer.extend_episode_batch(
            designs=values,
            outcomes=values,
            actions=values,
            rewards=transition_ids.reshape(1, 3),
            dones=torch.zeros(1, 3, dtype=torch.bool),
        )

    monkeypatch.setattr(
        torch,
        "randint",
        lambda high, size, device=None: torch.arange(high, device=device),
    )
    sample = buffer.sample()

    assert len(buffer) == 5
    torch.testing.assert_close(
        sample["action"].squeeze(-1),
        torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    torch.testing.assert_close(
        sample["reward"],
        torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    assert sample["observation", "t"].tolist() == [1, 2, 0, 1, 2]


def test_rlboed_requires_outcome_bounds():
    with pytest.raises(TypeError):
        RLBOEDConfig()


def test_compact_replay_buffer_respects_storage_device():
    requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    buffer = _CompactEpisodeReplayBuffer(
        buffer_size=2,
        batch_size=1,
        horizon=2,
        event_shape_design=torch.Size([1]),
        event_shape_y=torch.Size([1]),
        device=requested_device,
    )

    assert buffer.designs.device.type == requested_device

    values = torch.arange(2, dtype=torch.float32, device=buffer.designs.device).reshape(
        1, 2, 1
    )
    buffer.extend_episode_batch(
        designs=values,
        outcomes=values,
        actions=values,
        rewards=values.squeeze(-1),
        dones=torch.zeros(1, 2, dtype=torch.bool, device=buffer.designs.device),
    )
    sample = buffer.sample()

    assert sample.device == buffer.designs.device
    assert sample["action"].device == buffer.designs.device


def test_rlboed_rejects_unavailable_cuda_buffer_device():
    config = RLBOEDConfig(
        outcome_low=-3.0,
        outcome_high=10.0,
        buffer_device="cuda",
    )
    trainer = RLBOED(config)

    if torch.cuda.is_available():
        assert trainer._resolve_buffer_device().type == "cuda"
    else:
        with pytest.raises(ValueError, match="requires CUDA"):
            trainer._resolve_buffer_device()


def test_rlboed_rejects_gpu_buffer_device_alias():
    config = RLBOEDConfig(
        outcome_low=-3.0,
        outcome_high=10.0,
        buffer_device="gpu",
    )

    with pytest.raises(ValueError, match="cpu.*cuda"):
        RLBOED(config)._resolve_buffer_device()


def test_rlboed_config_generic_defaults_match_example():
    config = RLBOEDConfig(
        outcome_low=sac_location_finding.Args.outcome_low,
        outcome_high=sac_location_finding.Args.outcome_high,
    )

    assert config.outcome_low == sac_location_finding.Args.outcome_low
    assert config.outcome_high == sac_location_finding.Args.outcome_high
    assert not hasattr(config, "design_low")
    assert not hasattr(config, "design_high")

    for field_name in (
        "seed",
        "torch_deterministic",
        "n_contrastive",
        "num_training_iterations",
        "num_envs",
        "buffer_size",
        "buffer_device",
        "gamma",
        "tau",
        "batch_size",
        "learning_starts",
        "gradient_steps_per_iteration",
        "policy_lr",
        "q_lr",
        "policy_init_std",
        "policy_frequency",
        "target_network_frequency",
        "alpha",
        "initial_alpha",
        "target_entropy",
        "entropy_anneal_rate",
        "log_frequency",
    ):
        assert getattr(config, field_name) == getattr(
            sac_location_finding.Args, field_name
        )


def test_rlboed_requires_design_space():
    task = LocationFinding(n_sources=1, dim_theta=2)
    config = RLBOEDConfig(outcome_low=-3.0, outcome_high=10.0)

    with pytest.raises(TypeError):
        RLBOED(config).fit(task, horizon=2)


def test_deployable_policy_maps_raw_history_to_physical_designs():
    design_space = BoxDesignSpace(-4.0, 4.0, event_shape=(2,))
    actor = DeepSetGaussianActor(dim_y=1, dim_design=2)
    policy = TanhGaussianPolicyNet(
        design_space,
        actor=actor,
        normalize_design_history=True,
        design_history_bounds=(0.0, 1.0),
        outcome_bounds=(-3.0, 10.0),
    )
    raw_y = torch.tensor([[[3.5]]])
    raw_designs = torch.tensor([[[-4.0, 4.0]]])

    mean, _ = actor(
        torch.tensor([[[0.5]]]),
        torch.tensor([[[0.0, 1.0]]]),
    )
    expected = torch.tanh(mean) * 4.0
    actual = policy(raw_y, raw_designs)

    torch.testing.assert_close(actual, expected)
    assert torch.all(actual >= -4.0)
    assert torch.all(actual <= 4.0)


def test_rlboed_fit_returns_deployable_designer(tmp_path):
    task = LocationFinding(n_sources=1, dim_theta=2)
    design_space = BoxDesignSpace(-2.0, 3.0, event_shape=(2,))
    config = RLBOEDConfig(
        outcome_low=-3.0,
        outcome_high=10.0,
        n_contrastive=2,
        num_training_iterations=3,
        num_envs=2,
        buffer_size=20,
        batch_size=2,
        learning_starts=1,
        gradient_steps_per_iteration=1,
        log_frequency=1,
        verbose=False,
    )

    result = RLBOED(config).fit(
        task,
        horizon=2,
        design_space=design_space,
    )

    assert isinstance(result, FitResult)
    assert isinstance(result.designer, PolicyDesigner)
    assert isinstance(result.designer.policy_net, TanhGaussianPolicyNet)
    assert not result.designer.deterministic
    assert result.designer.policy_net.design_history_bounds == (0.0, 1.0)
    assert result.checkpoint["config"]["gamma"] == config.gamma
    assert result.checkpoint["alpha"].item() > 0
    assert "log_alpha" in result.checkpoint
    assert result.checkpoint["alpha_optimizer"] is not None
    assert result.checkpoint["optimization_step"] == 3
    assert result.checkpoint["env_steps"] == 12
    assert result.checkpoint["training_iterations"] == config.num_training_iterations
    assert result.checkpoint["global_step"] == config.num_training_iterations
    torch.testing.assert_close(
        result.checkpoint["design_low"],
        torch.full((2,), -2.0),
    )
    torch.testing.assert_close(
        result.checkpoint["design_high"],
        torch.full((2,), 3.0),
    )

    theta = task.sample_prior(2)
    y_history, design_history = result.designer.rollout(theta)
    assert y_history.shape == (2, 2, 1)
    assert design_history.shape == (2, 2, 2)
    assert torch.all(design_history >= -2.0)
    assert torch.all(design_history <= 3.0)

    checkpoint_path = tmp_path / "rlboed.pt"
    result.save(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert "designer" in checkpoint
    assert "actor_optimizer" in checkpoint


@pytest.mark.parametrize("alpha", [0.0, 0.2])
def test_rlboed_fixed_alpha_disables_temperature_optimizer(alpha):
    task = LocationFinding(n_sources=1, dim_theta=2)
    design_space = BoxDesignSpace(-2.0, 3.0, event_shape=(2,))
    config = RLBOEDConfig(
        outcome_low=-3.0,
        outcome_high=10.0,
        n_contrastive=1,
        num_training_iterations=1,
        num_envs=2,
        buffer_size=10,
        batch_size=2,
        learning_starts=1,
        gradient_steps_per_iteration=1,
        alpha=alpha,
        verbose=False,
    )

    result = RLBOED(config).fit(task, horizon=2, design_space=design_space)

    assert result.checkpoint["alpha"].item() == pytest.approx(alpha)
    assert result.checkpoint["alpha_optimizer"] is None
