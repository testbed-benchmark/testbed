"""Tests for the SAC location-finding example."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import LazyTensorStorage, ReplayBuffer

from testbed.design_nets import TanhGaussianPolicyNet
from testbed.designers import PolicyDesigner
from testbed.envs import BoxDesignSpace
from testbed.methods import RLBOED, FitResult, RLBOEDConfig
from testbed.tasks import LocationFinding

EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "sac_location_finding.py"
SPEC = importlib.util.spec_from_file_location("sac_location_finding", EXAMPLE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
sac_location_finding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sac_location_finding)
DADGaussianActor = sac_location_finding.DADGaussianActor
SoftQNetwork = sac_location_finding.SoftQNetwork


def _linear_shapes(module):
    return [
        (layer.in_features, layer.out_features)
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]


def test_actor_defaults_match_released_rl_boed_architecture():
    """The default actor reproduces the released source-location policy."""
    actor = DADGaussianActor(dim_y=1, dim_design=2)

    assert _linear_shapes(actor.encoder_net) == [(3, 128), (128, 128), (128, 64)]
    assert _linear_shapes(actor.emitter_net) == [(64, 128), (128, 128)]
    assert actor.mean_head.in_features == 128
    assert actor.log_std_head.in_features == 128
    assert actor.log_std_min == -20.0
    assert actor.log_std_max == 0.0
    assert not hasattr(actor, "null_enc")


def test_actor_has_released_initial_standard_deviation():
    """The log-std output bias starts at std sqrt(1/3)."""
    actor = DADGaussianActor(dim_y=1, dim_design=2)

    torch.testing.assert_close(
        actor.log_std_head.bias.exp(),
        torch.full_like(actor.log_std_head.bias, math.sqrt(1.0 / 3.0)),
    )


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


def test_soft_q_network_uses_same_coordinates_for_designs_and_actions():
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
        torch.zeros(1, 1, 2),
    )
    torch.testing.assert_close(
        inputs["emitter"][..., -2:],
        torch.zeros(1, 2),
    )


def test_replay_buffer_keeps_physical_transitions_on_cpu():
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(2, device="cpu"),
        batch_size=1,
    )
    observation = {
        "designs": torch.tensor([[[-2.0, 2.0]]]),
        "outcomes": torch.tensor([[[3.5]]]),
        "mask": torch.tensor([[True]]),
        "t": torch.tensor([1]),
    }
    action = torch.tensor([[-2.0, 2.0]])

    buffer.extend(
        TensorDict(
            {
                "observation": observation,
                "next_observation": observation,
                "action": action,
                "reward": torch.tensor([1.0]),
                "done": torch.tensor([False]),
            },
            batch_size=[1],
        )
    )
    sample = buffer.sample()

    assert sample.device == torch.device("cpu")
    assert sample.batch_size == torch.Size([1])
    assert sample["observation", "designs"].shape == (1, 1, 2)
    assert sample["observation", "mask"].dtype == torch.bool
    assert sample["observation", "t"].dtype == torch.long
    assert sample["done"].dtype == torch.bool
    torch.testing.assert_close(sample["action"][0], action[0])
    torch.testing.assert_close(
        sample["observation", "designs"][0],
        observation["designs"][0],
    )
    assert torch.all(sample["action"][0].abs() <= 4.0)


def test_replay_buffer_rollover_retains_newest_transitions():
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(3, device="cpu"),
        batch_size=3,
    )
    transition = TensorDict(
        {
            "observation": {"t": torch.arange(5)},
            "next_observation": {"t": torch.arange(1, 6)},
            "action": torch.arange(5, dtype=torch.float32).unsqueeze(-1),
            "reward": torch.arange(5, dtype=torch.float32),
            "done": torch.zeros(5, dtype=torch.bool),
        },
        batch_size=[5],
    )

    buffer.extend(transition)
    stored = buffer[:]

    assert len(buffer) == 3
    assert set(stored["observation", "t"].tolist()) == {2, 3, 4}


def test_rlboed_requires_outcome_bounds():
    with pytest.raises(TypeError):
        RLBOEDConfig()


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
        "total_timesteps",
        "num_envs",
        "buffer_size",
        "gamma",
        "tau",
        "batch_size",
        "learning_starts",
        "policy_lr",
        "q_lr",
        "policy_frequency",
        "target_network_frequency",
        "alpha",
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
    actor = DADGaussianActor(dim_y=1, dim_design=2)
    policy = TanhGaussianPolicyNet(
        design_space,
        actor=actor,
        normalize_design_history=True,
        outcome_bounds=(-3.0, 10.0),
    )
    raw_y = torch.tensor([[[3.5]]])
    raw_designs = torch.tensor([[[-4.0, 4.0]]])

    mean, _ = actor(
        torch.tensor([[[0.5]]]),
        torch.tensor([[[-1.0, 1.0]]]),
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
        total_timesteps=3,
        num_envs=2,
        buffer_size=20,
        batch_size=2,
        learning_starts=1,
        log_frequency=1,
    )

    result = RLBOED(config).fit(
        task,
        horizon=2,
        design_space=design_space,
    )

    assert isinstance(result, FitResult)
    assert isinstance(result.designer, PolicyDesigner)
    assert isinstance(result.designer.policy_net, TanhGaussianPolicyNet)
    assert result.checkpoint["config"]["gamma"] == config.gamma
    assert result.checkpoint["alpha"] == config.alpha
    assert "log_alpha" not in result.checkpoint
    assert "alpha_optimizer" not in result.checkpoint
    assert result.checkpoint["global_step"] == config.total_timesteps
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
