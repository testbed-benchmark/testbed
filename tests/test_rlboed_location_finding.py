"""Tests for the RL-BOED (SAC) training components."""

import json
import random
from math import log

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from testbed.core.design_spaces import BoxDesignSpace, FiniteDesignSpace
from testbed.core.manager import Manager
from testbed.design_nets import (
    AttentiveEncoder,
    CategoricalHead,
    CategoricalPolicyActor,
    CategoricalPolicyNet,
    DeepSetEncoder,
    GaussianHead,
    GaussianPolicyActor,
    GaussianPolicyNet,
)
from testbed.designers import PolicyDesigner
from testbed.envs.history import History
from testbed.eval.eval_policy import (
    build_design_net as reconstruct_design_net,
)
from testbed.eval.eval_policy import (
    evaluate_checkpoint,
)
from testbed.method_nets import RLBOEDDiscreteQNetwork, RLBOEDQNetwork
from testbed.methods.rlboed import (
    Trainer,
    _CompactEpisodeReplayBuffer,
    _resolve_buffer_device,
    _set_seed,
    build_env,
    build_replay_buffer,
    make_networks,
    validate_config,
)
from testbed.tasks import LocationFinding, PreyPopulation

SoftQNetwork = RLBOEDQNetwork


def _history(observation):
    """Unpacks an env History into the (y_history, design_history) convention."""
    return observation.outcomes, observation.designs


# =====================================================================

# Network architecture
# =====================================================================


def test_actor_initializes_requested_standard_deviation():
    actor = GaussianPolicyActor(
        DeepSetEncoder((1,), (2,)), GaussianHead(64, 2, init_std=0.4)
    )
    y_history = torch.randn(3, 4, 1)
    design_history = torch.randn(3, 4, 2)

    _, log_std = actor(y_history, design_history)

    torch.testing.assert_close(log_std.exp(), torch.full_like(log_std, 0.4))
    assert torch.count_nonzero(actor.head.log_std_head.weight) == 0
    assert actor.head.log_std_head.weight.requires_grad


@pytest.mark.parametrize("init_std", [0.0, -0.1, 1.0, 1e-10])
def test_actor_rejects_initial_std_outside_log_std_bounds(init_std):
    with pytest.raises(ValueError, match="init_std"):
        GaussianHead(64, 2, init_std=init_std)


def test_soft_q_network_is_permutation_invariant():
    """Permuting valid history entries does not change the Q estimate."""
    critic = SoftQNetwork(
        event_shape_y=(1,),
        event_shape_design=(2,),
        design_low=-4.0,
        design_high=4.0,
        outcome_low=-3.0,
        outcome_high=10.0,
    )
    designs = torch.randn(3, 5, 2)
    outcomes = torch.randn(3, 5, 1)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, False, False, False, False],
        ]
    )
    observation = History(
        max_episode_length=5, designs=designs, outcomes=outcomes, mask=mask
    )
    design = torch.randn(3, 2)
    permutation = torch.tensor([2, 0, 4, 1, 3])
    permuted_observation = History(
        max_episode_length=5,
        designs=designs[:, permutation],
        outcomes=outcomes[:, permutation],
        mask=mask[:, permutation],
    )

    actual = critic(*_history(observation), design, mask=observation.mask)
    permuted = critic(
        *_history(permuted_observation), design, mask=permuted_observation.mask
    )

    torch.testing.assert_close(actual, permuted)


def test_soft_q_network_ignores_masked_history_entries():
    """Changing padded history entries does not change the Q estimate."""
    critic = SoftQNetwork(
        event_shape_y=(1,),
        event_shape_design=(2,),
        design_low=-4.0,
        design_high=4.0,
        outcome_low=-3.0,
        outcome_high=10.0,
    )
    designs = torch.randn(2, 4, 2)
    outcomes = torch.randn(2, 4, 1)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    observation = History(
        max_episode_length=4, designs=designs, outcomes=outcomes, mask=mask
    )
    mod_designs = designs.clone()
    mod_outcomes = outcomes.clone()
    mod_designs[~mask] = 1000.0
    mod_outcomes[~mask] = -1000.0
    modified_observation = History(
        max_episode_length=4, designs=mod_designs, outcomes=mod_outcomes, mask=mask
    )
    design = torch.randn(2, 2)

    actual = critic(*_history(observation), design, mask=observation.mask)
    modified = critic(
        *_history(modified_observation), design, mask=modified_observation.mask
    )

    assert actual.shape == (2, 1)
    torch.testing.assert_close(actual, modified)


def test_soft_q_network_uses_reference_history_and_design_coordinates():
    critic = SoftQNetwork(
        event_shape_y=(1,),
        event_shape_design=(2,),
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

    critic.encoder.encoder_net.register_forward_pre_hook(record_encoder_input)
    critic.emitter_net.register_forward_pre_hook(record_emitter_input)
    physical_design = torch.tensor([[1.0, 4.0]])
    observation = History(
        max_episode_length=1,
        designs=physical_design.unsqueeze(1),
        outcomes=torch.tensor([[[3.5]]]),
        mask=torch.tensor([[True]]),
    )

    critic(*_history(observation), physical_design, mask=observation.mask)

    torch.testing.assert_close(
        inputs["encoder"][..., -2:],
        torch.full((1, 1, 2), 0.5),
    )
    torch.testing.assert_close(
        inputs["emitter"][..., -2:],
        torch.zeros(1, 2),
    )


def test_gaussian_policy_maps_physical_history_to_raw_proposal():
    design_space = BoxDesignSpace(-4.0, 4.0, event_shape=(2,))
    actor = GaussianPolicyActor(DeepSetEncoder((1,), (2,)), GaussianHead(64, 2))
    policy = GaussianPolicyNet(
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
    expected = mean
    actual = policy(raw_y, raw_designs)

    torch.testing.assert_close(actual, expected)
    assert design_space.contains(design_space.to_design(actual))


def test_categorical_policy_returns_declared_designs_and_greedy_argmax():
    design_space = FiniteDesignSpace(torch.tensor([[1.0], [5.0], [10.0]]))
    actor = CategoricalPolicyActor(DeepSetEncoder((1,), (1,)), CategoricalHead(64, 3))
    policy = CategoricalPolicyNet(
        design_space,
        actor=actor,
        normalize_design_history=True,
        outcome_bounds=(0.0, 10.0),
    )
    with torch.no_grad():
        actor.head.logits_head.bias.copy_(torch.tensor([0.0, -1.0, 2.0]))
    outcomes = torch.zeros(4, 0, 1)
    designs = torch.zeros(4, 0, 1)

    proposal = policy.propose(outcomes, designs)
    greedy = policy(outcomes, designs)

    assert design_space.contains(proposal.raw_design)
    torch.testing.assert_close(greedy, torch.full((4, 1), 10.0))
    torch.testing.assert_close(
        proposal.deterministic_raw_design, torch.full((4, 1), 10.0)
    )
    selected_log_probs = torch.log_softmax(actor.head.logits_head.bias, dim=0)[
        proposal.info["design_index"]
    ]
    torch.testing.assert_close(proposal.log_prob, selected_log_probs)


def test_discrete_q_network_is_masked_and_permutation_invariant():
    critic = RLBOEDDiscreteQNetwork(
        event_shape_y=(1,),
        event_shape_design=(1,),
        n_designs=3,
        design_low=1.0,
        design_high=10.0,
        outcome_low=0.0,
        outcome_high=10.0,
    )
    designs = torch.tensor([[[1.0], [5.0], [10.0]], [[5.0], [1.0], [10.0]]])
    outcomes = torch.tensor([[[1.0], [2.0], [1000.0]], [[3.0], [4.0], [1000.0]]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    observation = History(3, designs=designs, outcomes=outcomes, mask=mask)
    permuted = History(
        3,
        designs=designs[:, [1, 0, 2]],
        outcomes=outcomes[:, [1, 0, 2]],
        mask=mask[:, [1, 0, 2]],
    )

    values = critic(*_history(observation), mask=observation.mask)

    assert values.shape == (2, 3)
    torch.testing.assert_close(values, critic(*_history(permuted), mask=permuted.mask))


# =====================================================================

# Replay buffer
# =====================================================================


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
    policy_designs = torch.tensor(
        [[[-1.5, 1.5], [-0.5, 0.5], [0.25, -0.25]]],
    )

    buffer.extend_episode_batch(
        designs=designs,
        outcomes=outcomes,
        policy_designs=policy_designs,
        rewards=torch.tensor([[1.0, 2.0, 3.0]]),
        dones=torch.tensor([[False, False, True]]),
    )

    monkeypatch.setattr(
        torch,
        "randint",
        lambda high, size, device=None: torch.tensor([0, 1, 2], device=device),
    )
    sample = buffer.sample()

    assert sample.design.device == torch.device("cpu")
    assert sample.design.shape[:1] == torch.Size([3])
    assert sample.observation["designs"].shape == (3, 3, 2)
    assert sample.observation["mask"].dtype == torch.bool
    assert sample.done.dtype == torch.bool
    torch.testing.assert_close(sample.design, policy_designs[0])
    torch.testing.assert_close(sample.reward, torch.tensor([1.0, 2.0, 3.0]))
    assert sample.done.tolist() == [False, False, True]
    assert sample.observation["mask"].tolist() == [
        [False, False, False],
        [True, False, False],
        [True, True, False],
    ]
    assert sample.next_observation["mask"].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]
    torch.testing.assert_close(
        sample.observation["designs"][1],
        torch.tensor([[-2.0, 2.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(
        sample.next_observation["outcomes"][1],
        torch.tensor([[3.5], [4.5], [0.0]]),
    )
    assert torch.all(sample.design[0].abs() <= 4.0)


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
            policy_designs=values,
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
        sample.design.squeeze(-1),
        torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    torch.testing.assert_close(
        sample.reward,
        torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0]),
    )


# =====================================================================

# Buffer-device resolution
# =====================================================================


def test_rlboed_rejects_unavailable_cuda_buffer_device():
    if torch.cuda.is_available():
        assert _resolve_buffer_device("cuda").type == "cuda"
    else:
        with pytest.raises(ValueError, match="requires CUDA"):
            _resolve_buffer_device("cuda")


def test_rlboed_rejects_gpu_buffer_device_alias():
    with pytest.raises(ValueError, match="cpu.*cuda"):
        _resolve_buffer_device("gpu")


# =====================================================================

# Trainer
# =====================================================================


def _tiny_cfg(**sac_overrides):
    """A small CPU config that exercises a full warm-up + gradient step."""
    cfg = OmegaConf.create(
        {
            "seed": 373693,
            "torch_deterministic": True,
            "device": "cpu",
            "T": 2,
            "task": {
                "_target_": "testbed.tasks.LocationFinding",
                "n_sources": 1,
                "dim_theta": 2,
                "design_space": {
                    "_target_": "testbed.core.design_spaces.BoxDesignSpace",
                    "low": "${design_space.low}",
                    "high": "${design_space.high}",
                    "event_shape": [2],
                },
            },
            "design_space": {"low": -2.0, "high": 3.0},
            "reward": {"n_contrastive": 2, "outcome_low": -3.0, "outcome_high": 10.0},
            "critic_architecture": "deepset",
            "design_net": {
                "_target_": "testbed.design_nets.make_stochastic_policy",
                "architecture": "deepset",
                "dim_h": [128, 128],
                "dim_emb": 64,
                "log_std_min": -20.0,
                "log_std_max": 0.0,
                "init_std": 3**-0.5,
                "normalize_design_history": True,
                "design_history_bounds": [0.0, 1.0],
                "outcome_bounds": [
                    "${reward.outcome_low}",
                    "${reward.outcome_high}",
                ],
            },
            "eval_every": 0,
            "eval_n_contrastive": 4,
            "train_time_budget_seconds": 60,
            "checkpoint_every": 0,
            "sac": {
                "num_training_iterations": 3,
                "num_envs": 2,
                "buffer_size": 20,
                "buffer_device": "cpu",
                "gamma": 0.9,
                "tau": 1e-3,
                "batch_size": 2,
                "learning_starts": 1,
                "gradient_steps_per_iteration": 1,
                "policy_lr": 1e-4,
                "q_lr": 3e-4,
                "policy_frequency": 1,
                "target_network_frequency": 1,
                "alpha": None,
                "initial_alpha": 1.0,
                "target_entropy": None,
                "entropy_anneal_rate": 1 / 1.4e4,
                "entropy_anneal_total": 1.0,
                "log_frequency": 1,
            },
        }
    )
    for key, value in sac_overrides.items():
        cfg.sac[key] = value
    return cfg


def _build_trainer(cfg, *, manager=None):
    task = instantiate(cfg.task).to(cfg.device)
    _set_seed(cfg.seed, cfg.torch_deterministic)
    nets = make_networks(task, cfg, task.device)
    design_space = nets.design_space
    validate_config(task, cfg, design_space)
    env = build_env(cfg, task)
    replay_buffer = build_replay_buffer(cfg, task)
    designer = PolicyDesigner(task, cfg.T, nets.actor)
    return Trainer(task, designer, env, replay_buffer, nets, cfg, manager=manager)


def _episodic_returns(trainer):
    trainer.train()
    return [m["episodic_return"] for m in trainer.metrics if "episodic_return" in m]


def test_trainer_requires_policy_designer():
    cfg = _tiny_cfg()
    task = instantiate(cfg.task).to(cfg.device)
    _set_seed(cfg.seed, cfg.torch_deterministic)
    nets = make_networks(task, cfg, task.device)
    env = build_env(cfg, task)
    replay_buffer = build_replay_buffer(cfg, task)

    with pytest.raises(AssertionError):
        Trainer(task, object(), env, replay_buffer, nets, cfg)


def test_validate_config_requires_box_design_space():
    cfg = _tiny_cfg()
    task = LocationFinding(n_sources=1, dim_theta=2)

    with pytest.raises(TypeError, match="BoxDesignSpace"):
        validate_config(task, cfg, object())


def test_validate_config_accepts_categorical_design_space():
    cfg = _tiny_cfg()
    task = LocationFinding(n_sources=1, dim_theta=2)
    design_space = FiniteDesignSpace(torch.tensor([[-1.0, -1.0], [1.0, 1.0]]))

    validate_config(task, cfg, design_space)


def test_rlboed_uses_a_tasks_categorical_design_space():
    cfg = _tiny_cfg()
    task = PreyPopulation()
    cfg.task = {
        "_target_": "testbed.tasks.PreyPopulation",
    }
    design_space = make_networks(task, cfg, task.device).design_space

    assert isinstance(design_space, FiniteDesignSpace)
    torch.testing.assert_close(design_space.values, task.design_space.values)
    validate_config(task, cfg, design_space)


def test_validate_config_rejects_inverted_outcome_bounds():
    cfg = _tiny_cfg()
    cfg.reward.outcome_low = 10.0
    cfg.reward.outcome_high = -3.0
    task = LocationFinding(n_sources=1, dim_theta=2)
    design_space = BoxDesignSpace(-2.0, 3.0, task.event_shape_design)

    with pytest.raises(ValueError, match="outcome_low"):
        validate_config(task, cfg, design_space)


def test_train_step_updates_actor_parameters():
    trainer = _build_trainer(_tiny_cfg())
    before = [p.clone() for p in trainer.nets.actor.parameters()]

    trainer.train_step()

    assert any(
        not torch.equal(b, a) for b, a in zip(before, trainer.nets.actor.parameters())
    )


def test_categorical_train_step_updates_networks_with_finite_losses():
    cfg = _tiny_cfg()
    cfg.reward.outcome_low = 0.0
    cfg.reward.outcome_high = 300.0
    task = PreyPopulation().to(cfg.device)
    cfg.task = {"_target_": "testbed.tasks.PreyPopulation"}
    _set_seed(cfg.seed, cfg.torch_deterministic)
    nets = make_networks(task, cfg, task.device)
    design_space = nets.design_space
    validate_config(task, cfg, design_space)
    trainer = Trainer(
        task,
        PolicyDesigner(task, cfg.T, nets.actor),
        build_env(cfg, task),
        build_replay_buffer(cfg, task),
        nets,
        cfg,
    )
    actor_before = [parameter.clone() for parameter in trainer.nets.actor.parameters()]
    critic_before = [parameter.clone() for parameter in trainer.nets.qf1.parameters()]

    trainer.train_step()

    assert isinstance(trainer.nets.actor, CategoricalPolicyNet)
    assert torch.isfinite(trainer._qf_loss)
    assert torch.isfinite(trainer._actor_loss)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(actor_before, trainer.nets.actor.parameters())
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, trainer.nets.qf1.parameters())
    )


def test_categorical_temperature_increases_when_entropy_is_below_target():
    cfg = _tiny_cfg()
    cfg.reward.outcome_low = 0.0
    cfg.reward.outcome_high = 300.0
    task = PreyPopulation().to(cfg.device)
    cfg.task = {"_target_": "testbed.tasks.PreyPopulation"}
    nets = make_networks(task, cfg, task.device)
    design_space = nets.design_space
    with torch.no_grad():
        for parameter in nets.actor.parameters():
            parameter.zero_()
        nets.actor.actor.head.logits_head.bias[0] = 10.0
    trainer = Trainer(
        task,
        PolicyDesigner(task, cfg.T, nets.actor),
        build_env(cfg, task),
        build_replay_buffer(cfg, task),
        nets,
        cfg,
    )
    alpha_before = trainer.nets.log_alpha.exp().item()
    assert trainer.target_entropy == pytest.approx(0.98 * log(design_space.n_designs))

    trainer.train_step()

    assert trainer.nets.log_alpha.exp().item() > alpha_before


def test_train_is_reproducible_with_fixed_seed():
    returns_a = _episodic_returns(_build_trainer(_tiny_cfg()))
    returns_b = _episodic_returns(_build_trainer(_tiny_cfg()))

    assert len(returns_a) == 3
    assert returns_a == returns_b


def test_train_writes_checkpoints_and_metrics(tmp_path):
    cfg = _tiny_cfg()
    cfg.checkpoint_every = 1
    manager = Manager(
        cfg, root=str(tmp_path), name="run", timestamp=False, console=False
    )
    trainer = _build_trainer(cfg, manager=manager)

    trainer.train()

    run_dir = tmp_path / "run"
    final = torch.load(run_dir / "final_step.pt", weights_only=True)
    assert final["step"] == cfg.sac.num_training_iterations - 1
    assert "design_net" in final
    assert "designer" not in final
    assert "actor" not in final
    assert "replay_buffer" in final
    assert "rng_state" in final
    assert not (run_dir / "best.pt").exists()
    for iteration in range(1, 4):
        policy = torch.load(run_dir / f"step_{iteration:08d}.pt", weights_only=True)
        assert "design_net" in policy
        assert "replay_buffer" not in policy
        assert "actor_optimizer" not in policy
        assert policy["completed_iterations"] == iteration
        assert policy["step"] == iteration - 1
        assert 0 <= policy["elapsed_seconds"] <= final["elapsed_seconds"]

    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("episodic_return" in record for record in records)
    assert any("qf_loss" in record for record in records)


def test_trainer_produces_deployable_designer(tmp_path):
    cfg = _tiny_cfg()
    trainer = _build_trainer(cfg)

    trainer.train()

    assert isinstance(trainer.designer, PolicyDesigner)
    assert isinstance(trainer.designer.design_net, GaussianPolicyNet)
    assert trainer.designer.requires_policy_log_probs
    assert trainer.designer.design_net.design_history_bounds == (0.0, 1.0)

    checkpoint = trainer.build_checkpoint()
    assert checkpoint["config"]["sac"]["gamma"] == cfg.sac.gamma
    assert checkpoint["alpha"].item() > 0
    assert "log_alpha" in checkpoint
    assert checkpoint["alpha_optimizer"] is not None
    assert checkpoint["optimization_step"] == 3
    assert checkpoint["env_steps"] == 12
    assert checkpoint["training_iterations"] == cfg.sac.num_training_iterations
    assert checkpoint["global_step"] == cfg.sac.num_training_iterations
    torch.testing.assert_close(checkpoint["design_low"], torch.full((2,), -2.0))
    torch.testing.assert_close(checkpoint["design_high"], torch.full((2,), 3.0))

    theta = trainer.task.sample_prior(2)
    rollout = trainer.designer.rollout(theta)
    assert rollout.y_history.shape == (2, 2, 1)
    assert rollout.design_history.shape == (2, 2, 2)
    assert torch.all(rollout.design_history >= -2.0)
    assert torch.all(rollout.design_history <= 3.0)

    checkpoint_path = tmp_path / "rlboed.pt"
    torch.save(trainer.build_checkpoint(), checkpoint_path)
    reloaded = torch.load(checkpoint_path, weights_only=True)
    assert "design_net" in reloaded
    assert "designer" not in reloaded
    assert "actor" not in reloaded
    assert "actor_optimizer" in reloaded


@pytest.mark.parametrize("alpha", [0.0, 0.2])
def test_rlboed_fixed_alpha_disables_temperature_optimizer(alpha):
    cfg = _tiny_cfg(num_training_iterations=1, buffer_size=10, alpha=alpha)
    cfg.reward.n_contrastive = 1
    trainer = _build_trainer(cfg)

    trainer.train()

    checkpoint = trainer.build_checkpoint()
    assert checkpoint["alpha"].item() == pytest.approx(alpha)
    assert checkpoint["alpha_optimizer"] is None


@pytest.mark.parametrize("setup_seconds", [0.0, 12.0])
@pytest.mark.parametrize("gradient_steps", [1, 3])
def test_budget_checked_after_update_and_excludes_save(
    tmp_path, monkeypatch, setup_seconds, gradient_steps
):
    import testbed.methods.rlboed as module

    cfg = _tiny_cfg(gradient_steps_per_iteration=gradient_steps)
    cfg.train_time_budget_seconds = 10
    manager = Manager(
        cfg, root=str(tmp_path), name="timed", timestamp=False, console=False
    )
    trainer = _build_trainer(cfg, manager=manager)
    initial_entropy = trainer.target_entropy
    trainer._start_time = 0
    clock = [setup_seconds]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    update = module.sac_update
    save = torch.save
    sync_calls = []

    def slow_update(*args, **kwargs):
        result = update(*args, **kwargs)
        clock[0] += 11
        return result

    def slow_save(state, path):
        assert sync_calls == [1]
        clock[0] += 100
        save(state, path)

    def final_synchronize():
        sync_calls.append(trainer.optimization_step)
        clock[0] += 3  # Outstanding GPU work finishes before timing and saving.

    monkeypatch.setattr(module, "sac_update", slow_update)
    monkeypatch.setattr(torch, "save", slow_save)
    monkeypatch.setattr(trainer, "_synchronize", final_synchronize)
    trainer.train()
    state = torch.load(tmp_path / "timed" / "final_step.pt", weights_only=True)
    completed = int(gradient_steps == 1)
    assert state["completed_iterations"] == completed
    assert state["optimization_step"] == 1
    assert state["step"] == completed - 1
    assert state["stop_reason"] == "time_budget"
    assert state["training_elapsed_seconds"] == setup_seconds + 14
    assert state["elapsed_seconds"] == setup_seconds + 14
    assert state["deadline_overrun_seconds"] == setup_seconds + 4
    assert state["target_entropy"] == pytest.approx(
        initial_entropy - completed * cfg.sac.entropy_anneal_rate
    )
    if completed:
        assert state["iteration_progress"] is None
    else:
        assert state["iteration_progress"]["updates_completed"] == 1
    # Building another snapshot after saving must not count serialization time.
    assert trainer.build_policy_checkpoint()["elapsed_seconds"] == setup_seconds + 14


@pytest.mark.parametrize("stop_after_update", [1, 3])
def test_budget_stop_preserves_completed_iteration_metrics_and_checkpoint(
    tmp_path, monkeypatch, stop_after_update
):
    import testbed.methods.rlboed as module

    cfg = _tiny_cfg(gradient_steps_per_iteration=3)
    cfg.train_time_budget_seconds = 10
    cfg.checkpoint_every = 1
    cfg.eval_every = 1
    manager = Manager(
        cfg, root=str(tmp_path), name="budget", timestamp=False, console=False
    )
    trainer = _build_trainer(cfg, manager=manager)
    trainer._start_time = 0
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    update = module.sac_update

    def timed_update(*args, **kwargs):
        result = update(*args, **kwargs)
        if result.optimization_step == stop_after_update:
            clock[0] = cfg.train_time_budget_seconds
        return result

    def unexpected_evaluation():
        pytest.fail("Budget stop must not start periodic evaluation")

    monkeypatch.setattr(module, "sac_update", timed_update)
    monkeypatch.setattr(trainer, "evaluate", unexpected_evaluation)
    trainer.train()

    run_dir = tmp_path / "budget"
    final = torch.load(run_dir / "final_step.pt", weights_only=True)
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert final["stop_reason"] == "time_budget"
    assert final["optimization_step"] == stop_after_update
    completed = int(stop_after_update == 3)
    assert final["completed_iterations"] == completed
    assert len(trainer.metrics) == len(records) == completed
    policy_path = run_dir / "step_00000001.pt"
    assert policy_path.exists() == bool(completed)
    if completed:
        assert trainer.metrics[0]["step"] == final["step"] == 0
        assert trainer.metrics[0]["env_steps"] == final["env_steps"]
        assert records[0]["episodic_return"] == trainer.metrics[0]["episodic_return"]
        assert records[0]["step"] == 0
        policy = torch.load(policy_path, weights_only=True)
        assert policy["completed_iterations"] == completed
        torch.testing.assert_close(policy["design_net"], final["design_net"])


def test_evaluation_finishes_before_next_update_checks_budget(monkeypatch):
    import testbed.methods.rlboed as module

    cfg = _tiny_cfg()
    cfg.train_time_budget_seconds = 10
    cfg.eval_every = 1
    trainer = _build_trainer(cfg)
    trainer._start_time = 0
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    build_env = module.build_env
    eval_steps = []

    def slow_eval_env(*args, **kwargs):
        env = build_env(*args, **kwargs)
        step = env.step

        def slow_step(designs):
            result = step(designs)
            eval_steps.append(env.t)
            clock[0] = 12.0
            return result

        monkeypatch.setattr(env, "step", slow_step)
        return env

    monkeypatch.setattr(module, "build_env", slow_eval_env)
    trainer.train()
    assert eval_steps == [1, 2]
    assert any("eval/spce" in record for record in trainer.metrics)
    assert trainer.stop_reason == "time_budget"
    assert trainer.optimization_step == 2
    assert trainer.completed_iterations == 2
    assert trainer._iteration_progress is None
    assert trainer.designer.design_net.training
    assert "eval/spce" in trainer.evaluate()


@pytest.mark.parametrize("was_training", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_evaluation_restores_rng_and_mode(monkeypatch, was_training, fails):
    import testbed.methods.rlboed as module

    trainer = _build_trainer(_tiny_cfg())
    trainer.designer.design_net.train(was_training)
    build_env = module.build_env

    def randomized_env(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        if fails:
            raise RuntimeError("evaluation failed")
        return build_env(*args, **kwargs)

    monkeypatch.setattr(module, "build_env", randomized_env)
    before = module.rng_state()
    if fails:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            trainer.evaluate()
    else:
        trainer.evaluate()
    after = module.rng_state()
    assert after["python"] == before["python"]
    assert after["numpy"] == before["numpy"]
    torch.testing.assert_close(after["torch"], before["torch"])
    if before["cuda"] is not None:
        for actual, expected in zip(after["cuda"], before["cuda"]):
            torch.testing.assert_close(actual, expected)
    assert trainer.designer.design_net.training == was_training


@pytest.mark.parametrize("alpha", [None, 0.0, 0.2])
@pytest.mark.parametrize("stop_after_update", [2, 3])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_checkpoint_continues_remaining_updates(
    tmp_path, monkeypatch, alpha, stop_after_update, device
):
    import testbed.methods.rlboed as module

    cfg = _tiny_cfg(
        num_training_iterations=2, gradient_steps_per_iteration=3, alpha=alpha
    )
    cfg.device = device
    cfg.sac.buffer_device = device
    cfg.train_time_budget_seconds = 10
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    uninterrupted = _build_trainer(cfg)
    uninterrupted.train()
    expected = uninterrupted.build_checkpoint()

    interrupted = _build_trainer(cfg)
    initial_entropy = interrupted.target_entropy
    update = module.sac_update

    def stop_at_update(*args, **kwargs):
        result = update(*args, **kwargs)
        if result.optimization_step == stop_after_update:
            clock[0] = 10.0
        return result

    monkeypatch.setattr(module, "sac_update", stop_at_update)
    interrupted.train()
    path = tmp_path / "partial.pt"
    torch.save(interrupted.build_checkpoint(), path)
    state = torch.load(path, weights_only=True)
    completed = int(stop_after_update == 3)
    assert state["completed_iterations"] == completed
    assert state["optimization_step"] == stop_after_update
    if completed:
        assert state["iteration_progress"] is None
    else:
        assert state["iteration_progress"]["updates_completed"] == stop_after_update
    assert state["target_entropy"] == pytest.approx(
        initial_entropy - completed * cfg.sac.entropy_anneal_rate * (alpha is None)
    )

    # Loading excludes offline time and continues the interrupted update sequence.
    clock[0] = 1000.0
    monkeypatch.setattr(module, "sac_update", update)
    resumed = module.load_checkpoint(path, train_time_budget_seconds=20)
    assert resumed.cfg.train_time_budget_seconds == 20
    assert resumed.build_checkpoint()["elapsed_seconds"] == 10
    for name, value in state["losses"].items():
        torch.testing.assert_close(
            getattr(resumed, name), value.to(resumed.device), equal_nan=True
        )
    resumed.train()
    actual = resumed.build_checkpoint()
    for name in (
        "design_net",
        "qf1",
        "qf2",
        "qf1_target",
        "qf2_target",
        "actor_optimizer",
        "q_optimizer",
        "log_alpha",
        "alpha_optimizer",
        "target_entropy",
        "completed_iterations",
        "optimization_step",
        "env_steps",
        "completed_episodes",
        "replay_buffer",
        "iteration_progress",
        "losses",
    ):
        torch.testing.assert_close(
            actual[name],
            expected[name],
            rtol=0,
            atol=0,
            equal_nan=True,
            check_device=name != "losses",
        )
    assert actual["rng_state"]["python"] == expected["rng_state"]["python"]
    assert actual["rng_state"]["numpy"] == expected["rng_state"]["numpy"]
    torch.testing.assert_close(
        actual["rng_state"]["torch"], expected["rng_state"]["torch"]
    )
    if expected["rng_state"]["cuda"] is not None:
        for actual_rng, expected_rng in zip(
            actual["rng_state"]["cuda"], expected["rng_state"]["cuda"]
        ):
            torch.testing.assert_close(actual_rng, expected_rng, rtol=0, atol=0)


def test_load_checkpoint_restores_limits_and_counts_setup_time(tmp_path, monkeypatch):
    import testbed.methods.rlboed as module

    cfg = _tiny_cfg(num_training_iterations=1)
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    trainer = _build_trainer(cfg)
    clock[0] = 5.0
    trainer.train()
    path = tmp_path / "full.pt"
    torch.save(trainer.build_checkpoint(), path)
    with pytest.raises(ValueError, match="iteration limit is exhausted"):
        module.load_checkpoint(path)
    with pytest.raises(ValueError, match="time budget is exhausted"):
        module.load_checkpoint(path, train_time_budget_seconds=5)

    make_networks = module.make_networks
    clock[0] = 100.0

    def slow_networks(*args, **kwargs):
        result = make_networks(*args, **kwargs)
        clock[0] += 2
        return result

    monkeypatch.setattr(module, "make_networks", slow_networks)
    resumed = module.load_checkpoint(
        path, num_training_iterations=2, train_time_budget_seconds=8
    )
    assert resumed.build_checkpoint()["elapsed_seconds"] == 7
    update = module.sac_update

    def timed_update(*args, **kwargs):
        result = update(*args, **kwargs)
        clock[0] += 1
        return result

    monkeypatch.setattr(module, "sac_update", timed_update)
    resumed.train()
    assert resumed.completed_iterations == 2
    assert resumed.optimization_step == 2
    assert resumed.training_elapsed_seconds == 8
    assert resumed.stop_reason == "time_budget"


def test_load_checkpoint_rejects_policy_only_checkpoint(tmp_path):
    from testbed.methods.rlboed import load_checkpoint

    path = tmp_path / "policy.pt"
    torch.save(_build_trainer(_tiny_cfg()).build_policy_checkpoint(), path)
    with pytest.raises(ValueError, match="full version-2 checkpoint"):
        load_checkpoint(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_load_checkpoint_moves_training_state_to_cuda(tmp_path):
    from testbed.methods.rlboed import load_checkpoint

    trainer = _build_trainer(_tiny_cfg(num_training_iterations=1))
    trainer.train()
    path = tmp_path / "cpu.pt"
    torch.save(trainer.build_checkpoint(), path)
    resumed = load_checkpoint(
        path, device="cuda", buffer_device="cuda", num_training_iterations=2
    )
    assert resumed.device.type == "cuda"
    assert resumed.replay_buffer.device.type == "cuda"
    assert resumed.replay_buffer.designs.is_cuda
    assert resumed.nets.log_alpha.is_cuda
    assert next(resumed.nets.actor.parameters()).is_cuda
    assert not resumed.replay_buffer.pin_memory
    resumed.train()
    assert resumed.optimization_step == 2
    assert torch.isfinite(resumed._qf_loss)

    # A GPU checkpoint can also be reconstructed on a CPU-only destination.
    torch.save(resumed.build_checkpoint(), path)
    restored_cpu = load_checkpoint(
        path, device="cpu", buffer_device="cpu", num_training_iterations=3
    )
    assert restored_cpu.replay_buffer.device.type == "cpu"
    assert not next(restored_cpu.nets.actor.parameters()).is_cuda
    restored_cpu.train()
    assert restored_cpu.optimization_step == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("train_time_budget_seconds", 0),
        ("train_time_budget_seconds", float("nan")),
        ("checkpoint_every", -1),
        ("checkpoint_every", 1.5),
    ],
)
def test_invalid_timing_config(key, value):
    cfg = _tiny_cfg()
    cfg[key] = value
    with pytest.raises(ValueError, match=key):
        _build_trainer(cfg)


def test_warmup_finishes_before_first_budget_check(monkeypatch):
    import testbed.methods.rlboed as module

    cfg = _tiny_cfg(learning_starts=9, gradient_steps_per_iteration=3)
    cfg.train_time_budget_seconds = 10
    trainer = _build_trainer(cfg)
    trainer._start_time = 0
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    step = trainer.env.step
    rollout_steps = []

    def slow_step(designs):
        result = step(designs)
        rollout_steps.append(trainer.env.t)
        clock[0] = 12.0
        return result

    monkeypatch.setattr(trainer.env, "step", slow_step)
    trainer.train()
    state = trainer.build_checkpoint()
    assert rollout_steps == [1, 2, 1, 2, 1, 2]
    assert state["completed_episodes"] == 3 * cfg.sac.num_envs
    assert state["env_steps"] == 12
    assert len(trainer.replay_buffer) == 12
    assert state["optimization_step"] == 1
    assert state["completed_iterations"] == 0
    assert state["iteration_progress"]["updates_completed"] == 1
    assert state["deadline_overrun_seconds"] == 2


def test_periodic_policy_loads_in_evaluator(tmp_path):
    cfg = _tiny_cfg(num_training_iterations=1)
    cfg.checkpoint_every = 1
    manager = Manager(
        cfg, root=str(tmp_path), name="eval", timestamp=False, console=False
    )
    trainer = _build_trainer(cfg, manager=manager)
    trainer.train()
    run_dir = tmp_path / "eval"
    (run_dir / ".hydra").mkdir()
    OmegaConf.save(cfg, run_dir / ".hydra" / "config.yaml")
    result = evaluate_checkpoint(
        str(run_dir),
        checkpoint="step_00000001.pt",
        n_outer=2,
        n_contrastive=2,
        outer_batch_size=2,
        contrastive_chunk_size=2,
    )
    assert result["weights_used"] == "design_net"


def test_main_starts_timer_before_setup(monkeypatch, tmp_path):
    import testbed.methods.rlboed as module

    cfg = _tiny_cfg()
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    manager = Manager(
        cfg, root=str(tmp_path), name="main", timestamp=False, console=False
    )
    task = instantiate(cfg.task).to(cfg.device)

    def setup(_cfg):
        clock[0] = 107.0
        return manager, task

    class RecordingTrainer:
        def __init__(self, **kwargs):
            assert kwargs["start_time"] == 100.0
            assert clock[0] == 107.0

        def train(self):
            pass

        def evaluate(self):
            return {"eval/spce": 0.0, "eval/spce_se": 0.0}

    monkeypatch.setattr(module, "setup", setup)
    monkeypatch.setattr(module, "Trainer", RecordingTrainer)
    module.main.__wrapped__(cfg)
    manager.metric_logger.close()


@pytest.mark.parametrize("budget", [None, float("inf")])
def test_no_time_limit_never_stops_training_on_the_clock(monkeypatch, budget):
    """null / inf disable the deadline, so only the iteration cap stops training."""
    import testbed.methods.rlboed as module

    cfg = _tiny_cfg()
    cfg.train_time_budget_seconds = budget
    trainer = _build_trainer(cfg)
    trainer._start_time = 0
    # A clock far past any plausible budget must not end the run.
    monkeypatch.setattr(module.time, "monotonic", lambda: 1e9)

    trainer.train()

    assert trainer.stop_reason == "iteration_limit"
    assert trainer.completed_iterations == cfg.sac.num_training_iterations


@pytest.mark.parametrize("total,drop", [(1.0, 1.0), (None, 2.5)])
def test_entropy_anneal_stops_at_the_configured_total(total, drop):
    """Unbounded, a wall-clock stop leaves the setpoint wherever the host reached."""
    cfg = _tiny_cfg(num_training_iterations=5, entropy_anneal_rate=0.5)
    cfg.sac.entropy_anneal_total = total
    trainer = _build_trainer(cfg)
    initial = trainer.target_entropy

    trainer.train()

    # 5 iterations x 0.5 would drop 2.5; a total of 1.0 has to stop it at 1.0.
    assert trainer.target_entropy == pytest.approx(initial - drop)


def test_entropy_anneal_floor_survives_a_resume(tmp_path):
    """The floor is derived from cfg, so a resumed run cannot anneal past it."""
    from testbed.methods.rlboed import load_checkpoint

    cfg = _tiny_cfg(num_training_iterations=1, entropy_anneal_rate=0.5)
    cfg.sac.entropy_anneal_total = 1.0
    trainer = _build_trainer(cfg)
    initial = trainer.target_entropy
    trainer.train()
    assert trainer.target_entropy == pytest.approx(initial - 0.5)

    path = tmp_path / "full.pt"
    torch.save(trainer.build_checkpoint(), path)
    resumed = load_checkpoint(path, num_training_iterations=5)

    resumed.train()

    assert resumed.target_entropy == pytest.approx(initial - 1.0)


@pytest.mark.parametrize(
    "actor_architecture,actor_encoder_type",
    [
        ("deepset", DeepSetEncoder),
        ("attention", AttentiveEncoder),
    ],
)
@pytest.mark.parametrize(
    "critic_architecture,critic_encoder_type",
    [
        ("deepset", DeepSetEncoder),
        ("attention", AttentiveEncoder),
    ],
)
@pytest.mark.parametrize("design_kind", ["box", "finite", "unconstrained"])
def test_independent_architectures_train_and_reload(
    actor_architecture,
    actor_encoder_type,
    critic_architecture,
    critic_encoder_type,
    design_kind,
):
    # Enough transitions that the sampled batch isn't all empty histories,
    # which give DeepSet encoders zero gradients.
    cfg = _tiny_cfg(num_envs=8, batch_size=16)
    cfg.design_net._target_ = "testbed.design_nets.make_stochastic_policy"
    cfg.design_net.architecture = actor_architecture
    cfg.critic_architecture = critic_architecture
    cfg.design_net.dim_h = [16, 16]
    cfg.design_net.dim_emb = 16
    cfg.design_net.attention = {"num_heads": 2, "num_blocks": 1, "max_T": cfg.T}
    if design_kind == "finite":
        cfg.task = {"_target_": "testbed.tasks.PreyPopulation"}
        cfg.reward.outcome_low = 0.0
        cfg.reward.outcome_high = 300.0
    elif design_kind == "unconstrained":
        del cfg.task.design_space
    if design_kind == "unconstrained":
        with pytest.warns(RuntimeWarning, match="unconstrained"):
            trainer = _build_trainer(cfg)
    else:
        trainer = _build_trainer(cfg)
    nets = trainer.nets
    actor_type = (
        CategoricalPolicyActor if design_kind == "finite" else GaussianPolicyActor
    )
    assert type(nets.actor.actor) is actor_type
    actor_encoder = nets.actor.actor.encoder
    critic_encoders = [
        nets.qf1.encoder,
        nets.qf2.encoder,
        nets.qf1_target.encoder,
        nets.qf2_target.encoder,
    ]
    assert isinstance(actor_encoder, actor_encoder_type)
    assert all(isinstance(encoder, critic_encoder_type) for encoder in critic_encoders)
    encoders = [actor_encoder, *critic_encoders]
    parameter_ids = [{id(p) for p in encoder.parameters()} for encoder in encoders]
    for i, ids in enumerate(parameter_ids):
        assert all(ids.isdisjoint(other) for other in parameter_ids[i + 1 :])
    before = [
        [p.detach().clone() for p in encoder.parameters()] for encoder in encoders[:3]
    ]
    trainer.train_step()
    assert torch.isfinite(trainer._actor_loss)
    assert torch.isfinite(trainer._qf_loss)
    for old, encoder in zip(before, encoders):
        assert any(not torch.equal(a, b) for a, b in zip(old, encoder.parameters()))
        assert all(torch.isfinite(p).all() for p in encoder.parameters())
    checkpoint = trainer.build_checkpoint()
    saved_cfg = OmegaConf.create(checkpoint["config"])
    restored = reconstruct_design_net(saved_cfg, trainer.task)
    restored.load_state_dict(checkpoint["design_net"])
    observation = History.empty(
        batch_shape=(2,),
        max_episode_length=cfg.T,
        event_shape_design=trainer.task.event_shape_design,
        event_shape_y=trainer.task.event_shape_y,
        device=trainer.task.device,
    )
    torch.testing.assert_close(
        restored(observation.outcomes, observation.designs, mask=observation.mask),
        nets.actor(observation.outcomes, observation.designs, mask=observation.mask),
    )


@pytest.mark.parametrize("use_time_embedding", [False, True])
def test_attentive_padding_matches_compact_histories_and_has_finite_gradients(
    use_time_embedding,
):
    encoder = AttentiveEncoder(
        (1,),
        (2,),
        dim_emb=16,
        num_heads=2,
        num_blocks=2,
        max_T=3,
        use_time_embedding=use_time_embedding,
    )
    y = torch.randn(3, 5, 1)
    designs = torch.randn(3, 5, 2)
    mask = torch.tensor(
        [
            [False, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
        ]
    )
    # The physical buffer may be longer than max_T when its valid histories fit.
    y[~mask] = float("nan")
    designs[~mask] = float("inf")
    y.requires_grad_()
    designs.requires_grad_()
    actual = encoder(y, designs, mask=mask)
    expected = torch.cat(
        [encoder(y[i : i + 1, mask[i]], designs[i : i + 1, mask[i]]) for i in range(3)]
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().sum().backward()
    assert torch.isfinite(y.grad).all()
    assert torch.isfinite(designs.grad).all()
    assert torch.count_nonzero(y.grad[~mask]) == 0
    assert torch.count_nonzero(designs.grad[~mask]) == 0
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters()
    )


def test_attentive_history_limit_reports_clear_error():
    encoder = AttentiveEncoder((1,), (2,), max_T=2)
    with pytest.raises(ValueError, match="max_T=2"):
        encoder(torch.zeros(1, 3, 1), torch.zeros(1, 3, 2))


@pytest.mark.parametrize("encoder_type", [DeepSetEncoder, AttentiveEncoder])
@pytest.mark.parametrize("discrete", [False, True])
def test_critics_accept_independent_encoder_and_head(encoder_type, discrete):
    encoder = encoder_type((1,), (2,), dim_emb=8)
    head = nn.Linear(8 if discrete else 10, 3 if discrete else 1)
    original_weights = head.weight.detach().clone()
    kwargs = {
        "event_shape_y": (1,),
        "event_shape_design": (2,),
        "design_low": -1.0,
        "design_high": 1.0,
        "outcome_low": 0.0,
        "outcome_high": 1.0,
        "encoder": encoder,
        "head": head,
    }
    critic = (
        RLBOEDDiscreteQNetwork(n_designs=3, **kwargs)
        if discrete
        else RLBOEDQNetwork(**kwargs)
    )
    assert critic.encoder is encoder
    assert critic.emitter_net is head
    torch.testing.assert_close(head.weight, original_weights)
    observation = History.empty((2,), 3, (2,), (1,))
    observation.write(0, torch.zeros(2, 2), torch.ones(2, 1))
    output = (
        critic(*_history(observation), mask=observation.mask)
        if discrete
        else critic(*_history(observation), torch.zeros(2, 2), mask=observation.mask)
    )
    assert output.shape == (2, 3 if discrete else 1)
    output.square().sum().backward()
    assert torch.isfinite(head.weight.grad).all()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters()
    )
