"""Tests for the RL-BOED (SAC) training components."""

import json
from math import log

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from testbed.core.design_spaces import BoxDesignSpace, FiniteDesignSpace
from testbed.core.manager import Manager
from testbed.design_nets import (
    CategoricalPolicyNet,
    DeepSetCategoricalActor,
    DeepSetEncoder,
    DeepSetGaussianActor,
    GaussianPolicyNet,
    RLBOEDDiscreteQNetwork,
    RLBOEDQNetwork,
)
from testbed.designers import PolicyDesigner
from testbed.envs.history import History
from testbed.eval.eval_policy import (
    build_design_net as reconstruct_design_net,
)
from testbed.eval.eval_policy import (
    evaluate_checkpoint,
)
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


# ============================================================================
# Network architecture
# ============================================================================


def test_actor_initializes_requested_standard_deviation():
    actor = DeepSetGaussianActor(
        event_shape_y=(1,), event_shape_design=(2,), init_std=0.4
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
        DeepSetGaussianActor(
            event_shape_y=(1,), event_shape_design=(2,), init_std=init_std
        )


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

    actual = critic(observation, design)
    permuted = critic(permuted_observation, design)

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

    actual = critic(observation, design)
    modified = critic(modified_observation, design)

    assert actual.shape == (2, 1)
    torch.testing.assert_close(actual, modified)


def test_soft_q_network_uses_reference_history_and_design_coordinates():
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

    critic.encoder.encoder_net.register_forward_pre_hook(record_encoder_input)
    critic.emitter_net.register_forward_pre_hook(record_emitter_input)
    physical_design = torch.tensor([[1.0, 4.0]])
    observation = History(
        max_episode_length=1,
        designs=physical_design.unsqueeze(1),
        outcomes=torch.tensor([[[3.5]]]),
        mask=torch.tensor([[True]]),
    )

    critic(observation, physical_design)

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
    actor = DeepSetGaussianActor(event_shape_y=(1,), event_shape_design=(2,))
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
    actor = DeepSetCategoricalActor(
        event_shape_y=(1,), event_shape_design=(1,), n_designs=3
    )
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
        dim_y=1,
        dim_design=1,
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

    values = critic(observation)

    assert values.shape == (2, 3)
    torch.testing.assert_close(values, critic(permuted))


# ============================================================================
# Replay buffer
# ============================================================================


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


# ============================================================================
# Buffer-device resolution
# ============================================================================


def test_rlboed_rejects_unavailable_cuda_buffer_device():
    if torch.cuda.is_available():
        assert _resolve_buffer_device("cuda").type == "cuda"
    else:
        with pytest.raises(ValueError, match="requires CUDA"):
            _resolve_buffer_device("cuda")


def test_rlboed_rejects_gpu_buffer_device_alias():
    with pytest.raises(ValueError, match="cpu.*cuda"):
        _resolve_buffer_device("gpu")


# ============================================================================
# Trainer
# ============================================================================


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
            "design_net": {
                "_target_": "testbed.design_nets.make_deepset_stochastic_policy",
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
    assert (run_dir / "best.pt").exists()
    best = torch.load(run_dir / "best.pt", weights_only=True)
    assert "design_net" in best
    assert "designer" not in best
    assert "actor" not in best

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

    checkpoint = trainer.build_checkpoint(step=cfg.sac.num_training_iterations - 1)
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
