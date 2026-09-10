import typing

import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace, FiniteDesignSpace
from testbed.envs import (
    BEDEnv,
    CallableReward,
    History,
    RLBOEDDenseReward,
    RLBOEDTerminalReward,
    ZeroReward,
)
from testbed.tasks import LocationFinding


def make_task(design_space=None):
    if design_space is None:
        design_space = BoxDesignSpace(-1.0, 1.0, event_shape=(2,))
    return LocationFinding(
        n_sources=1,
        dim_theta=2,
        noise_scale=0.2,
        design_space=design_space,
    )


# BoxDesignSpace's own behaviour is covered in tests/test_design_spaces.py;
# what follows exercises how BEDEnv uses a design space.


def test_bed_env_rejects_invalid_categorical_designs_without_projection():
    task = make_task(FiniteDesignSpace(torch.tensor([[-1.0, -1.0], [1.0, 1.0]])))
    # FiniteDesignSpace.project snaps to the nearest valid design, so an invalid
    # design only reaches validation with projection turned off.
    env = BEDEnv(
        task,
        horizon=1,
        batch_shape=(1,),
        project_designs=False,
    )
    env.reset()

    with pytest.raises(ValueError, match="outside the design space"):
        env.step(torch.tensor([[0.5, 0.5]]))


def test_bed_env_init_type_hints_resolve_at_runtime():
    assert "task" in typing.get_type_hints(BEDEnv.__init__)


def test_history_empty_write_append_prefix_and_bounds():
    history = History.empty(
        batch_shape=(2,),
        max_episode_length=3,
        event_shape_design=(2,),
        event_shape_y=(1,),
        design_dtype=torch.float64,
    )

    assert history.batch_shape == torch.Size([2])
    assert history.T == 0
    assert history.designs.dtype == torch.float64

    design0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    outcome0 = torch.tensor([[0.5], [1.5]])
    history.write(0, design0, outcome0)
    assert history.T == 1
    assert torch.equal(history.designs[:, 0], design0)
    assert torch.equal(history.outcomes[:, 0], outcome0)
    assert torch.equal(
        history.mask, torch.tensor([[True, False, False], [True, False, False]])
    )

    design1 = torch.zeros(2, 2, dtype=torch.float64)
    outcome1 = torch.ones(2, 1)
    history.write(1, design1, outcome1)
    assert history.T == 2

    prefix = history.prefix(1)
    assert prefix.max_episode_length == 1
    assert prefix.designs.shape == (2, 1, 2)
    assert torch.equal(prefix.mask, torch.tensor([[True], [True]]))

    with pytest.raises(IndexError):
        history.write(-1, design0, outcome0)
    with pytest.raises(IndexError):
        history.write(3, design0, outcome0)
    with pytest.raises(ValueError):
        History.empty((), 0, (), ())


def test_bed_env_reset_and_step_shapes():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=3,
        reward_fn=RLBOEDDenseReward(n_contrastive=2),
        batch_shape=(5,),
    )

    obs, info = env.reset(seed=123)
    assert info["t"] == 0
    assert obs.designs.shape == (5, 3, 2)
    assert obs.outcomes.shape == (5, 3, 1)
    assert obs.mask.shape == (5, 3)
    assert env.reward_fn.theta_all.shape == (5, 3, 1, 2)

    design = torch.zeros(5, 2)
    obs, reward, terminated, truncated, info = env.step(design)
    assert reward.shape == (5,)
    assert terminated.shape == (5,)
    assert not terminated.any()
    assert not truncated.any()
    assert info["history"].designs.shape == (5, 3, 2)
    assert info["history"].mask.shape == (5, 3)
    assert obs.mask.sum().item() == 5


def test_bed_env_requires_reset_and_rejects_invalid_horizon():
    task = make_task()

    with pytest.raises(ValueError):
        BEDEnv(task, horizon=0)

    env = BEDEnv(task, horizon=1)
    with pytest.raises(RuntimeError):
        _ = env.history
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(2))


def test_bed_env_reset_accepts_explicit_theta_and_info_options():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=2,
        reward_fn=RLBOEDDenseReward(n_contrastive=1),
        batch_shape=(2,),
        include_theta_in_info=True,
    )
    theta = torch.zeros(2, 1, 2)

    _, info = env.reset(theta=theta, options={"tag": "episode"})

    assert info["options"] == {"tag": "episode"}
    assert torch.equal(info["theta_0"], theta)
    assert env.reward_fn.theta_all.shape == (2, 2, 1, 2)
    assert torch.equal(env.theta_0, theta)


def test_bed_env_reset_accepts_theta_all_and_batch_shape_override():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=1,
        reward_fn=RLBOEDDenseReward(n_contrastive=2),
        batch_shape=(1,),
    )
    theta_all = torch.zeros(4, 3, 1, 2)

    obs, _ = env.reset(theta_all=theta_all, batch_shape=(4,))

    assert env.batch_shape == torch.Size([4])
    assert torch.equal(env.reward_fn.theta_all, theta_all)
    assert torch.equal(env.theta_0, theta_all[:, 0])
    assert obs.designs.shape == (4, 1, 2)


def test_bed_env_reset_rejects_bad_theta_shapes():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=1,
        reward_fn=RLBOEDDenseReward(n_contrastive=1),
        batch_shape=(2,),
    )

    with pytest.raises(ValueError, match="theta must have shape"):
        env.reset(theta=torch.zeros(1, 2))
    with pytest.raises(ValueError, match="theta_all must have shape"):
        env.reset(theta_all=torch.zeros(2, 1, 1, 2))


def test_bed_env_reset_rejects_bad_theta_all_shape_with_zero_reward():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=1,
        reward_fn=ZeroReward(),
        batch_shape=(2,),
    )

    with pytest.raises(ValueError, match="theta_all must have shape"):
        env.reset(theta_all=torch.zeros(2, 2, 1, 2))


def test_bed_env_design_validation_without_projection():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=1,
        reward_fn=ZeroReward(),
        batch_shape=(2,),
        project_designs=False,
    )
    env.reset(seed=0)

    with pytest.raises(ValueError, match="outside the design space"):
        env.step(torch.tensor([[2.0, 0.0], [0.0, 0.0]]))


def test_bed_env_rejects_wrong_design_shape_after_projection():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=1,
        reward_fn=ZeroReward(),
        batch_shape=(2,),
    )
    env.reset(seed=0)

    with pytest.raises(ValueError, match="Design must have shape"):
        env.step(torch.zeros(2))


def test_bed_env_can_reset_after_termination():
    task = make_task()
    env = BEDEnv(task, horizon=1, reward_fn=ZeroReward())

    env.reset(seed=0)
    env.step(torch.zeros(2))
    assert env.terminated

    obs, info = env.reset(seed=1)
    assert info["t"] == 0
    assert not env.terminated
    assert obs.mask.sum() == 0


def test_bed_env_projects_box_designs():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=1,
        reward_fn=ZeroReward(),
        batch_shape=(2,),
    )
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(torch.tensor([[2.0, -3.0], [0.5, 0.0]]))

    assert terminated.all()
    assert torch.all(info["design"] <= 1.0)
    assert torch.all(info["design"] >= -1.0)
    with pytest.raises(RuntimeError):
        env.step(torch.zeros(2, 2))


def test_bed_env_step_detaches_rollout_tensors_by_default():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=1,
        reward_fn=CallableReward(lambda env, design, outcome: design.sum(dim=-1)),
        project_designs=False,
    )
    design = torch.zeros(2, requires_grad=True)

    env.reset(seed=0)
    obs, reward, _, _, info = env.step(design)

    assert design.requires_grad
    assert not reward.requires_grad
    assert not obs.designs.requires_grad
    assert not obs.outcomes.requires_grad
    assert not info["raw_design"].requires_grad
    assert not info["design"].requires_grad
    assert not info["outcome"].requires_grad


def test_bed_env_scalar_batch_shape():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=1,
        reward_fn=ZeroReward(),
    )

    obs, _ = env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(torch.zeros(2))

    assert obs.designs.shape == (1, 2)
    assert reward.shape == ()
    assert terminated.shape == ()
    assert terminated
    assert not truncated
    assert info["history"].outcomes.shape == (1, 1)
    assert info["history"].mask.shape == (1,)


def test_dense_reward_sum_matches_terminal_reward_for_same_rollout():
    task = make_task()
    horizon = 4
    batch_shape = (6,)
    n_contrastive = 3

    dense_env = BEDEnv(
        task,
        horizon=horizon,
        reward_fn=RLBOEDDenseReward(n_contrastive),
        batch_shape=batch_shape,
    )
    terminal_env = BEDEnv(
        task,
        horizon=horizon,
        reward_fn=RLBOEDTerminalReward(n_contrastive),
        batch_shape=batch_shape,
    )

    dense_env.reset(seed=11)
    theta_all = dense_env.reward_fn.theta_all.detach().clone()
    terminal_env.reset(theta_all=theta_all)

    generator = torch.Generator().manual_seed(22)
    designs = [
        -1.0
        + 2.0 * torch.rand(*batch_shape, *task.event_shape_design, generator=generator)
        for _ in range(horizon)
    ]

    dense_rewards = []
    terminal_rewards = []
    for design in designs:
        torch.manual_seed(33 + len(dense_rewards))
        _, reward, _, _, dense_info = dense_env.step(design)
        dense_rewards.append(reward)

        terminal_env.history.write(
            terminal_env.t, dense_info["design"], dense_info["outcome"]
        )
        terminal_rewards.append(
            terminal_env.reward_fn.step(
                terminal_env, dense_info["design"], dense_info["outcome"]
            )
        )
        terminal_env.t += 1

    assert torch.allclose(sum(dense_rewards), terminal_rewards[-1], atol=1e-5)
    assert all(torch.allclose(r, torch.zeros_like(r)) for r in terminal_rewards[:-1])


def test_contrastive_rewards_handle_zero_contrastive_particles():
    task = make_task()
    dense_env = BEDEnv(
        task,
        horizon=2,
        reward_fn=RLBOEDDenseReward(n_contrastive=0),
        batch_shape=(3,),
    )
    terminal_env = BEDEnv(
        task,
        horizon=2,
        reward_fn=RLBOEDTerminalReward(n_contrastive=0),
        batch_shape=(3,),
    )

    dense_env.reset(seed=5)
    terminal_env.reset(theta_all=dense_env.reward_fn.theta_all.detach().clone())

    for _ in range(2):
        _, dense_reward, _, _, dense_info = dense_env.step(torch.zeros(3, 2))
        terminal_env.history.write(
            terminal_env.t, dense_info["design"], dense_info["outcome"]
        )
        terminal_reward = terminal_env.reward_fn.step(
            terminal_env, dense_info["design"], dense_info["outcome"]
        )
        terminal_env.t += 1
        assert torch.allclose(dense_reward, torch.zeros_like(dense_reward), atol=1e-6)

    assert torch.allclose(terminal_reward, torch.zeros_like(terminal_reward), atol=1e-6)


def test_reward_error_paths_and_callable_reward():
    task = make_task()
    env = BEDEnv(
        task,
        horizon=1,
        reward_fn=CallableReward(lambda env, design, outcome: design.sum(dim=-1)),
        batch_shape=(2,),
    )

    env.reset(seed=0)
    _, reward, _, _, _ = env.step(torch.tensor([[0.25, 0.5], [-0.5, 1.0]]))
    assert torch.equal(reward, torch.tensor([0.75, 0.5]))
    assert env.reward_fn.requires_likelihood

    with pytest.raises(ValueError):
        RLBOEDDenseReward(n_contrastive=-1)

    dense_reward = RLBOEDDenseReward(n_contrastive=1)
    with pytest.raises(RuntimeError, match="Reward function has not been reset"):
        dense_reward.step(env, torch.zeros(2, 2), torch.zeros(2, 1))

    terminal_reward = RLBOEDTerminalReward(n_contrastive=1)
    with pytest.raises(RuntimeError, match="Reward function has not been reset"):
        terminal_reward.step(env, torch.zeros(2, 2), torch.zeros(2, 1))


def test_step_conditions_on_the_history_for_a_task_with_memory():
    # The env writes into a padded History but used to sample each outcome with
    # no history, silently simulating a memoryless process. Pin it against the
    # task's own recursion: y_t = make_mean(theta, x, y_{<t}) + sqrt(dt)*sigma*eps.
    import math

    from testbed.tasks import StochasticPendulum

    task = StochasticPendulum()
    assert not task.memoryless
    design = torch.zeros(1, 1)  # fixed, so any drift comes from the history

    torch.manual_seed(0)
    env = BEDEnv(task, horizon=6, batch_shape=(1,))
    env.reset()
    theta = env.theta_0

    torch.manual_seed(7)
    for _ in range(6):
        env.step(design)

    torch.manual_seed(7)
    previous, expected = torch.zeros(1, 2), []
    for _ in range(6):
        mean = task.make_mean(theta, design, previous.unsqueeze(1))
        previous = mean + math.sqrt(task.dt) * task.noise_scale * torch.randn_like(mean)
        expected.append(previous[0])

    assert torch.allclose(env.history.outcomes[0], torch.stack(expected), atol=1e-6)


def test_step_is_unchanged_for_a_memoryless_task():
    # A memoryless task drops the history, so threading it through must not
    # perturb the common path.
    task = make_task()
    design = torch.zeros(1, 2)

    torch.manual_seed(0)
    env = BEDEnv(task, horizon=4, batch_shape=(1,))
    env.reset()
    theta = env.theta_0

    torch.manual_seed(3)
    for _ in range(4):
        env.step(design)

    torch.manual_seed(3)
    direct = torch.stack([task.sample_outcome(theta, design)[0] for _ in range(4)])

    assert torch.equal(env.history.outcomes[0], direct)
