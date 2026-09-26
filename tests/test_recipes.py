"""Tests that the Hydra recipes compose into a valid config.

These guard the config-composition machinery introduced by the Hydra
migration — defaults lists, `_self_` ordering, `net@<pkg>` package redirects,
the shared `system`/`task`/`wandb` groups, and the `save_dir` default — none
of which the unit tests (which build configs by hand) exercise. A broken
defaults path or dropped `@package` would otherwise pass CI and only fail when
a user launches `python testbed/methods/<x>.py`.
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

RECIPES_DIR = str(Path(__file__).resolve().parents[1] / "testbed" / "recipes")

ENTRY_POINT_RECIPES = [
    "method/dad",
    "method/actionbed",
    "method/action",
    "method/rlboed",
    "method/var_post",
    "method/joint_var_post",
    "method/static",
    "experiments/benchmarking/location_finding_2src/dad",
    "experiments/benchmarking/location_finding_2src/joint_var_post",
    "experiments/benchmarking/temporal_discounting/dad",
    "experiments/benchmarking/temporal_discounting/joint_var_post",
    "experiments/benchmarking/constant_elasticity_of_substitution/dad",
    "experiments/benchmarking/constant_elasticity_of_substitution/joint_var_post",
    "experiments/benchmarking/michaelis_menten/dad",
    "experiments/benchmarking/michaelis_menten/joint_var_post",
    "experiments/benchmarking/location_finding_2src/joint_var_post_single_gaussian",
    "experiments/benchmarking/temporal_discounting/joint_var_post_single_gaussian",
    "experiments/benchmarking/constant_elasticity_of_substitution/joint_var_post_single_gaussian",
    "experiments/benchmarking/michaelis_menten/joint_var_post_single_gaussian",
    "experiments/benchmarking/pharmacokinetic/dad",
    "experiments/benchmarking/pharmacokinetic/joint_var_post",
    "experiments/benchmarking/pharmacokinetic/joint_var_post_single_gaussian",
    "experiments/benchmarking/cart_pole/dad",
    "experiments/benchmarking/cart_pole/joint_var_post",
    "experiments/benchmarking/cart_pole/joint_var_post_single_gaussian",
    "experiments/benchmarking/double_link/dad",
    "experiments/benchmarking/double_link/joint_var_post",
    "experiments/benchmarking/double_link/joint_var_post_single_gaussian",
    "experiments/benchmarking/stochastic_pendulum/dad",
    "experiments/benchmarking/stochastic_pendulum/joint_var_post",
    "experiments/benchmarking/stochastic_pendulum/joint_var_post_single_gaussian",
    "experiments/benchmarking/location_finding_2src/static",
    "experiments/benchmarking/temporal_discounting/static",
    "experiments/benchmarking/constant_elasticity_of_substitution/static",
    "experiments/benchmarking/cart_pole/static",
    "experiments/benchmarking/double_link/static",
    "experiments/benchmarking/pharmacokinetic/static",
    "experiments/benchmarking/stochastic_pendulum/static",
    "experiments/benchmarking/michaelis_menten/static",
]

BENCHMARK_TASKS = [
    "cart_pole",
    "constant_elasticity_of_substitution",
    "double_link",
    "location_finding_2src",
    "michaelis_menten",
    "pharmacokinetic",
    "stochastic_pendulum",
    "temporal_discounting",
]

# One ActionBED recipe per (task, loss); the whole grid is an entry point.
# truelogmse is undefined on location finding's Gaussian prior (log theta).
ACTIONBED_RECIPES = [
    f"experiments/benchmarking/{task}/actionbed_{loss}"
    for task in BENCHMARK_TASKS
    for loss in ("logmse", "mse", "truelogmse")
    if (task, loss) != ("location_finding_2src", "truelogmse")
]

ENTRY_POINT_RECIPES += ACTIONBED_RECIPES


# Static designs have no design net, and ActionBED runs to a wall-clock budget;
# both sit outside the shared step-budget configuration below.
BENCHMARK_RECIPES = [
    r
    for r in ENTRY_POINT_RECIPES
    if "/benchmarking/" in r and not r.endswith("/static") and "/actionbed_" not in r
]


def _compose(recipe):
    with initialize_config_dir(version_base="1.3", config_dir=RECIPES_DIR):
        return compose(config_name=recipe)


@pytest.mark.parametrize("recipe", ENTRY_POINT_RECIPES)
def test_recipe_composes(recipe):
    """Every entry-point recipe composes with its defaults groups resolved."""
    cfg = _compose(recipe)

    # Shared `system` group is merged in: device plus a non-missing save_dir so
    # a bare run (no `save_dir=` override) still resolves its run directory.
    assert cfg.device
    assert not OmegaConf.is_missing(cfg, "save_dir")
    assert cfg.save_dir

    # Shared task/wandb groups land under their expected keys.
    assert cfg.task._target_ in {
        "testbed.tasks.LocationFinding",
        "testbed.tasks.PreyPopulation",
        "testbed.tasks.CES",
        "testbed.tasks.HyperbolicTemporalDiscounting",
        "testbed.tasks.CartPole",
        "testbed.tasks.DoubleLink",
        "testbed.tasks.PharmacoKinetic",
        "testbed.tasks.StochasticPendulum",
        "testbed.tasks.MichaelisMenten",
    }
    assert cfg.wandb.enabled is False


def test_dad_recipe_wires_design_net():
    cfg = _compose("method/dad")
    assert cfg.design_net._target_ == "testbed.design_nets.AttentivePolicyNet"
    assert cfg.objective.name == "spce"


def test_actionbed_recipe_wires_design_and_action_nets():
    cfg = _compose("method/actionbed")
    assert cfg.design_net._target_ == "testbed.design_nets.AttentivePolicyNet"
    assert cfg.action_net._target_ == "testbed.design_nets.AttentivePolicyNet"

    instantiate(
        cfg.design_net,
        event_shape_y=(1,),
        event_shape_design=(2,),
        event_shape_out=(2,),
    )


def test_action_recipe_wires_a_frozen_designer_and_action_net():
    cfg = _compose("method/action")
    # Designs come from the task's design space until a run dir says otherwise.
    assert cfg.designer._target_ == "testbed.designers.RandomDesigner"
    assert cfg.designer_run_dir is None
    assert cfg.use_raw_history is None

    # The MLP flattens the whole history; build_policy_net refits the first and
    # last widths, but the recipe's own numbers should fit the default task.
    assert cfg.action_net._target_ == "testbed.design_nets.MLPPolicyNet"
    dim_y, dim_design = 1, 2
    assert cfg.action_net.dims[0] == cfg.T * (dim_y + dim_design)

    instantiate(
        cfg.action_net,
        event_shape_y=(dim_y,),
        event_shape_design=(dim_design,),
        event_shape_out=(cfg.action_net.dims[-1],),
    )


def test_rlboed_method_recipe_requires_environment_outcome_bounds():
    cfg = _compose("method/rlboed")
    assert cfg.task.n_sources == 1
    assert cfg.design_net._target_ == "testbed.design_nets.make_stochastic_policy"
    assert cfg.design_net.dim_h == [128, 128]
    assert cfg.design_net.dim_emb == 64
    assert cfg.design_net.log_std_min == -20.0
    assert cfg.design_net.log_std_max == 0.0
    assert cfg.design_net.init_std == pytest.approx(3**-0.5)
    assert cfg.design_net.normalize_design_history is True
    assert cfg.design_net.design_history_bounds == [0.0, 1.0]
    assert OmegaConf.is_missing(cfg.reward, "outcome_low")
    assert OmegaConf.is_missing(cfg.reward, "outcome_high")
    assert "design_space" not in cfg
    assert cfg.sac.num_envs > 0


def test_two_source_rlboed_recipe_uses_two_source_task():
    cfg = _compose("experiments/benchmarking/location_finding_2src/rlboed")
    assert cfg.task.n_sources == 2
    assert cfg.T == 30
    assert "design_space" not in cfg
    assert (cfg.task.design_space.low, cfg.task.design_space.high) == (-4.0, 4.0)
    assert (cfg.reward.outcome_low, cfg.reward.outcome_high) == (-3.0, 10.0)


def test_var_post_recipe_wires_posterior_net():
    cfg = _compose("method/var_post")
    assert (
        cfg.posterior_net._target_
        == "testbed.design_nets.AttentiveVariationalPosteriorNet"
    )
    assert cfg.posterior_net.n_components == 10
    assert cfg.joint is False
    assert cfg.objective.name == "barber_agakov"


def test_joint_var_post_recipe_sets_joint_true():
    cfg = _compose("method/joint_var_post")
    assert cfg.joint is True
    assert cfg.posterior_net.n_components == 10


@pytest.mark.parametrize("actor_architecture", ["deepset", "attention"])
@pytest.mark.parametrize("critic_architecture", ["deepset", "attention"])
def test_rlboed_architecture_overrides(actor_architecture, critic_architecture):
    from testbed.core.setup import instantiate_design_net
    from testbed.design_nets import AttentiveEncoder, DeepSetEncoder

    with initialize_config_dir(version_base="1.3", config_dir=RECIPES_DIR):
        cfg = compose(
            config_name="experiments/benchmarking/location_finding_2src/rlboed",
            overrides=[
                f"actor_architecture={actor_architecture}",
                f"critic_architecture={critic_architecture}",
            ],
        )
    task = instantiate(cfg.task)
    policy = instantiate_design_net(cfg.design_net, task)
    encoder_type = (
        AttentiveEncoder if actor_architecture == "attention" else DeepSetEncoder
    )
    assert isinstance(policy.actor.encoder, encoder_type)
    assert cfg.design_net.architecture == actor_architecture
    assert cfg.critic_architecture == critic_architecture
    if actor_architecture == "attention":
        # The shared attention preset uses AttentiveEncoder's default capacity.
        assert policy.actor.encoder.max_T >= cfg.T


def test_rlboed_rejects_unknown_architecture():
    from hydra.errors import InstantiationException

    from testbed.core.setup import instantiate_design_net

    with initialize_config_dir(version_base="1.3", config_dir=RECIPES_DIR):
        cfg = compose(
            config_name="experiments/benchmarking/location_finding_2src/rlboed",
            overrides=["actor_architecture=unknown"],
        )
    with pytest.raises(InstantiationException, match="architecture must be"):
        instantiate_design_net(cfg.design_net, instantiate(cfg.task))


def test_ces_joint_var_post_recipe_uses_a_constrained_posterior():
    cfg = _compose(
        "experiments/benchmarking/constant_elasticity_of_substitution/joint_var_post"
    )

    assert cfg.task._target_ == "testbed.tasks.CES"
    assert (
        cfg.posterior_net._target_
        == "testbed.design_nets.ConstrainedAttentiveVariationalPosteriorNet"
    )
    # 4 unconstrained coordinates -> (rho, alpha_1, alpha_2, alpha_3, u).
    assert cfg.posterior_net.transform.raw_sizes == [1, 2, 1]
    assert cfg.posterior_net.transform.theta_sizes == [1, 3, 1]
    assert cfg.posterior_net.canonicalize_theta is False


def test_htd_joint_var_post_recipe_uses_a_constrained_posterior():
    cfg = _compose("experiments/benchmarking/temporal_discounting/joint_var_post")

    assert cfg.task._target_ == "testbed.tasks.HyperbolicTemporalDiscounting"
    assert (
        cfg.posterior_net._target_
        == "testbed.design_nets.ConstrainedAttentiveVariationalPosteriorNet"
    )
    # Two width-1 blocks: exp on k, identity on alpha. alpha must NOT go
    # through a log; that is what NaN'd an earlier sweep.
    assert cfg.posterior_net.transform.raw_sizes == [1, 1]
    assert cfg.posterior_net.transform.theta_sizes == [1, 1]
    targets = [t._target_ for t in cfg.posterior_net.transform.transforms]
    assert targets == [
        "torch.distributions.transforms.ExpTransform",
        "torch.distributions.transforms.AffineTransform",
    ]
    assert cfg.posterior_net.canonicalize_theta is False


def test_htd_posterior_transform_matches_the_task_theta_width():
    from testbed.methods.var_post import build_posterior_net

    cfg = _compose("experiments/benchmarking/temporal_discounting/joint_var_post")
    task = instantiate(cfg.task)
    net = build_posterior_net(cfg, task)

    assert net.transform.dim_theta == task.event_shape_theta.numel() == 2
    assert net.transform.dim_raw == net.dim_latent == 2
    assert net.loc_head.out_features == cfg.posterior_net.n_components * 2


def test_var_post_recipes_actually_instantiate_their_posterior_net():
    """Builds the nets, rather than only asserting on `_target_` strings.

    Every other test here checks the composed config without instantiating, so a
    posterior_net block with a misspelled kwarg composes cleanly and passes. This
    is the only thing that catches it.
    """
    from testbed.methods.var_post import build_posterior_net

    recipes = [
        "method/var_post",
        "method/joint_var_post",
        *[r for r in BENCHMARK_RECIPES if r.endswith("/joint_var_post")],
    ]
    for recipe in recipes:
        cfg = _compose(recipe)
        task = instantiate(cfg.task)
        net = build_posterior_net(cfg, task)
        assert sum(p.numel() for p in net.parameters()) > 0, recipe


# DAD on these tasks OOMs at the suite's 2_048; see their recipes.
HALF_BATCH_RECIPES = {
    "experiments/benchmarking/cart_pole/dad",
    "experiments/benchmarking/double_link/dad",
}


@pytest.mark.parametrize("recipe", BENCHMARK_RECIPES)
def test_benchmark_recipes_share_one_configuration(recipe):
    """The benchmarking suite varies the task and the method, nothing else.

    Its whole point is that a table built from it compares methods rather than
    architectures or budgets, which holds only while these agree.

    n_train_iter is deliberately not asserted to agree. The shared budget is the
    wall clock (see the test below); n_train_iter is only the cap that must not
    bind before it, and how many steps 12h buys differs by an order of magnitude
    across these tasks. Pinning it to one number is what let recipes end on the
    iteration limit at ~6h while claiming a 12h budget.
    """
    cfg = _compose(recipe)

    assert cfg.design_net._target_ == "testbed.design_nets.AttentivePolicyNet"
    assert cfg.n_outer == (1_024 if recipe in HALF_BATCH_RECIPES else 2_048)
    task = recipe.split("/")[-2]
    assert cfg.wandb.project == f"benchmarking_{task}"


def test_benchmark_recipes_share_one_training_budget():
    """Every benchmark run gets the same wall clock and the same checkpoint cadence.

    Asserted as agreement rather than against fixed numbers, so retuning the
    budget stays a one-line change per recipe.
    """
    budgets = {r: _compose(r).train_time_budget_seconds for r in BENCHMARK_RECIPES}
    intervals = {r: _compose(r).checkpoint_every for r in BENCHMARK_RECIPES}

    assert len(set(budgets.values())) == 1, budgets
    assert len(set(intervals.values())) == 1, intervals
    # A null budget here would mean the suite is iteration-bound again.
    assert next(iter(budgets.values())) is not None


@pytest.mark.parametrize("recipe", ACTIONBED_RECIPES)
def test_actionbed_benchmark_recipes_share_one_configuration(recipe):
    """The ActionBED grid varies the task and the loss, nothing else."""
    cfg = _compose(recipe)
    task, loss = recipe.split("/")[-2], recipe.split("_")[-1]

    assert cfg.design_net._target_ == "testbed.design_nets.AttentivePolicyNet"
    assert cfg.action_net._target_ == "testbed.design_nets.MLPPolicyNet"
    assert cfg.train_time_budget_seconds == 43_200
    assert cfg.n_action_warmup_iter == 25_000
    assert cfg.n_outer == 2_000
    assert cfg.eval_every == 0  # the final_eval block handles it
    assert cfg.wandb.project == f"benchmarking_{task}"
    assert cfg.wandb.group.startswith(f"actionbed_{loss}_big_attention")


@pytest.mark.parametrize("recipe", ACTIONBED_RECIPES)
def test_actionbed_recipes_size_their_action_net_to_the_task(recipe):
    """Runs the nets, rather than only asserting on the composed config.

    MLPPolicyNet flattens the whole history, so its first and last `dims` are
    fixed by the task. A stale pair composes cleanly and only fails on the
    first training step; this is the only thing that catches it.
    """
    from testbed.methods.actionbed import build_policy_nets

    cfg = _compose(recipe)
    task = instantiate(cfg.task)
    _, action_net = build_policy_nets(cfg, task)

    y = torch.zeros(2, cfg.T, *task.event_shape_y)
    designs = torch.zeros(2, cfg.T, *task.event_shape_design)
    assert action_net(y, designs).shape == (2, *task.action_space.event_shape)


def test_ces_posterior_transform_matches_the_task_theta_width():
    """The recipe's transform must emit exactly CES's 5 theta coordinates.

    Instantiating the net is what enforces it; the constructor raises otherwise.
    """
    from testbed.methods.var_post import build_posterior_net

    cfg = _compose(
        "experiments/benchmarking/constant_elasticity_of_substitution/joint_var_post"
    )
    task = instantiate(cfg.task)
    net = build_posterior_net(cfg, task)

    assert net.transform.dim_theta == task.event_shape_theta.numel() == 5
    assert net.transform.dim_raw == net.dim_latent == 4
    assert net.loc_head.out_features == cfg.posterior_net.n_components * 4
