"""Tests that the Hydra recipes compose and build working networks.

The unit tests build configs by hand, so a broken defaults path, a dropped
`@package` or a misspelled network kwarg in a recipe would otherwise only fail
when a run is launched.
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

import testbed  # registers the auto_device resolver

RECIPES_PATH = Path(__file__).resolve().parents[1] / "testbed" / "recipes"
RECIPES_DIR = str(RECIPES_PATH)


def _recipes(pattern):
    return sorted(
        str(path.relative_to(RECIPES_PATH).with_suffix(""))
        for path in RECIPES_PATH.glob(pattern)
    )


ENTRY_POINT_RECIPES = _recipes("method/*.yaml") + _recipes("experiments/*/*.yaml")


def _compose(recipe, overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=RECIPES_DIR):
        return compose(config_name=recipe, overrides=list(overrides))


@pytest.mark.parametrize("recipe", ENTRY_POINT_RECIPES)
def test_recipe_composes_and_builds_its_task(recipe):
    cfg = _compose(recipe)
    assert cfg.device in {"cpu", "cuda"}
    assert cfg.save_dir
    instantiate(cfg.task)


@pytest.mark.parametrize("recipe", _recipes("experiments/*/posthoc_action.yaml"))
def test_action_recipes_report_losses_that_fit_their_task(recipe):
    """Reported losses must apply to the action net's output.

    A task whose parameters are a set of objects is scored permutation-invariantly
    and no other task is, and the loss the net trains on is among those reported.
    """
    cfg = _compose(recipe)
    reported = {k for k, v in cfg.final_eval.losses.items() if v is not None}
    objects = len(instantiate(cfg.task).event_shape_theta) > 1

    assert all(("permutation_invariant" in k) == objects for k in reported)
    trained = cfg.loss._target_.rsplit(".", 1)[-1].replace("Loss", "").lower()
    assert any(trained in k.replace("_", "") for k in reported)


@pytest.mark.parametrize(
    "recipe", ["method/actionbed", *_recipes("experiments/*/actionbed.yaml")]
)
def test_actionbed_recipes_build_nets_that_fit_the_task(recipe):
    """MLPPolicyNet flattens the whole history, so its `dims` must fit the task."""
    from testbed.methods.actionbed import build_policy_nets

    cfg = _compose(recipe)
    task = instantiate(cfg.task)
    _, action_net = build_policy_nets(cfg, task)

    y = torch.zeros(2, cfg.T, *task.event_shape_y)
    designs = torch.zeros(2, cfg.T, *task.event_shape_design)
    assert action_net(y, designs).shape == (2, *task.event_shape_theta)


@pytest.mark.parametrize(
    "recipe",
    [
        "method/var_post",
        *_recipes("experiments/*/var_post.yaml"),
        *_recipes("experiments/*/joint_var_post.yaml"),
    ],
)
def test_var_post_recipes_build_a_posterior_net_that_fits_the_task(recipe):
    from testbed.methods.var_post import build_posterior_net

    cfg = _compose(recipe)
    task = instantiate(cfg.task)
    net = build_posterior_net(cfg, task)

    transform = getattr(net, "transform", None)
    if transform is not None:
        assert transform.dim_theta == task.event_shape_theta.numel()
        assert transform.dim_raw == net.dim_latent


@pytest.mark.parametrize("actor_architecture", ["deepset", "attention"])
@pytest.mark.parametrize("critic_architecture", ["deepset", "attention"])
def test_rlboed_architecture_overrides(actor_architecture, critic_architecture):
    from testbed.core.setup import instantiate_design_net
    from testbed.design_nets import AttentiveEncoder, DeepSetEncoder

    cfg = _compose(
        "experiments/location_finding_2src/rlboed",
        [
            f"actor_architecture={actor_architecture}",
            f"critic_architecture={critic_architecture}",
        ],
    )
    policy = instantiate_design_net(cfg.design_net, instantiate(cfg.task))
    encoder_type = (
        AttentiveEncoder if actor_architecture == "attention" else DeepSetEncoder
    )
    assert isinstance(policy.actor.encoder, encoder_type)
    if actor_architecture == "attention":
        assert policy.actor.encoder.max_T >= cfg.T


def test_rlboed_rejects_unknown_architecture():
    from hydra.errors import InstantiationException

    from testbed.core.setup import instantiate_design_net

    cfg = _compose(
        "experiments/location_finding/rlboed", ["actor_architecture=unknown"]
    )
    with pytest.raises(InstantiationException, match="architecture must be"):
        instantiate_design_net(cfg.design_net, instantiate(cfg.task))
