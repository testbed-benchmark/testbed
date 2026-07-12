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
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

RECIPES_DIR = str(Path(__file__).resolve().parents[1] / "testbed" / "recipes")

ENTRY_POINT_RECIPES = ["dad", "actionbed", "rlboed"]


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
    assert cfg.task._target_ == "testbed.tasks.LocationFinding"
    assert cfg.wandb.enabled is False


def test_dad_recipe_wires_design_net():
    cfg = _compose("dad")
    assert cfg.design_net._target_ == "testbed.design_nets.AttentivePolicyNet"
    assert cfg.objective.name == "spce"


def test_actionbed_recipe_wires_design_and_action_nets():
    cfg = _compose("actionbed")
    assert cfg.design_net._target_ == "testbed.design_nets.AttentivePolicyNet"
    assert cfg.action_net._target_ == "testbed.design_nets.AttentivePolicyNet"

    instantiate(
        cfg.design_net,
        event_shape_y=(1,),
        event_shape_design=(2,),
        event_shape_out=(2,),
    )


def test_rlboed_recipe_overrides_shared_task():
    cfg = _compose("rlboed")
    # The recipe overrides the shared 1-source task group to the 2-source variant.
    assert cfg.task.n_sources == 2
    assert cfg.sac.num_envs > 0


def test_temporal_discounting_dad_preserves_objective_defaults():
    cfg = _compose("temporal_discounting/dad")
    assert cfg.objective.reparameterize_policy is False
    assert cfg.objective.requires_policy_log_probs is False


def test_two_source_actionbed_preserves_grad_clip_default():
    cfg = _compose("location_finding_2src/actionbed")
    assert cfg.grad_clip is None
