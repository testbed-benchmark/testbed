"""Tests for the actions-only trainer on top of a frozen designer."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from testbed.core.manager import Manager
from testbed.design_nets import DeepSetPolicyNet
from testbed.design_objectives import MSELoss
from testbed.designers import (
    Designer,
    PolicyDesigner,
    RandomDesigner,
    StaticDesigner,
)
from testbed.methods.posthoc_action import Trainer, adopt_run_task, build_policy_net
from testbed.tasks import LocationFinding

T = 3


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_net(task, event_shape_out):
    return DeepSetPolicyNet(
        event_shape_y=task.event_shape_y,
        event_shape_design=task.event_shape_design,
        event_shape_out=event_shape_out,
    )


def _make_designer(kind, task, *, use_raw_history=False):
    if kind == "random":
        return RandomDesigner(task, T)
    if kind == "static":
        return StaticDesigner(task, T)
    return PolicyDesigner(
        task,
        T,
        design_net=_make_net(task, task.event_shape_design),
        use_raw_history=use_raw_history,
    )


def _make_trainer(designer, task, *, use_raw_history=None):
    action_net = _make_net(task, task.event_shape_theta)
    cfg = SimpleNamespace(
        n_outer=4,
        n_outer_eval=4,
        n_train_iter=2,
        eval_every=0,
        grad_clip=None,
        use_raw_history=use_raw_history,
    )
    optimizer = torch.optim.Adam(action_net.parameters(), lr=1e-2)
    trainer = Trainer(task, designer, action_net, MSELoss(), optimizer, cfg)
    return trainer, action_net


@pytest.mark.parametrize("kind", ["random", "static", "policy"])
def test_any_designer_trains_the_action_net_alone(kind):
    task = _make_task()
    designer = _make_designer(kind, task)
    trainer, action_net = _make_trainer(designer, task)

    # Skip the hooks, which own the action net that is meant to change.
    design_side = [k for k in designer.state_dict() if not k.startswith("hooks.")]
    before = copy.deepcopy(designer.state_dict())
    action_before = copy.deepcopy(action_net.state_dict())
    for _ in range(2):
        trainer.train_step()

    after = designer.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in design_side)
    assert any(
        not torch.equal(action_before[k], action_net.state_dict()[k])
        for k in action_before
    )


def test_designer_is_frozen_but_the_task_is_untouched():
    task = _make_task()
    designer = _make_designer("policy", task)
    designer.design_net.train()
    _make_trainer(designer, task)

    assert not any(p.requires_grad for p in designer.design_net.parameters())
    assert not designer.design_net.training
    assert task.training


@pytest.mark.parametrize("use_raw_history", [False, True])
def test_action_net_inherits_the_designers_history_convention(use_raw_history):
    task = _make_task()
    designer = _make_designer("policy", task, use_raw_history=use_raw_history)
    trainer, _ = _make_trainer(designer, task)

    assert trainer.use_raw_history is use_raw_history
    assert trainer.designer.hooks[-1].use_raw_history is use_raw_history


def test_cfg_overrides_the_designers_history_convention():
    task = _make_task()
    designer = _make_designer("policy", task, use_raw_history=True)
    trainer, _ = _make_trainer(designer, task, use_raw_history=False)

    assert trainer.use_raw_history is False


def test_history_convention_defaults_to_physical_without_either():
    task = _make_task()
    designer = RandomDesigner(task, T)
    action_net = _make_net(task, task.event_shape_theta)
    trainer = Trainer(
        task,
        designer,
        action_net,
        MSELoss(),
        torch.optim.Adam(action_net.parameters()),
        SimpleNamespace(),
    )
    assert trainer.use_raw_history is False


@pytest.mark.parametrize(
    ("kind", "present", "absent"),
    [("policy", "design_net", "raw_design"), ("static", "raw_design", "design_net")],
)
def test_checkpoint_carries_the_frozen_designers_state(kind, present, absent):
    task = _make_task()
    designer = _make_designer(kind, task)
    trainer, _ = _make_trainer(designer, task)

    state = trainer._policy_state(0)
    assert state[present] is not None
    assert state[absent] is None
    if kind == "static":
        assert torch.equal(state["raw_design"], designer.raw_design.detach().cpu())


def test_run_dir_supplies_the_task_and_T(tmp_path):
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "T": T,
                "task": {"_target_": "testbed.tasks.LocationFinding", "n_sources": 1},
            }
        ),
        tmp_path / ".hydra" / "config.yaml",
    )
    cfg = OmegaConf.create({"designer_run_dir": str(tmp_path), "task": {}, "T": 99})

    adopt_run_task(cfg)

    assert cfg.T == T
    assert cfg.task.n_sources == 1


def test_action_net_dims_are_refitted_to_the_task():
    task = _make_task()
    cfg = OmegaConf.create(
        {
            "T": T,
            "action_net": {
                "_target_": "testbed.design_nets.MLPPolicyNet",
                "dims": [1, 16, 1],  # placeholders, for some other task
            },
        }
    )

    net = build_policy_net(cfg, task)

    y = torch.zeros(2, T, *task.event_shape_y)
    designs = torch.zeros(2, T, *task.event_shape_design)
    assert net(y, designs).shape == (2, *task.event_shape_theta)
    # The recipe is left as written; the shapes live in the built net.
    assert cfg.action_net.dims == [1, 16, 1]


def test_trainer_rejects_a_non_designer():
    task = _make_task()
    action_net = _make_net(task, task.event_shape_theta)
    with pytest.raises(AssertionError):
        Trainer(
            task,
            object(),
            action_net,
            MSELoss(),
            torch.optim.Adam(action_net.parameters()),
            SimpleNamespace(use_raw_history=None),
        )


def test_a_designer_that_skips_its_hooks_says_so():
    # AlineDesigner overrides rollout() without running hooks, so no action is
    # computed; a bare KeyError would not say why.
    task = _make_task()

    class HooklessDesigner(RandomDesigner):
        def rollout(self, theta):
            return Designer.rollout(RandomDesigner(self.task, self.T), theta)

    trainer, _ = _make_trainer(HooklessDesigner(task, T), task)

    with pytest.raises(RuntimeError, match="never fired"):
        trainer.train_step()


def test_train_scores_the_checkpoint_it_saved(tmp_path):
    task = _make_task()
    designer = _make_designer("random", task)
    action_net = _make_net(task, task.event_shape_theta)
    cfg = OmegaConf.create(
        {
            "T": T,
            "device": "cpu",
            "n_outer": 4,
            "n_outer_eval": 4,
            "n_train_iter": 1,
            "eval_every": 0,
            "grad_clip": None,
            "use_raw_history": None,
            "task": {
                "_target_": "testbed.tasks.LocationFinding",
                "n_sources": 1,
                "dim_theta": 2,
            },
            "designer": {"_target_": "testbed.designers.RandomDesigner"},
            "action_net": {"_target_": "testbed.design_nets.DeepSetPolicyNet"},
            "final_eval": {
                "enabled": True,
                "checkpoint": "final_step.pt",
                "n_outer": 4,
                "outer_batch_size": 4,
                "losses": {"mse": {"_target_": "testbed.design_objectives.MSELoss"}},
            },
        }
    )
    manager = Manager(cfg, root=str(tmp_path), name="run", timestamp=False)
    trainer = Trainer(
        task,
        designer,
        action_net,
        MSELoss(),
        torch.optim.Adam(action_net.parameters()),
        cfg,
        manager=manager,
    )

    trainer.train()

    results = json.loads((tmp_path / "run" / "eval.json").read_text())
    assert results["n_outer"] == 4
    assert results["mse"] > 0


def test_train_skips_the_final_eval_when_it_is_disabled(tmp_path):
    task = _make_task()
    action_net = _make_net(task, task.event_shape_theta)
    cfg = SimpleNamespace(
        n_outer=4,
        n_outer_eval=4,
        n_train_iter=1,
        eval_every=0,
        grad_clip=None,
        use_raw_history=None,
    )
    manager = Manager(cfg, root=str(tmp_path), name="off", timestamp=False)
    Trainer(
        task,
        _make_designer("random", task),
        action_net,
        MSELoss(),
        torch.optim.Adam(action_net.parameters()),
        cfg,
        manager=manager,
    ).train()

    assert not (tmp_path / "off" / "eval.json").exists()
