"""Tests for StaticDesigner, including the transform_designs flag."""

import pytest
import torch
from torch import nn

from testbed.designers import ActionHook, RolloutHook, StaticDesigner
from testbed.tasks import HyperbolicTemporalDiscounting, LocationFinding


def test_rollout_shapes():
    task = LocationFinding(n_sources=1, dim_theta=2)
    b, T = 5, 3
    designer = StaticDesigner(task, T=T)
    theta = task.sample_prior(b)

    rollout = designer.rollout(theta)

    assert rollout.design_history.shape == (b, T, *task.event_shape_design)
    assert rollout.y_history.shape == (b, T, *task.event_shape_y)


def test_raw_design_is_always_exposed():
    # The designer computes a raw design either way, so it is always reported;
    # callers that don't need it can ignore the key.
    task = HyperbolicTemporalDiscounting()
    designer = StaticDesigner(task, T=3)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    assert "raw_design_history" in rollout.extra


def test_raw_design_history_matches_design_parameter():
    task = HyperbolicTemporalDiscounting()
    designer = StaticDesigner(task, T=3)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    raw_design_history = rollout.extra["raw_design_history"]
    assert raw_design_history.shape == (4, 3, 2)
    # Every batch entry mirrors the same learnable raw parameter.
    assert torch.allclose(raw_design_history[0], designer.raw_design)
    assert torch.allclose(raw_design_history[1], designer.raw_design)
    # The exposed raw design is what design_history is the forward transform of.
    assert torch.allclose(
        task.design_space.to_design(raw_design_history), rollout.design_history
    )


def test_raw_design_agrees_between_vectorized_and_per_step_paths():
    # StaticDesigner.rollout() has a vectorized fast path (no hooks) and a
    # per-step fallback (with hooks); both must expose the same raw designs.
    task = HyperbolicTemporalDiscounting()
    designer = StaticDesigner(task, T=3)
    theta = task.sample_prior(4)

    vectorized = designer.rollout(theta)

    designer.hooks.append(RolloutHook())
    per_step = designer.rollout(theta)

    assert torch.allclose(
        vectorized.extra["raw_design_history"], per_step.extra["raw_design_history"]
    )
    assert torch.allclose(vectorized.design_history, per_step.design_history)


def test_default_transform_designs_matches_task_design_space():
    task = HyperbolicTemporalDiscounting()
    designer = StaticDesigner(task, T=3)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    raw_design_history = rollout.extra["raw_design_history"]
    assert torch.allclose(
        task.design_space.to_design(raw_design_history), rollout.design_history
    )
    # HTDDesignSpace.to_design is a real (non-identity) transform, so raw and
    # design must actually differ here.
    assert not torch.allclose(raw_design_history, rollout.design_history)


def test_transform_designs_false_opts_out():
    task = HyperbolicTemporalDiscounting()
    # With transform_designs=False, the raw learnable parameter (~N(0,1)) is
    # used directly as the design, which won't satisfy HTDDesignSpace's real
    # domain constraints (D1 > 0, 0 < R0 < 100) -- disable validation since
    # this test is only about the designer-level opt-out, not design validity.
    task.validate_designs = False
    designer = StaticDesigner(task, T=3, transform_designs=False)
    theta = task.sample_prior(4)

    rollout = designer.rollout(theta)

    # No transform applied: raw and constrained designs coincide exactly.
    assert torch.equal(rollout.extra["raw_design_history"], rollout.design_history)


def test_transform_designs_false_agrees_between_vectorized_and_per_step_paths():
    task = HyperbolicTemporalDiscounting()
    task.validate_designs = False
    designer = StaticDesigner(task, T=3, transform_designs=False)
    theta = task.sample_prior(4)

    vectorized = designer.rollout(theta)

    designer.hooks.append(RolloutHook())
    per_step = designer.rollout(theta)

    assert torch.allclose(
        vectorized.extra["raw_design_history"], per_step.extra["raw_design_history"]
    )
    assert torch.allclose(vectorized.design_history, per_step.design_history)
    assert torch.equal(
        vectorized.design_history, vectorized.extra["raw_design_history"]
    )


class _SumActionNet(nn.Module):
    """Minimal action net: a linear map of the summed outcome history."""

    def __init__(self, dim_y=1, dim_action=2):
        super().__init__()
        self.linear = nn.Linear(dim_y, dim_action)

    def forward(self, y_history, design_history):
        return self.linear(y_history.sum(dim=1))


# The moves below are exercised with a dtype cast rather than a device move:
# `.to(torch.float64)` goes through the same nn.Module._apply machinery as
# `.to(device)`, but runs without a GPU. The CUDA move is checked separately.
def test_to_moves_task_hooks_and_own_parameters():
    task = LocationFinding(n_sources=1, dim_theta=2)
    action_net = _SumActionNet()
    designer = StaticDesigner(task, T=3, hooks=[ActionHook(action_net)])

    designer.to(torch.float64)

    assert designer.raw_design.dtype == torch.float64
    assert task.prior.loc.dtype == torch.float64
    assert action_net.linear.weight.dtype == torch.float64


def test_hook_appended_after_construction_is_registered():
    task = LocationFinding(n_sources=1, dim_theta=2)
    designer = StaticDesigner(task, T=3)
    action_net = _SumActionNet()

    designer.hooks.append(ActionHook(action_net))
    designer.to(torch.float64)

    assert action_net in set(designer.modules())
    assert action_net.linear.weight.dtype == torch.float64


def test_train_eval_reaches_hook_modules():
    task = LocationFinding(n_sources=1, dim_theta=2)
    action_net = _SumActionNet()
    designer = StaticDesigner(task, T=3, hooks=[ActionHook(action_net)])

    designer.eval()
    assert not action_net.training

    designer.train()
    assert action_net.training


def test_hook_that_skips_module_init_raises():
    class _UninitializedHook(RolloutHook):
        def __init__(self):
            self.value = 1  # no super().__init__()

    task = LocationFinding(n_sources=1, dim_theta=2)
    with pytest.raises(TypeError, match="super"):
        StaticDesigner(task, T=3, hooks=[_UninitializedHook()])


def test_non_module_hook_raises():
    class _NotAHook:
        def after_step(self, state):
            pass

    task = LocationFinding(n_sources=1, dim_theta=2)
    with pytest.raises(TypeError, match="RolloutHook"):
        StaticDesigner(task, T=3, hooks=[_NotAHook()])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_cuda_moves_the_whole_rollout():
    task = LocationFinding(n_sources=1, dim_theta=2)
    action_net = _SumActionNet()
    designer = StaticDesigner(task, T=3, hooks=[ActionHook(action_net)])

    designer.to("cuda")

    assert designer.device.type == "cuda"
    assert task.device.type == "cuda"
    assert action_net.linear.weight.device.type == "cuda"

    rollout = designer.rollout(task.sample_prior(2))
    assert rollout.design_history.device.type == "cuda"
    assert rollout.extra["action"].device.type == "cuda"
