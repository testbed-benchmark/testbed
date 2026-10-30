"""Tests for RandomDesigner's two design sources: a given dist, or the design space."""

import pytest
import torch
from torch.distributions import Normal, constraints
from torch.distributions.transforms import Transform

from testbed.core.distributions import TorchDistribution
from testbed.core.hooks import RolloutHook
from testbed.designers import RandomDesigner
from testbed.tasks import CES, LocationFinding, PreyPopulation


def make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def make_dist():
    return TorchDistribution(
        torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
        )
    )


# ---------------------------------------------------------------------------
# Design-space default (dist=None)
# ---------------------------------------------------------------------------


def test_defaults_to_the_design_space_and_produces_valid_designs():
    torch.manual_seed(0)
    task = CES()
    designer = RandomDesigner(task, T=3)

    rollout = designer.rollout(task.sample_prior(4))

    assert rollout.design_history.shape == (4, 3, *task.event_shape_design)
    assert task.contains_design(rollout.design_history)


def test_design_space_default_works_for_a_discrete_space():
    # RandomDesigner is the one designer that works with a finite space: it
    # samples the space directly instead of going through to_design.
    torch.manual_seed(0)
    task = PreyPopulation()
    designer = RandomDesigner(task, T=3)

    rollout = designer.rollout(task.sample_prior(4))

    assert task.contains_design(rollout.design_history)


def test_design_space_default_reports_no_policy_log_probs():
    # The design space exposes no density, so the key must be absent rather
    # than present-and-None.
    torch.manual_seed(0)
    task = make_task()
    designer = RandomDesigner(task, T=3)

    rollout = designer.rollout(task.sample_prior(4))

    assert "policy_log_probs" not in rollout.extra

    design, log_prob = designer.sample_design(
        torch.zeros(4, 0, *task.event_shape_y), torch.zeros(4, 0, 2)
    )
    assert design.shape == (4, *task.event_shape_design)
    assert log_prob is None


def test_random_designer_reports_raw_design_history():
    # Designs are drawn physically, so the raw values a policy would have
    # proposed are recovered by inverting the design space. Anything
    # conditioning on raw history then sees the same coordinates during a random
    # warmup as during a policy phase.
    torch.manual_seed(0)
    task = CES()

    # Both the vectorized fast path and the per-step path a hook forces.
    for hooks in ((), (RolloutHook(),)):
        rollout = RandomDesigner(task, T=3, hooks=hooks).rollout(task.sample_prior(4))

        raw = rollout.extra["raw_design_history"]
        assert raw.shape == rollout.design_history.shape
        torch.testing.assert_close(task.design_transform(raw), rollout.design_history)


def test_random_designer_reports_no_raw_design_history_without_an_inverse():
    # Every built-in design transform is invertible, so this is the contract
    # for a custom one that is not: the key stays absent and callers fall back
    # to physical designs, rather than the rollout failing.
    class _NoInverse(Transform):
        domain = codomain = constraints.independent(constraints.real, 1)
        bijective = True

        def _call(self, x):
            return x

    # a design prior sets its parametrisation by defining a transform
    design_prior = TorchDistribution(Normal(torch.zeros(2), 1.0))
    design_prior.transform = _NoInverse()

    torch.manual_seed(0)
    task = LocationFinding(n_sources=1, dim_theta=2, design_prior=design_prior)

    rollout = RandomDesigner(task, T=3).rollout(task.sample_prior(4))

    assert "raw_design_history" not in rollout.extra


def test_device_falls_back_to_the_task_without_a_dist():
    task = make_task()

    assert RandomDesigner(task, T=2).device == task.device


# ---------------------------------------------------------------------------
# Supplied distribution
# ---------------------------------------------------------------------------


def test_supplied_dist_is_used_and_carries_log_probs():
    torch.manual_seed(0)
    task = make_task()
    designer = RandomDesigner(task, T=3, dist=make_dist())

    rollout = designer.rollout(task.sample_prior(4))

    assert rollout.design_history.shape == (4, 3, 2)
    assert rollout.extra["policy_log_probs"].shape == (4, 3)
    assert torch.isfinite(rollout.extra["policy_log_probs"]).all()


# ---------------------------------------------------------------------------
# The hooked and vectorized paths must agree
# ---------------------------------------------------------------------------


def _shapes(rollout):
    return rollout.design_history.shape, rollout.y_history.shape


def test_hooked_and_vectorized_paths_agree_on_shapes_and_extras():
    # rollout() has a vectorized fast path taken only when no hooks are
    # attached; both must draw from the same source and report the same keys.
    from testbed.core.hooks import RolloutHook

    task = make_task()
    theta = task.sample_prior(4)

    for dist, expect_log_probs in ((None, False), (make_dist(), True)):
        torch.manual_seed(0)
        fast = RandomDesigner(task, T=3, dist=dist).rollout(theta)
        torch.manual_seed(0)
        hooked = RandomDesigner(task, T=3, dist=dist, hooks=[RolloutHook()]).rollout(
            theta
        )

        assert _shapes(fast) == _shapes(hooked)
        assert ("policy_log_probs" in fast.extra) is expect_log_probs
        assert ("policy_log_probs" in hooked.extra) is expect_log_probs


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def test_wrapped_dist_reports_the_device_it_was_built_on():
    # A wrapper that was never moved must keep reporting where its parameters
    # are, so wrapping a dist built on an accelerator is not a silent downgrade.
    dist = TorchDistribution(
        torch.distributions.Normal(
            torch.zeros(2, device="meta"),
            torch.ones(2, device="meta"),
            validate_args=False,  # validation calls .item(), which meta forbids
        )
    )

    assert dist.device.type == "meta"


def test_wrapped_composite_dist_reports_its_nested_parameters_device():
    # Composite distributions hold no tensors directly, so the device has to be
    # found by searching nested distributions rather than defaulting to cpu.
    loc = torch.zeros(4, 3, 2, device="meta")
    component = torch.distributions.Independent(
        torch.distributions.Normal(loc, torch.ones_like(loc), validate_args=False), 1
    )
    mixture = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(
            logits=torch.zeros(4, 3, device="meta"), validate_args=False
        ),
        component,
        validate_args=False,
    )

    assert TorchDistribution(component).device.type == "meta"
    assert TorchDistribution(mixture).device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_to_cuda_moves_designs_drawn_from_a_wrapped_dist():
    # The wrapped torch distribution's own parameters stay on the CPU -- only
    # what crosses the wrapper's interface is moved.
    task = make_task()
    designer = RandomDesigner(task, T=3, dist=make_dist())

    designer.to("cuda")

    assert designer.device.type == "cuda"
    rollout = designer.rollout(task.sample_prior(4))
    assert rollout.design_history.device.type == "cuda"
    assert rollout.extra["policy_log_probs"].device.type == "cuda"
