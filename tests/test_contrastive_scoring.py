"""Tests that the contrastive scoring paths agree with the per-step loop.

Two changes are under test here, and they carry different guarantees:

- Broadcasting the histories over the contrastive axis, rather than
  replicating them to ``[b * L, T, ...]`` up front, feeds ``_outcome_logp``
  the same flat tensor it saw before, element for element. That is required
  to be **bitwise** identical.
- Scoring the steps in one fused call sums over T in a different order than
  the loop does, so it is required only to agree to float roundoff.
"""

import pytest
import torch

from testbed.core.rollout import Rollout
from testbed.design_objectives.contrastive import contrastive_logsumexps
from testbed.tasks import CES, LocationFinding
from testbed.tasks.hyperbolic_temporal_discounting import HyperbolicTemporalDiscounting

MEMORYLESS_TASKS = {
    "location_finding": lambda: LocationFinding(n_sources=2, dim_theta=2),
    "ces": CES,
    "htd": HyperbolicTemporalDiscounting,
}


def _rollout(task, b, T):
    """A rollout of T random designs, scored under b prior draws."""
    theta = task.sample_prior(b)
    design = task.design_space.sample(batch_shape=(b, T))
    y = task.sample_outcome(theta.unsqueeze(1), design)
    return Rollout(theta=theta, y_history=y, design_history=design)


def _replicated_reference(task, rollout, L):
    """The pre-broadcast scoring: histories copied L times, then one flat call."""
    b = rollout.theta.shape[0]

    def replicate(x):
        return x.unsqueeze(1).expand(b, L, *x.shape[1:]).reshape(b * L, *x.shape[1:])

    torch.manual_seed(0)
    theta = task.sample_current_prior(b * L).reshape(b * L, *task.event_shape_theta)
    return task.history_logp(
        replicate(rollout.y_history), replicate(rollout.design_history), theta
    ).reshape(b, L)


@pytest.mark.parametrize("name", sorted(MEMORYLESS_TASKS))
def test_broadcasting_the_histories_is_bitwise_identical(name):
    # The flat tensor reaching _outcome_logp is unchanged, so the arithmetic is
    # too: anything looser would be a silent change to every reported bound.
    task = MEMORYLESS_TASKS[name]()
    b, T, L = 3, 4, 8
    torch.manual_seed(1)
    rollout = _rollout(task, b, T)

    expected = _replicated_reference(task, rollout, L)
    torch.manual_seed(0)
    _, chunks = contrastive_logsumexps(task, rollout, L, chunk_size=L)

    assert torch.equal(chunks.squeeze(1), torch.logsumexp(expected, dim=1))


@pytest.mark.parametrize("name", sorted(MEMORYLESS_TASKS))
def test_fused_scoring_matches_the_per_step_loop(name):
    task = MEMORYLESS_TASKS[name]()
    b, T, L = 3, 4, 16
    torch.manual_seed(1)
    rollout = _rollout(task, b, T)

    torch.manual_seed(0)
    _, looped = contrastive_logsumexps(task, rollout, L, chunk_size=L)
    torch.manual_seed(0)
    _, fused = contrastive_logsumexps(task, rollout, L, chunk_size=L, fused=True)

    assert torch.allclose(looped, fused, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", sorted(MEMORYLESS_TASKS))
def test_fused_scoring_matches_the_loop_on_gradients(name):
    # Training differentiates through this, so matching values is not enough:
    # the gradient reaching the policy has to match too.
    task = MEMORYLESS_TASKS[name]()
    b, T, L = 4, 5, 32
    torch.manual_seed(1)
    base = _rollout(task, b, T)

    def grads(fused):
        design = base.design_history.clone().requires_grad_(True)
        y = base.y_history.clone().requires_grad_(True)
        rollout = Rollout(theta=base.theta, y_history=y, design_history=design)
        torch.manual_seed(0)
        joint, chunks = contrastive_logsumexps(task, rollout, L, None, fused=fused)
        (joint - torch.logsumexp(chunks, dim=1)).mean().backward()
        # HTD's outcomes are discrete, so y carries no gradient there.
        return design.grad, y.grad

    looped, fused = grads(False), grads(True)
    for a, c in zip(looped, fused):
        assert (a is None) == (c is None)
        if a is not None:
            assert torch.allclose(a, c, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("chunk_size", [1, 5, None])
def test_fused_scoring_still_chunks_exactly(chunk_size):
    # Chunk boundaries must only change how much is scored at once, not the
    # reduction. Pin the draws so the comparison isolates that.
    task = LocationFinding(n_sources=2, dim_theta=2)
    b, L = 3, 12
    torch.manual_seed(1)
    rollout = _rollout(task, b, T=4)
    pool = task.sample_prior(b * L).reshape(b, L, *task.event_shape_theta)
    cursor = 0

    def pinned(n_samples=1):
        nonlocal cursor
        k = n_samples // b
        out = pool[:, cursor : cursor + k].reshape(b * k, *task.event_shape_theta)
        cursor += k
        return out

    task.sample_current_prior = pinned

    _, chunks = contrastive_logsumexps(task, rollout, L, chunk_size, fused=True)
    cursor = 0
    _, whole = contrastive_logsumexps(task, rollout, L, None, fused=True)

    assert torch.allclose(
        torch.logsumexp(chunks, dim=1), torch.logsumexp(whole, dim=1), atol=1e-5
    )


def test_fused_is_ignored_for_a_task_with_dependent_steps():
    # eval sweeps every task, so fused=True must degrade rather than raise.
    from testbed.tasks import StochasticPendulum

    task = StochasticPendulum()
    assert not task.memoryless
    torch.manual_seed(1)
    rollout = _rollout(task, b=2, T=3)

    torch.manual_seed(0)
    _, plain = contrastive_logsumexps(task, rollout, 6, chunk_size=6)
    torch.manual_seed(0)
    _, asked = contrastive_logsumexps(task, rollout, 6, chunk_size=6, fused=True)

    assert torch.equal(plain, asked)


def test_history_logp_rejects_fusing_dependent_steps():
    # The Task-level entry point is explicit, so there it is an error.
    from testbed.tasks import StochasticPendulum

    task = StochasticPendulum()
    torch.manual_seed(1)
    rollout = _rollout(task, b=2, T=3)

    with pytest.raises(ValueError, match="not memoryless"):
        task.history_logp(
            rollout.y_history, rollout.design_history, rollout.theta, fused=True
        )
