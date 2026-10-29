"""Tests that the contrastive scoring paths agree with the per-step loop.

Three changes are under test here, and they carry different guarantees:

- Broadcasting the histories over the contrastive axis, rather than
  replicating them to ``[b * L, T, ...]`` up front, feeds ``_outcome_logp``
  the same flat tensor it saw before, element for element. That is required
  to be **bitwise** identical.
- Scoring a memoryless task's steps in one fused call sums over T in a
  different order than the loop does, so it is required only to agree to
  float roundoff.
- Scoring a history-dependent task with fused=True compiles the loop itself,
  whose fused kernels are likewise required only to agree to float roundoff.
"""

import pytest
import torch

from testbed.core.rollout import Rollout
from testbed.design_objectives.contrastive import contrastive_logsumexps
from testbed.tasks import CES, LocationFinding, StochasticPendulum
from testbed.tasks.temporal_discounting import TemporalDiscounting

MEMORYLESS_TASKS = {
    "location_finding": lambda: LocationFinding(n_sources=2, dim_theta=2),
    "ces": CES,
    "td": TemporalDiscounting,
}

# The compiled loop is task-agnostic and compiling it dominates test time, so
# one cheap history-dependent task is enough.
DEPENDENT_TASKS = {"stochastic_pendulum": StochasticPendulum}

ALL_TASKS = {**MEMORYLESS_TASKS, **DEPENDENT_TASKS}


def _rollout(task, b, T):
    """A rollout of T random designs, scored under b prior draws."""
    theta = task.sample_prior(b)
    design = task.design_prior.sample((b, T))
    y = task.sample_outcome(theta.unsqueeze(1), design)
    return Rollout(theta=theta, y_history=y, design_history=design)


def _replicated_reference(task, rollout, L):
    """The pre-broadcast scoring: histories copied L times, then one flat call."""
    b = rollout.theta.shape[0]

    def replicate(x):
        return x.unsqueeze(1).expand(b, L, *x.shape[1:]).reshape(b * L, *x.shape[1:])

    torch.manual_seed(0)
    theta = task.sample_current_prior(b * L)
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


@pytest.mark.parametrize("name", sorted(ALL_TASKS))
def test_fused_scoring_matches_the_per_step_loop(name):
    task = ALL_TASKS[name]()
    b, T, L = 3, 4, 16
    torch.manual_seed(1)
    rollout = _rollout(task, b, T)

    task.max_folded_steps = 0  # a batch this small would otherwise skip the loop
    torch.manual_seed(0)
    _, looped = contrastive_logsumexps(task, rollout, L, chunk_size=L)
    torch.manual_seed(0)
    _, fused = contrastive_logsumexps(task, rollout, L, chunk_size=L, fused=True)

    assert torch.allclose(looped, fused, rtol=1e-5, atol=1e-5)


@pytest.fixture
def float64():
    """Runs a test in float64, restoring float32 after."""
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("name", sorted(ALL_TASKS))
def test_fused_scoring_matches_the_loop_on_gradients(name, float64):
    # Training differentiates through this, so matching values is not enough:
    # the gradient reaching the policy has to match too. In float64, since the
    # dynamical tasks' gradients reach O(1e2), where fp32 roundoff alone
    # exceeds any useful atol.
    task = ALL_TASKS[name]()
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
        # TD's outcomes are discrete, so y carries no gradient there.
        return design.grad, y.grad

    task.max_folded_steps = 0  # a batch this small would otherwise skip the loop
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
    pool = task.sample_prior((b, L))
    cursor = 0

    def pinned(sample_shape=()):
        nonlocal cursor
        _, k = sample_shape
        out = pool[:, cursor : cursor + k]
        cursor += k
        return out

    task.sample_current_prior = pinned

    _, chunks = contrastive_logsumexps(task, rollout, L, chunk_size, fused=True)
    cursor = 0
    _, whole = contrastive_logsumexps(task, rollout, L, None, fused=True)

    assert torch.allclose(
        torch.logsumexp(chunks, dim=1), torch.logsumexp(whole, dim=1), atol=1e-5
    )


@pytest.mark.parametrize("name", sorted(DEPENDENT_TASKS))
def test_fused_history_logp_still_conditions_on_the_context(name):
    # Unlike the memoryless fold, the compiled loop must not drop the context.
    task = DEPENDENT_TASKS[name]()
    torch.manual_seed(1)
    rollout = _rollout(task, b=3, T=4)
    context = _rollout(task, b=3, T=2)
    args = (rollout.y_history, rollout.design_history, rollout.theta)
    ctx = {"y_context": context.y_history, "design_context": context.design_history}

    looped = task.history_logp(*args, **ctx)
    fused = task.history_logp(*args, **ctx, fused=True)

    assert torch.allclose(looped, fused, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(fused, task.history_logp(*args, fused=True))
