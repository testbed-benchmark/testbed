"""Tests for the Task API's history contract: null_history is well formed,
_outcome_logp/_sample_outcome receive y_history/design_history, history_logp
forwards the running history without requiring an override, and
Designer/designer fast paths respect the memoryless flag.
"""

import pytest
import torch
from torch.distributions import Independent, Normal

from testbed.core.distributions import TorchDistribution
from testbed.designers import RandomDesigner, StaticDesigner
from testbed.tasks import LocationFinding
from testbed.tasks.base import Task


def _scalar_prior():
    return TorchDistribution(Independent(Normal(torch.zeros(1), torch.ones(1)), 1))


class _AutoregressiveTask(Task):
    """y_t ~ N(theta + y_{t-1}, 1), y_{-1} := 0. Minimal history-dependent task."""

    @property
    def reparameterizable_outcomes(self) -> bool:
        return True

    def __init__(self):
        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[1],
            event_shape_design=[1],
            prior=_scalar_prior(),
            memoryless=False,
        )

    def _loc(self, theta, y_history):
        if y_history is None or y_history.shape[1] == 0:
            return theta
        return theta + y_history[:, -1]

    def _outcome_logp(self, y, theta, design, y_history=None, **kwargs):
        del design
        return Normal(self._loc(theta, y_history), 1.0).log_prob(y).sum(dim=-1)

    def _sample_outcome(self, theta, design, y_history=None, **kwargs):
        del design
        return self._loc(theta, y_history) + torch.randn_like(theta)


class _HistorySpyTask(Task):
    """Records the y_history/design_history it's called with."""

    @property
    def reparameterizable_outcomes(self) -> bool:
        return True

    def __init__(self):
        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[1],
            event_shape_design=[1],
            prior=_scalar_prior(),
            memoryless=False,
        )
        self.logp_calls = []
        self.sample_calls = []

    def _outcome_logp(
        self, y, theta, design, y_history=None, design_history=None, **kwargs
    ):
        self.logp_calls.append((y_history.clone(), design_history.clone()))
        return Normal(theta.squeeze(-1), 1.0).log_prob(y.squeeze(-1))

    def _sample_outcome(
        self, theta, design, y_history=None, design_history=None, **kwargs
    ):
        self.sample_calls.append((y_history.clone(), design_history.clone()))
        return theta + torch.randn_like(theta)


class _CountingTask(Task):
    """Records how many times _sample_outcome is invoked."""

    @property
    def reparameterizable_outcomes(self) -> bool:
        # Irrelevant to what's under test; keep the gradient-gating contract
        # in base.py out of the way of counting calls.
        return False

    def __init__(self, memoryless):
        super().__init__(
            event_shape_y=[1],
            event_shape_theta=[1],
            event_shape_design=[1],
            prior=_scalar_prior(),
            memoryless=memoryless,
        )
        self.call_count = 0

    def _outcome_logp(self, y, theta, design, **kwargs):
        return Normal(theta.squeeze(-1), 1.0).log_prob(y.squeeze(-1))

    def _sample_outcome(self, theta, design, **kwargs):
        self.call_count += 1
        return theta + torch.randn_like(theta)


class _MatrixDesignTask(Task):
    """Multi-dim y/design event shapes, which a flattened null history loses."""

    @property
    def reparameterizable_outcomes(self) -> bool:
        return True

    def __init__(self):
        super().__init__(
            event_shape_y=[2],
            event_shape_theta=[1],
            event_shape_design=[2, 3],
            prior=_scalar_prior(),
        )

    def _outcome_logp(self, y, theta, design, y_history=None, design_history=None):
        return Normal(theta, 1.0).log_prob(y).sum(dim=-1)

    def _sample_outcome(self, theta, design, y_history=None, design_history=None):
        return theta + torch.randn(theta.shape[0], 2)


@pytest.mark.parametrize("b", [None, 0, 1, 5])
def test_null_history_is_empty_and_carries_the_task_event_shapes(b):
    task = LocationFinding(n_sources=1, dim_theta=2)
    batch = () if b is None else (b,)

    y_history, design_history = task.null_history(b=b)

    assert y_history.shape == (*batch, 0, *task.event_shape_y)
    assert design_history.shape == (*batch, 0, *task.event_shape_design)
    assert y_history.numel() == 0
    assert design_history.numel() == 0
    # T is the axis just before the event dims, which is how base.py reads it.
    assert y_history.shape[-(len(task.event_shape_y) + 1)] == 0
    assert design_history.shape[-(len(task.event_shape_design) + 1)] == 0


def test_null_history_defaults_to_no_batch_dim():
    task = LocationFinding(n_sources=1, dim_theta=2)

    y_history, design_history = task.null_history()

    assert y_history.shape == (0, *task.event_shape_y)
    assert design_history.shape == (0, *task.event_shape_design)


@pytest.mark.parametrize("b", [None, 4])
def test_null_history_preserves_multi_dim_event_shapes(b):
    task = _MatrixDesignTask()
    batch = () if b is None else (b,)

    y_history, design_history = task.null_history(b=b)

    assert y_history.shape == (*batch, 0, 2)
    assert design_history.shape == (*batch, 0, 2, 3)


def test_null_history_is_on_the_task_device():
    task = LocationFinding(n_sources=1, dim_theta=2)

    assert all(h.device == task.device for h in task.null_history(b=2))


@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)])
def test_default_null_history_broadcasts_against_any_batch(batch_shape):
    # The no-batch-dim default is what lets a caller with no history take the
    # same path as one with a real history. Any stray leading dim here -- a
    # size-0 batch especially -- fails to broadcast against a real batch.
    task = LocationFinding(n_sources=1, dim_theta=2)
    theta = task.sample_prior().expand(*batch_shape, *task.event_shape_theta)
    design = task.design_prior.sample(batch_shape)

    y = task.sample_outcome(theta, design)
    logp = task.outcome_logp(y, theta, design)

    assert y.shape == (*batch_shape, *task.event_shape_y)
    assert logp.shape == batch_shape


def test_memoryless_task_accepts_and_ignores_history_kwargs():
    # LocationFinding's _outcome_logp/_sample_outcome only declare **kwargs for
    # interface parity; passing history it doesn't need must not raise.
    task = LocationFinding(n_sources=1, dim_theta=2)
    theta = task.sample_prior(4)
    design = torch.randn(4, 2)
    y = task.sample_outcome(theta, design)

    logp = task.outcome_logp(
        y,
        theta,
        design,
        y_history=torch.zeros(4, 0, 1),
        design_history=torch.zeros(4, 0, 2),
    )
    assert torch.isfinite(logp).all()


def test_non_memoryless_task_does_not_require_history_logp_override():
    # Previously, memoryless=False (nee separable=False) raised TypeError at
    # construction unless history_logp was overridden.
    task = _AutoregressiveTask()
    assert type(task).history_logp is Task.history_logp


def test_history_logp_default_passes_strictly_prior_history_at_each_step():
    task = _HistorySpyTask()
    b, T = 2, 3
    y_history = torch.randn(b, T, 1)
    design_history = torch.randn(b, T, 1)
    theta = torch.randn(b, 1)

    task.history_logp(y_history, design_history, theta)

    assert len(task.logp_calls) == T
    for t, (seen_y, seen_design) in enumerate(task.logp_calls):
        assert torch.equal(seen_y, y_history[:, :t])
        assert torch.equal(seen_design, design_history[:, :t])


def _location_finding_history(b=3, T=4):
    torch.manual_seed(0)
    task = LocationFinding(n_sources=2, dim_theta=2)
    theta = task.sample_prior(b)
    design = task.design_prior.sample((b, T))
    y_history = task.sample_outcome(theta.unsqueeze(1), design)
    return task, y_history, design, theta


def test_memoryless_history_logp_folds_small_batches_and_loops_large_ones(monkeypatch):
    task, y_history, design, theta = _location_finding_history(b=3, T=4)
    calls = []
    for name in ("_memoryless_history_logp", "_sequential_history_logp"):
        method = getattr(task, name)
        monkeypatch.setattr(
            task, name, lambda *args, m=method, n=name: calls.append(n) or m(*args)
        )

    task.max_folded_steps = 12  # exactly b * T
    task.history_logp(y_history, design, theta)
    task.max_folded_steps = 11
    task.history_logp(y_history, design, theta)

    assert calls == ["_memoryless_history_logp", "_sequential_history_logp"]


def test_folded_and_looped_history_logp_agree_on_values_and_gradients():
    task, y_history, design, theta = _location_finding_history()

    def score(max_folded_steps):
        task.max_folded_steps = max_folded_steps
        leaves = theta.clone().requires_grad_(True), design.clone().requires_grad_(True)
        logp = task.history_logp(y_history, leaves[1], leaves[0])
        logp.sum().backward()
        return logp, *(leaf.grad for leaf in leaves)

    for folded, looped in zip(score(Task.max_folded_steps), score(0)):
        torch.testing.assert_close(folded, looped)


def test_history_logp_correct_for_history_dependent_likelihood():
    task = _AutoregressiveTask()
    b, T = 3, 4
    theta = torch.randn(b, 1)
    design = torch.zeros(b, T, 1)
    y_history = torch.randn(b, T, 1)

    logp = task.history_logp(y_history, design, theta)

    manual = torch.zeros(b)
    prev = torch.zeros(b, 1)
    for t in range(T):
        manual += Normal(theta + prev, 1.0).log_prob(y_history[:, t]).sum(dim=-1)
        prev = y_history[:, t]

    assert torch.allclose(logp, manual, atol=1e-5)


def test_designer_rollout_passes_accumulated_history_to_sample_outcome():
    task = _HistorySpyTask()
    designer = RandomDesigner(task, T=3)

    designer.rollout(torch.randn(2, 1))

    assert len(task.sample_calls) == 3
    for t, (seen_y, seen_design) in enumerate(task.sample_calls):
        assert seen_y.shape == (2, t, 1)
        assert seen_design.shape == (2, t, 1)


def test_static_designer_takes_vectorized_path_for_memoryless_task():
    task = _CountingTask(memoryless=True)
    StaticDesigner(task, T=4).rollout(torch.randn(3, 1))

    # One call across all T steps at once, not one per step.
    assert task.call_count == 1


def test_static_designer_takes_per_step_path_for_non_memoryless_task():
    task = _CountingTask(memoryless=False)
    StaticDesigner(task, T=4).rollout(torch.randn(3, 1))

    assert task.call_count == 4


def test_random_designer_takes_vectorized_path_for_memoryless_task():
    task = _CountingTask(memoryless=True)
    RandomDesigner(task, T=4).rollout(torch.randn(3, 1))

    assert task.call_count == 1


def test_random_designer_takes_per_step_path_for_non_memoryless_task():
    task = _CountingTask(memoryless=False)
    RandomDesigner(task, T=4).rollout(torch.randn(3, 1))

    assert task.call_count == 4
