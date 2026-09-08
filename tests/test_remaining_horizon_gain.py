"""Tests for the three conditioning channels in remaining_horizon_gain.

The observed history has to reach the theta draws, the design net and the
objective. Dropping any one of them still runs and still produces a plausible
number, so each channel gets a test that fails if it is dropped.
"""

import torch

from testbed.core.distributions import EmpiricalDistribution
from testbed.design_nets import DeepSetPolicyNet
from testbed.design_objectives import SPCE
from testbed.designers import SemiAmortizedDesigner
from testbed.tasks import LocationFinding


def _make_task():
    return LocationFinding(n_sources=1, dim_theta=2)


def _make_designer(task, T=4):
    return SemiAmortizedDesigner(
        task,
        T=T,
        design_net=DeepSetPolicyNet(
            event_shape_y=task.event_shape_y,
            event_shape_design=task.event_shape_design,
            event_shape_out=task.event_shape_design,
        ),
    )


def test_theta_come_from_current_prior_not_the_prior():
    # Channel 1. A posterior of identical particles is distinguishable: every
    # theta drawn must equal that value, which the prior would essentially never
    # give. Two copies rather than one because EmpiricalDistribution squeezes its
    # weights, so a single particle leaves multinomial a 0-dim tensor.
    torch.manual_seed(0)
    task = _make_task()
    designer = _make_designer(task)
    observed = designer.rollout(task.sample_prior(1))

    particle = torch.full((1, *task.event_shape_theta), 3.0)
    task.update_prior(EmpiricalDistribution(locs=particle.expand(2, -1, -1)))

    seen = []

    class _RecordingSPCE(SPCE):
        def estimate_gain(self, rollout, y_context=None, design_context=None):
            seen.append(rollout.theta)
            return super().estimate_gain(
                rollout, y_context=y_context, design_context=design_context
            )

    designer.remaining_horizon_gain(
        _RecordingSPCE(task, n_contrastive=4),
        observed.y_history,
        observed.design_history,
        n_steps=2,
        batch_size=8,
    )

    torch.testing.assert_close(seen[0], particle.expand(8, *task.event_shape_theta))


def test_the_design_net_conditions_on_the_observed_history():
    # Channel 2. Sum-pooled encodings mean a non-empty history genuinely moves
    # the proposal, so a dropped history changes the gain rather than nothing.
    torch.manual_seed(0)
    task = _make_task()
    designer = _make_designer(task)
    observed = designer.rollout(task.sample_prior(1))

    seen = []

    class _SpyNet(type(designer.design_net)):
        def propose(self, y_history, design_history, **kwargs):
            seen.append(y_history.shape[1])
            return super().propose(y_history, design_history, **kwargs)

    designer.design_net.__class__ = _SpyNet

    designer.remaining_horizon_gain(
        SPCE(task, n_contrastive=4),
        observed.y_history,
        observed.design_history,
        n_steps=2,
        batch_size=8,
    )

    S = observed.T
    assert seen == [S, S + 1]


def test_the_objective_conditions_on_the_history_without_scoring_it():
    # Channel 3. The context must arrive at the objective, and arrive as
    # context: scoring the history too would add its own log-prob on top.
    torch.manual_seed(0)
    task = _make_task()
    designer = _make_designer(task)
    observed = designer.rollout(task.sample_prior(1))

    seen = {}

    class _RecordingSPCE(SPCE):
        def estimate_gain(self, rollout, y_context=None, design_context=None):
            seen["y_context"] = y_context
            seen["design_context"] = design_context
            seen["scored_steps"] = rollout.T
            return super().estimate_gain(
                rollout, y_context=y_context, design_context=design_context
            )

    designer.remaining_horizon_gain(
        _RecordingSPCE(task, n_contrastive=4),
        observed.y_history,
        observed.design_history,
        n_steps=2,
        batch_size=8,
    )

    # The same history that seeded the continuation, handed over as context.
    assert seen["y_context"] is observed.y_history
    assert seen["design_context"] is observed.design_history
    # And only the simulated steps are in the rollout being scored.
    assert seen["scored_steps"] == 2


def test_context_is_conditioned_on_but_not_scored():
    # The algebraic identity behind channel 3, at the task level: scoring the
    # history and the continuation together is exactly the history's own
    # log-prob more than scoring the continuation in its context.
    torch.manual_seed(0)
    task = _make_task()
    designer = _make_designer(task)
    observed = designer.rollout(task.sample_prior(1))
    theta = task.sample_prior(5)

    rollout = designer.continue_from(
        theta, observed.y_history, observed.design_history, n_steps=2
    )
    y_context = observed.y_history.expand(5, -1, -1)
    design_context = observed.design_history.expand(5, -1, -1)

    with_context = task.history_logp(
        rollout.y_history,
        rollout.design_history,
        theta,
        y_context=y_context,
        design_context=design_context,
    )
    context_only = task.history_logp(y_context, design_context, theta)
    scored_together = task.history_logp(
        torch.cat([y_context, rollout.y_history], dim=1),
        torch.cat([design_context, rollout.design_history], dim=1),
        theta,
    )

    torch.testing.assert_close(scored_together, with_context + context_only)
