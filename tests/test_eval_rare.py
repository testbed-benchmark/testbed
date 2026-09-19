"""Tests for the rare-tail sPCE/sNMC evaluation.

The rare bounds differ from the headline ones in two ways that are easy to get
silently wrong: the outer theta must come from the fixed tail rather than the
prior, and the two bounds must contrast against *different* measures. Both are
checked here, along with the threshold routes that define the tail.
"""

import pytest
import torch

from testbed.designers import RandomDesigner
from testbed.eval.base import chunked_spce_snmc
from testbed.eval.eval_rare import (
    rare_eval_kwargs,
    rare_metrics,
    rare_settings,
    rare_spce_snmc,
    rare_tail_prior,
)
from testbed.tasks import LocationFinding

# A shallow tail keeps rejection sampling cheap while still being a tail, and
# the pilot only has to be big enough to place the threshold roughly.
PERCENTILE = 25
N_PILOT = 20_000
N_SAMPLES = 1_000


@pytest.fixture
def task():
    """A small location-finding task, used for its prior."""
    torch.manual_seed(0)
    return LocationFinding(n_sources=2, dim_theta=2, dim_y=1)


@pytest.fixture
def tail(task):
    """The bottom PERCENTILE% of that task's prior by log-density."""
    return rare_tail_prior(task.prior, percentile=PERCENTILE, n_pilot=N_PILOT)


def test_samples_fall_below_the_threshold(task, tail):
    """Every draw is in the tail, and there are exactly as many as asked for."""
    theta = tail.sample(n_samples=N_SAMPLES)
    assert theta.shape == (N_SAMPLES, 2, 2)
    assert (task.prior.log_prob(theta) < tail.threshold).all()


def test_realised_rate_matches_the_requested_percentile(task, tail):
    """Acceptance measures the tail mass, so it recovers the percentile asked for."""
    tail.sample(n_samples=N_SAMPLES)
    assert tail.realised_rate == pytest.approx(PERCENTILE / 100, abs=0.03)


def test_log_prob_is_the_base_shifted_by_a_constant(task, tail):
    """Truncation only renormalises, and everything above the cutoff is impossible."""
    tail.sample(n_samples=N_SAMPLES)
    inside = tail.sample(n_samples=100)
    shift = tail.log_prob(inside) - task.prior.log_prob(inside)
    assert torch.allclose(shift, shift[0].expand_as(shift))

    outside = task.prior.sample(n_samples=500)
    outside = outside[task.prior.log_prob(outside) >= tail.threshold]
    assert torch.isneginf(tail.log_prob(outside)).all()


def test_seeded_pilot_gives_every_method_the_same_threshold(task):
    """The pilot is seeded, so the tail does not drift between runs."""
    first = rare_tail_prior(task.prior, percentile=PERCENTILE, n_pilot=N_PILOT)
    second = rare_tail_prior(task.prior, percentile=PERCENTILE, n_pilot=N_PILOT)
    assert first.threshold == second.threshold


def test_pilot_leaves_the_global_rng_untouched(task):
    """Seeding the pilot must not also fix the evaluation's own draws."""
    torch.manual_seed(123)
    before = torch.get_rng_state()
    rare_tail_prior(task.prior, percentile=PERCENTILE, n_pilot=N_PILOT)
    assert torch.equal(torch.get_rng_state(), before)


def test_a_pinned_threshold_draws_no_pilot(task, monkeypatch):
    """The threshold route skips pilot sampling entirely."""
    monkeypatch.setattr(
        task.prior,
        "sample",
        lambda *a, **k: pytest.fail("threshold route should not sample a pilot"),
    )
    pinned = rare_tail_prior(task.prior, threshold=-7.5)
    assert pinned.threshold.item() == pytest.approx(-7.5)


def test_a_pinned_threshold_reports_the_percentile_it_realises(task, tail):
    """A threshold with no percentile gets one back from the acceptance rate."""
    pinned = rare_tail_prior(task.prior, threshold=tail.threshold.item())
    pinned.sample(n_samples=N_SAMPLES)
    assert rare_metrics(pinned)["rare_percentile"] == pytest.approx(PERCENTILE, abs=3.0)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"percentile": PERCENTILE, "threshold": -7.5}],
    ids=["neither", "both"],
)
def test_the_rare_set_must_be_specified_exactly_once(task, kwargs):
    """Neither is an error too: by here a rare evaluation has been asked for."""
    with pytest.raises(SystemExit, match="exactly one"):
        rare_tail_prior(task.prior, **kwargs)


def test_lower_bound_contrasts_against_the_rare_prior(task, tail):
    """With the two measures made equal, the lower bound is the shared-draw one.

    The upper bound is not, and should not be: `rare_spce_snmc` draws it a
    second, independent set of contrastives rather than reusing the first.
    """
    designer = RandomDesigner(task, 2)
    rollout = designer.rollout(tail.sample(n_samples=8))

    torch.manual_seed(5)
    lower, _ = rare_spce_snmc(
        task,
        rollout,
        200,
        50,
        fused=False,
        rare_prior=task.prior,
        base_prior=task.prior,
    )
    torch.manual_seed(5)
    expected, _ = chunked_spce_snmc(task, rollout, 200, 50, fused=False)
    assert torch.allclose(lower, expected)


def test_bounds_sandwich_and_restore_the_prior(task, tail):
    """The rare bounds bracket, and scoring leaves the task's prior as it found it."""
    designer = RandomDesigner(task, 2)
    rollout = designer.rollout(tail.sample(n_samples=8))

    lower, upper = rare_spce_snmc(
        task, rollout, 200, 50, fused=False, rare_prior=tail, base_prior=task.prior
    )
    assert lower.mean() <= upper.mean()
    assert task.current_prior is task.prior


def test_settings_come_from_the_recipe_group_not_the_run():
    """Settings are read post hoc, so any finished checkpoint can be scored."""
    assert rare_settings()["percentile"] is not None


def test_asking_for_a_threshold_clears_the_percentile():
    """The two are a pair: a flag replaces both, so they cannot both end up set."""
    settings = rare_settings(threshold=-7.5)
    assert settings["threshold"] == -7.5
    assert settings["percentile"] is None


def test_overrides_leave_unmentioned_settings_alone():
    """Only what was asked for changes; the rest stays as the group has it."""
    default = rare_settings()
    overridden = rare_settings(percentile=1, n_outer=64)
    assert overridden["n_outer"] == 64
    assert overridden["n_contrastive"] == default["n_contrastive"]


def test_eval_kwargs_draw_outer_theta_from_the_tail(task):
    """The outer expectation is over the rare set, not the prior."""
    eval_kwargs, tail = rare_eval_kwargs(task, percentile=PERCENTILE, n_pilot=N_PILOT)
    theta = eval_kwargs["theta_sampler"](64)
    assert (task.prior.log_prob(theta) < tail.threshold).all()
