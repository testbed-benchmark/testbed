"""Tests for the inference API: LogPosterior, the engines, and PosteriorUpdateHook."""

import pickle
import sys
from functools import partial

import pytest
import torch
from torch.distributions import MultivariateNormal, biject_to

import testbed.inference
from testbed.core.distributions import TorchDistribution
from testbed.core.hooks import PosteriorUpdateHook, RolloutHook
from testbed.design_objectives import NMC
from testbed.designers import RandomDesigner, SequentialDesigner
from testbed.inference import LogPosterior, run_importance_sampling
from testbed.tasks import CES, LinearRegression, LocationFinding, Pharmacokinetic


def _linear_regression_history(n=8, noise_scale=0.5):
    """A history on LinearRegression, with its exact Gaussian posterior."""
    torch.manual_seed(0)
    task = LinearRegression(dim=1, noise_scale=noise_scale)
    designs = torch.linspace(-1, 1, n).unsqueeze(-1)
    y = task.sample_outcome(task.sample_prior(), designs)

    features = torch.cat([torch.ones(n, 1), designs], dim=-1)  # mean is theta . [1, x]
    prior_precision = torch.linalg.inv(task.prior.dist.covariance_matrix)
    cov = torch.linalg.inv(prior_precision + features.T @ features / noise_scale**2)
    mean = cov @ (
        prior_precision @ task.prior.dist.mean + features.T @ y[:, 0] / noise_scale**2
    )
    return task, y, designs, mean, cov


def _assert_matches(posterior, mean, cov, atol):
    samples = posterior.sample((20_000,))
    torch.testing.assert_close(samples.mean(0), mean, atol=atol, rtol=0)
    torch.testing.assert_close(samples.T.cov(), cov, atol=atol / 3, rtol=0)


# ---------------------------------------------------------------------------
# LogPosterior
# ---------------------------------------------------------------------------


def test_log_posterior_adds_the_jacobian_of_the_support_bijection():
    torch.manual_seed(0)
    task = Pharmacokinetic()
    designs = task.design_prior.sample((4,))
    y = task.sample_outcome(task.sample_prior(), designs)
    log_post = LogPosterior(task, y, designs)
    z = torch.randn(6, log_post.dim)

    theta = log_post.to_theta(z)
    transform = biject_to(task.prior.support)
    expected = (
        task.prior.log_prob(theta)
        + transform.log_abs_det_jacobian(z, theta)
        + task.history_logp(y, designs, theta)
    )

    torch.testing.assert_close(log_post(z), expected)
    assert task.prior.support.check(theta).all()


@pytest.mark.parametrize(
    "task, dim", [(CES(), 4), (LocationFinding(n_sources=2, dim_theta=3), 6)]
)
def test_log_posterior_uses_flat_raw_coordinates(task, dim):
    designs = task.design_prior.sample((3,))
    y = task.sample_outcome(task.sample_prior(), designs)
    log_post = LogPosterior(task, y, designs)
    theta = task.sample_prior((5,))

    z = log_post.to_raw(theta)

    assert log_post.dim == dim
    assert z.shape == (5, dim)
    assert log_post(z).shape == (5,)
    torch.testing.assert_close(log_post.to_theta(z), theta, rtol=1e-4, atol=1e-4)


def test_log_posterior_matches_a_hand_written_change_of_variables():
    # Pharmacokinetic's parameters are positive, so the bijection is exp and
    # the Jacobian term is sum(z): the whole density can be written by hand,
    # independently of torch's transform machinery.
    torch.manual_seed(0)
    task = Pharmacokinetic()
    designs = task.design_prior.sample((4,))
    y = task.sample_outcome(task.sample_prior(), designs)
    log_post = LogPosterior(task, y, designs)
    z = torch.randn(6, 3)

    expected = (
        task.prior.log_prob(z.exp())
        + z.sum(-1)
        + task.history_logp(y, designs, z.exp())
    )

    torch.testing.assert_close(log_post(z), expected)
    torch.testing.assert_close(log_post.to_theta(z), z.exp())


def test_log_posterior_squeezes_a_batch_of_one():
    task, y, designs, _, _ = _linear_regression_history()
    z = torch.randn(4, 2)

    batched = LogPosterior(task, y.unsqueeze(0), designs.unsqueeze(0))

    torch.testing.assert_close(batched(z), LogPosterior(task, y, designs)(z))


# ---------------------------------------------------------------------------
# Engines, against linear regression's exact posterior
# ---------------------------------------------------------------------------


def test_importance_sampling_recovers_the_exact_posterior():
    task, y, designs, mean, cov = _linear_regression_history()

    _assert_matches(run_importance_sampling(task, y, designs), mean, cov, atol=0.03)


def test_mcmc_recovers_the_exact_posterior():
    pytest.importorskip("pyro")
    task, y, designs, mean, cov = _linear_regression_history()
    torch.manual_seed(0)

    posterior = testbed.inference.run_mcmc(
        task, y, designs, n_samples=600, n_warmup=300
    )

    _assert_matches(posterior, mean, cov, atol=0.06)


def test_mcmc_samples_stay_in_a_constrained_support():
    pytest.importorskip("pyro")
    torch.manual_seed(0)
    task = Pharmacokinetic()
    designs = task.design_prior.sample((4,))
    y = task.sample_outcome(task.sample_prior(), designs)

    posterior = testbed.inference.run_mcmc(task, y, designs, n_samples=50, n_warmup=50)

    assert task.prior.support.check(posterior.locs).all()


def test_mcmc_pools_the_samples_of_several_chains():
    pytest.importorskip("pyro")
    task, y, designs, _, _ = _linear_regression_history()
    torch.manual_seed(0)

    posterior = testbed.inference.run_mcmc(
        task, y, designs, n_samples=40, n_warmup=40, num_chains=2
    )

    assert posterior.locs.shape == (80, 2)
    assert torch.isfinite(posterior.locs).all()


def test_missing_pyro_raises_an_error_naming_the_extra(monkeypatch):
    # hide Pyro, including any of its modules an earlier test imported
    for name in [m for m in sys.modules if m.split(".")[0] == "pyro"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "pyro", None)
    monkeypatch.delitem(sys.modules, "testbed.inference.pyro_sampler", raising=False)

    with pytest.raises(ImportError, match=r"testbed\[pyro\]"):
        testbed.inference.run_mcmc  # noqa: B018


# ---------------------------------------------------------------------------
# PosteriorUpdateHook
# ---------------------------------------------------------------------------


def _gaussian_fit(task, y_history, design_history, calls):
    """A user-written inference: a Gaussian fitted to importance samples."""
    calls.append((y_history.shape, design_history.shape))
    samples = run_importance_sampling(task, y_history, design_history).sample((2_000,))
    flat = samples.reshape(len(samples), -1)
    return MultivariateNormal(
        flat.mean(0), flat.T.cov() + 1e-4 * torch.eye(flat.shape[1])
    )


def test_hook_passes_unbatched_histories_and_wraps_a_bare_torch_posterior():
    torch.manual_seed(0)
    task = LinearRegression(dim=1)
    calls = []
    hook = PosteriorUpdateHook(partial(_gaussian_fit, calls=calls), store_history=True)

    rollout = RandomDesigner(task, T=3, hooks=[hook]).rollout(task.sample_prior(1))

    assert calls == [((t, 1), (t, 1)) for t in (1, 2, 3)]
    assert isinstance(rollout.extra["posterior"], TorchDistribution)
    assert len(rollout.extra["posterior_history"]) == 3
    assert rollout.extra["posterior_history"][-1] is rollout.extra["posterior"]


def test_hook_restores_the_prior_and_returns_the_posterior():
    torch.manual_seed(0)
    task = LocationFinding(n_sources=1, dim_theta=2)
    designer = RandomDesigner(
        task,
        T=3,
        hooks=[PosteriorUpdateHook(partial(run_importance_sampling, n_proposal=500))],
    )

    rollout = designer.rollout(task.sample_prior(1))

    assert task.current_prior is task.prior
    assert rollout.extra["posterior"] is not task.prior


class _RecordPrior(RolloutHook):
    """Records task.current_prior as seen at the end of the rollout."""

    def on_rollout_end(self, state):
        return {"prior_at_end": state.task.current_prior}


class _FailAtStep(RolloutHook):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def after_step(self, state):
        if state.t == self.t:
            raise RuntimeError("boom")


def test_hooks_after_the_posterior_update_see_the_posterior_at_rollout_end():
    task = LocationFinding(n_sources=1, dim_theta=2)
    inference = partial(run_importance_sampling, n_proposal=500)
    designer = RandomDesigner(
        task, T=2, hooks=[PosteriorUpdateHook(inference), _RecordPrior()]
    )

    rollout = designer.rollout(task.sample_prior(1))

    assert rollout.extra["prior_at_end"] is rollout.extra["posterior"]
    assert rollout.extra["prior_at_end"] is not task.prior
    assert task.current_prior is task.prior


def test_a_failing_rollout_still_restores_the_prior():
    task = LocationFinding(n_sources=1, dim_theta=2)
    inference = partial(run_importance_sampling, n_proposal=500)
    designer = RandomDesigner(
        task, T=3, hooks=[PosteriorUpdateHook(inference), _FailAtStep(1)]
    )

    with pytest.raises(RuntimeError, match="boom"):
        designer.rollout(task.sample_prior(1))

    assert task.current_prior is task.prior


def _sequential_rollout(task, seed):
    designer = SequentialDesigner(
        task,
        T=2,
        objective=NMC(task, n_outer=4, n_inner=4),
        lr=0.05,
        n_step_opt=2,
        inference=partial(run_importance_sampling, n_proposal=500),
    )
    torch.manual_seed(seed)
    return designer.rollout(task.sample_prior(1))


def test_consecutive_rollouts_on_one_task_match_fresh_tasks():
    # Regression: the posterior left on a reused task used to steer the next
    # rollout's first design.
    task = LocationFinding(n_sources=1, dim_theta=2)
    reused = [_sequential_rollout(task, seed) for seed in (1, 2)]
    fresh = [
        _sequential_rollout(LocationFinding(n_sources=1, dim_theta=2), seed)
        for seed in (1, 2)
    ]

    for a, b in zip(reused, fresh):
        torch.testing.assert_close(a.design_history, b.design_history)
        torch.testing.assert_close(a.y_history, b.y_history)


def test_log_posterior_pickles_for_engines_that_use_other_processes():
    # e.g. Pyro runs extra MCMC chains in spawned processes on macOS or CUDA
    task = CES()
    designs = task.design_prior.sample((3,))
    y = task.sample_outcome(task.sample_prior(), designs)
    log_post = LogPosterior(task, y, designs)
    z = log_post.to_raw(task.sample_prior((4,)))

    torch.testing.assert_close(pickle.loads(pickle.dumps(log_post))(z), log_post(z))
