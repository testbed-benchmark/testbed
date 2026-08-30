import jax.numpy as jnp
import numpy as np

from screening.action_rules import bayes_action, fixed_action
from screening.config import Config
from screening.metrics import decision_metrics, mean_interval, summarize
from screening.posterior import posterior, update
from screening.quadrature import expected_prior_max


def test_incremental_posterior_and_count_variance():
    config = Config()
    rng = np.random.default_rng(190)
    n = jnp.zeros((8, config.d), dtype=jnp.int32)
    q = jnp.zeros(n.shape, dtype=jnp.float64)
    ys = rng.normal(size=(24, 8))
    for t, y in enumerate(ys):
        n, q = update(n, q, jnp.zeros(8, dtype=jnp.int32), y)
        mean, sd = posterior(n, q, config)
        expected_var = (
            config.tau**2
            * config.sigma**2
            / (config.sigma**2 + (t + 1) * config.tau**2)
        )
        expected_mean = (
            config.sigma**2 * config.mu + config.tau**2 * ys[: t + 1].sum(0)
        ) / (config.sigma**2 + (t + 1) * config.tau**2)
        np.testing.assert_allclose(mean[:, 0], expected_mean, atol=1e-14)
        np.testing.assert_allclose(sd[:, 0] ** 2, expected_var, atol=1e-14)
        np.testing.assert_array_equal(sd, posterior(n, q * 17 + 11, config)[1])
        assert np.isfinite(mean).all() and np.isfinite(sd).all() and (sd > 0).all()


def test_posterior_predictive_moments():
    config = Config()
    mean, sd = posterior(jnp.array([5]), jnp.array([3.2]), config)
    rng = np.random.default_rng(876)
    y = rng.normal(float(mean[0]), float(sd[0]), 200000) + rng.normal(
        0, config.sigma, 200000
    )
    variance = float(sd[0] ** 2 + config.sigma**2)
    assert abs(y.mean() - mean[0]) < 5 * np.sqrt(variance / len(y))
    assert abs(y.var() - variance) < 5 * variance * np.sqrt(2 / (len(y) - 1))


def test_zero_agreement_100000_states_and_eligibility():
    rng = np.random.default_rng(77)
    mean = jnp.asarray(rng.normal(size=(100000, 16)))
    sd = jnp.asarray(np.exp(rng.normal(size=mean.shape)))
    np.testing.assert_array_equal(bayes_action(mean), fixed_action(mean, sd, 0.0))
    for kappa in (0.0, 1.0, 1.5, 2.0, 2.5, 3.0):
        eligible = np.asarray(mean - kappa * sd > 0)
        action = np.asarray(fixed_action(mean, sd, kappa))
        np.testing.assert_array_equal(action == 0, ~eligible.any(-1))
        mask = action > 0
        assert eligible[np.arange(len(action))[mask], action[mask] - 1].all()
        expected = np.where(eligible, mean, -np.inf).argmax(-1) + 1
        np.testing.assert_array_equal(action[mask], expected[mask])


def test_ties_default_boundary_and_selection_by_mean_not_margin():
    mean = jnp.array(
        [[0.0, 0.0, -1.0], [2.0, 2.0, 1.0], [3.0, 2.0, 1.0], [1.0, -1.0, -1.0]]
    )
    sd = jnp.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [2.5, 0.1, 0.1], [1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(fixed_action(mean, sd, 1.0), [0, 1, 1, 0])
    np.testing.assert_array_equal(bayes_action(mean), [0, 1, 1, 1])


def test_risk_rb_and_prior_max_monte_carlo():
    config = Config()
    prior = expected_prior_max(config.d, config.mu, config.tau)
    rng = np.random.default_rng(45)
    theta = rng.normal(config.mu, config.tau, (200000, config.d))
    maxima = np.maximum(0, theta.max(-1))
    assert abs(maxima.mean() - prior["value"]) < 5 * mean_interval(maxima)["se"]
    assert prior["quadrature_error"] < 1e-10
    n = jnp.full(theta.shape, 2)
    sums = 2 * theta + rng.normal(0, config.sigma * np.sqrt(2), theta.shape)
    mean, sd = posterior(n, sums, config)
    for kappa in config.kappa_grid:
        metrics = decision_metrics(mean, sd, theta, kappa)
        summary = summarize(metrics, prior["value"])
        assert summary["decomposition_error"] < 1e-10
        if kappa == 0:
            np.testing.assert_array_equal(metrics["incompatibility"], 0)
        difference = mean_interval(metrics["return"] - metrics["selected_utility"])
        assert abs(difference["mean"]) < 5 * difference["se"]
