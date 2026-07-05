import torch

from testbed.tasks import CES


def make_task():
    return CES()


def make_design(batch_shape=(4,)):
    return torch.rand(*batch_shape, 6) * 100


def test_event_shapes():
    task = make_task()
    assert task.event_shape_theta == torch.Size([5])
    assert task.event_shape_design == torch.Size([6])
    assert task.event_shape_y == torch.Size([1])


def test_prior_sample_shape_and_simplex():
    task = make_task()
    theta = task.sample_prior(16)

    assert theta.shape == (16, 5)
    # rho in (0, 1)
    assert ((theta[:, 0] > 0) & (theta[:, 0] < 1)).all()
    # alpha lies on the simplex
    assert torch.allclose(theta[:, 1:4].sum(-1), torch.ones(16), atol=1e-5)
    # u is positive
    assert (theta[:, 4] > 0).all()


def test_prior_log_prob_finite():
    task = make_task()
    theta = task.sample_prior(16)
    logp = task.prior.log_prob(theta)

    assert logp.shape == (16,)
    assert torch.isfinite(logp).all()


def test_sample_joint_shapes():
    task = make_task()
    design = make_design((4,))

    y, theta = task.sample_joint(design, n_samples=3)
    assert y.shape == (3, 4, 1)
    assert theta.shape == (3, 4, 5)


def test_outcome_logp_broadcasts():
    task = make_task()
    design = make_design((4,))
    y, theta = task.sample_joint(design, n_samples=3)

    logp = task.outcome_logp(y, theta, design)
    assert logp.shape == (3, 4)
    assert torch.isfinite(logp).all()


def test_outcomes_respect_censoring_bounds():
    task = make_task()
    design = make_design((8,))
    theta = task.sample_prior(8)

    y = task.sample_outcome(theta, design, n_samples=500)
    assert (y >= task.epsilon).all()
    assert (y <= 1.0 - task.epsilon).all()


def test_likelihood_normalizes():
    # The mixed (continuous + 2 boundary masses) density should integrate to 1.
    task = make_task()
    eps = task.epsilon

    # With uniform alpha, two baskets that are permutations of each other have
    # equal utility, giving mu = 0 and a broad sigma -- an interior, well-spread
    # density that the trapezoidal grid below can integrate accurately.
    theta = torch.tensor([[0.9, 1 / 3, 1 / 3, 1 / 3, 1.0]])
    design = torch.tensor([[10.0, 50.0, 90.0, 90.0, 50.0, 10.0]])
    mu, sigma = task.make_eta_params(theta, design)

    grid = torch.linspace(eps, 1.0 - eps, 200_000)[1:-1].unsqueeze(-1)
    log_cont = task.outcome_logp(grid, theta, design.expand(grid.shape[0], 6))
    cont_mass = torch.trapz(log_cont.exp(), grid.squeeze(-1))

    low = torch.special.log_ndtr((task.logit_lower - mu) / sigma).exp()
    high = torch.special.log_ndtr((mu - task.logit_upper) / sigma).exp()
    total = cont_mass + low.squeeze() + high.squeeze()

    assert torch.allclose(total, torch.tensor(1.0), atol=1e-2)


def test_outcome_logp_backward_finite_tiny_sigma():
    # Small u -> tiny sigma -> the censoring log_ndtr arguments blow up to ~1e8.
    # torch.where evaluates all branches, so a non-finite gradient in a
    # censoring branch used to poison interior rows (and saturated boundary
    # rows) via 0 * inf = NaN.
    #
    # Baskets use the design-box lower bound 0.01 (not 0): the reference model
    # evaluates x^rho directly, relying on the [0.01, 100] design box, so x = 0
    # is out of domain (its x^rho gradient is separately singular).
    task = make_task()
    eps = task.epsilon
    theta = torch.tensor(
        [
            [0.9, 1 / 3, 1 / 3, 1 / 3, 1e-7],
            [0.9, 1 / 3, 1 / 3, 1 / 3, 1e-7],
            [0.9, 1 / 3, 1 / 3, 1 / 3, 1e-7],
        ],
        requires_grad=True,
    )
    design = torch.tensor(
        [
            [0.01, 0.01, 0.01, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 0.01, 0.01, 0.01],
            [10.0, 50.0, 90.0, 90.0, 50.0, 10.0],
        ],
        requires_grad=True,
    )
    y = torch.tensor([[0.5], [eps], [1.0 - eps]])  # interior + both boundaries

    logp = task.outcome_logp(y, theta, design)
    assert torch.isfinite(logp).all()

    logp.sum().backward()
    assert torch.isfinite(theta.grad).all()
    assert torch.isfinite(design.grad).all()


def test_outcome_logp_backward_finite_prior_batch():
    # Check that the gradients are finite for a random batch.
    torch.manual_seed(0)
    task = make_task()
    theta = task.sample_prior(20_000).requires_grad_(True)
    design = (torch.rand(20_000, 6) * 100).requires_grad_(True)
    y = task.sample_outcome(theta.detach(), design.detach())

    logp = task.outcome_logp(y, theta, design)
    logp.sum().backward()
    assert torch.isfinite(theta.grad).all()
    assert torch.isfinite(design.grad).all()


def test_sample_outcome_is_reparametrized():
    task = make_task()
    design = (make_design((8,))).requires_grad_(True)
    theta = task.sample_prior(8)

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()
