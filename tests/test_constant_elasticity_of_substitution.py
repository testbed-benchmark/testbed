import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace
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
            [0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 0.0, 0.0, 0.0],
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


# A tiny u drives sigma = (1 + ||x - x'||) * tau * u down with it, which is the
# regime both numerical guards in _outcome_logp exist for. The two baskets differ
# by 0.001 rather than being identical, since ||x - x'|| is not differentiable at
# zero and that is a separate problem.
TINY_SIGMA_THETA = [0.9, 1 / 3, 1 / 3, 1 / 3, 2e-8]
NEAR_IDENTICAL_BASKETS = [10.0, 10.0, 10.0, 10.0, 10.0, 10.001]


@pytest.mark.parametrize("u", [2e-8, 1e-10, 1e-12])
def test_censored_rows_ignore_the_discarded_continuous_branch(u):
    # Regression test: torch.where evaluates the continuous branch even for a
    # censored y and hands it a zero upstream gradient. When that branch's own
    # gradient was non-finite, 0 * inf = NaN summed into theta.grad and
    # design.grad alongside the real contribution from the censoring branch.
    task = make_task()
    theta_values = TINY_SIGMA_THETA[:4] + [u]
    y = torch.tensor([[task.epsilon]])

    theta = torch.tensor([theta_values], requires_grad=True)
    design = torch.tensor([NEAR_IDENTICAL_BASKETS], requires_grad=True)
    task.outcome_logp(y, theta, design).sum().backward()

    # The censoring branch alone, which is what a censored row should see.
    theta_ref = torch.tensor([theta_values], requires_grad=True)
    design_ref = torch.tensor([NEAR_IDENTICAL_BASKETS], requires_grad=True)
    mu, sigma = task.make_eta_params(theta_ref, design_ref)
    log_lower = torch.special.log_ndtr(
        ((task.logit_lower - mu) / sigma).clamp(min=-1e3)
    )
    log_lower.sum().backward()

    assert torch.equal(theta.grad, theta_ref.grad)
    assert torch.equal(design.grad, design_ref.grad)


@pytest.mark.parametrize("y_value", [1e-6, 1e-5, 1.0 - 1e-6])
def test_outcome_logp_backward_finite_near_boundary_with_tiny_sigma(y_value):
    # The continuous branch is selected here, since y sits inside the censoring
    # bounds, but logit(y) is ~14 away from mu while sigma is ~1e-10. That is
    # what made normal_log_prob's gradient overflow: the value is large but
    # representable, so the gradient has to be too.
    task = make_task()
    theta = torch.tensor([TINY_SIGMA_THETA], requires_grad=True)
    design = torch.tensor([NEAR_IDENTICAL_BASKETS], requires_grad=True)

    logp = task.outcome_logp(torch.tensor([[y_value]]), theta, design)
    assert torch.isfinite(logp).all()

    logp.sum().backward()
    assert torch.isfinite(theta.grad).all()
    assert torch.isfinite(design.grad).all()
    assert (theta.grad.abs() > 0).any()


def test_sample_outcome_is_reparametrized():
    task = make_task()
    design = (make_design((8,))).requires_grad_(True)
    theta = task.sample_prior(8)

    y = task.sample_outcome(theta, design)
    y.sum().backward()

    assert design.grad is not None
    assert torch.isfinite(design.grad).all()
    assert (design.grad.abs() > 0).any()


def test_default_design_space_is_the_basket_box():
    task = make_task()

    assert isinstance(task.design_space, BoxDesignSpace)
    assert task.design_space.event_shape == torch.Size([6])
    assert torch.equal(task.design_space.low, torch.zeros(6))
    assert torch.equal(task.design_space.high, torch.full((6,), 100.0))


def test_design_space_accepts_the_closed_box_and_rejects_outside():
    task = make_task()
    space = task.design_space

    # The boundary baskets are legitimate designs.
    assert space.contains(torch.zeros(6))
    assert space.contains(torch.full((6,), 100.0))
    assert not space.contains(torch.full((6,), -1.0))
    assert not space.contains(torch.full((6,), 101.0))


def test_task_rejects_designs_outside_the_box():
    task = make_task()
    theta = task.sample_prior(1)
    design = torch.full((1, 6), 150.0)

    with pytest.raises(ValueError, match="Design lies outside"):
        task.sample_outcome(theta, design)


def test_design_space_sample_and_project_stay_in_the_box():
    torch.manual_seed(0)
    task = make_task()
    space = task.design_space

    assert space.contains(space.sample((64,)))
    assert space.contains(space.project(torch.randn(64, 6) * 500))


def test_design_space_reparametrisation_maps_into_the_box():
    torch.manual_seed(0)
    task = make_task()
    space = task.design_space

    # Deep in tanh saturation the inverse loses float32 precision: recovering
    # raw from a design near 0 or 100 subtracts the bias of 50 from it, so the
    # round-trip is only accurate while the squash is off its asymptotes.
    raw = (torch.randn(64, 6) * 3).clamp(-2.5, 2.5)
    design = space.to_design(raw)

    assert space.contains(design)
    assert torch.allclose(space.to_raw(design), raw, atol=1e-3)


def test_design_space_reparametrisation_stays_in_the_box_when_saturated():
    torch.manual_seed(0)
    task = make_task()
    space = task.design_space

    # Whatever a policy emits, the design is valid -- that is the point of
    # squashing rather than clipping.
    design = space.to_design(torch.randn(256, 6) * 50)

    assert space.contains(design)
    assert torch.isfinite(space.to_raw(design)).all()
