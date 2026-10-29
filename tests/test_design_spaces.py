import pytest
import torch
from torch.distributions import Normal, constraints
from torch.distributions.transforms import ExpTransform, Transform

from testbed.core import Box, EmpiricalDistribution, TorchDistribution
from testbed.core.constraints import default_design_transform, in_support
from testbed.core.distributions import Distribution
from testbed.envs.bed_env import _project
from testbed.tasks import LocationFinding, PreyPopulation
from testbed.tasks.temporal_discounting import TDDesignPrior, TDDesignTransform


class PositivePrior(Distribution):
    """Minimal design prior over positive designs; only declares its support."""

    def __init__(self, event_shape):
        super().__init__(event_shape=event_shape)

    @property
    def support(self):
        return constraints.independent(constraints.positive, len(self.event_shape))

    def sample(self, sample_shape=()):
        return torch.rand(self._extended_shape(sample_shape)) + 0.5

    def log_prob(self, x):
        raise NotImplementedError


def _standard_normal(event_shape):
    return TorchDistribution(Normal(torch.zeros(event_shape), 1.0))


def _autograd_log_det(transform, raw):
    """Log |det J| of transform at a single raw point, via autograd."""
    jacobian = torch.autograd.functional.jacobian(transform, raw)
    return torch.linalg.slogdet(jacobian)[1]


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


def test_in_support_is_all_or_nothing_over_a_batch():
    support = PositivePrior((2,)).support

    assert in_support(support, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    # One bad element in the batch fails the whole batch.
    assert not in_support(support, torch.tensor([[1.0, 2.0], [3.0, -4.0]]))


def test_unconstrained_support_accepts_any_value():
    support = _standard_normal((2,)).support

    assert in_support(support, torch.tensor([-1e30, 1e30]))
    assert in_support(support, torch.tensor([float("nan"), float("inf")]))


def test_contains_design_checks_shape():
    task = LocationFinding(n_sources=1, dim_theta=2)

    assert task.contains_design(torch.ones(5, 4, 2))  # leading dims are batch
    assert task.contains_design([1.0, 2.0])  # non-tensor input
    assert not task.contains_design(torch.ones(3))
    assert not task.contains_design(torch.ones(5, 1))
    assert not task.contains_design(torch.tensor(1.0))


def test_task_rejects_a_design_prior_of_the_wrong_shape():
    with pytest.raises(ValueError, match="event_shape_design"):
        LocationFinding(n_sources=1, dim_theta=2, design_prior=_standard_normal((3,)))


# ---------------------------------------------------------------------------
# Change of variables
# ---------------------------------------------------------------------------


def test_default_transform_follows_biject_to():
    transform = default_design_transform(PositivePrior((2,)).support)
    raw = torch.tensor([0.3, -1.2])

    assert torch.allclose(transform(raw), torch.exp(raw))
    assert torch.allclose(
        transform.log_abs_det_jacobian(raw, transform(raw)),
        _autograd_log_det(transform, raw),
    )


def test_transform_without_log_det_jacobian_raises_rather_than_silently_wrong():
    class Forgetful(ExpTransform):
        def log_abs_det_jacobian(self, x, y):
            return Transform.log_abs_det_jacobian(self, x, y)

    raw = torch.randn(4, 2)

    assert torch.allclose(Forgetful()(raw), torch.exp(raw))  # still fine
    with pytest.raises(NotImplementedError):
        Forgetful().log_abs_det_jacobian(raw, torch.exp(raw))


def test_unconstrained_transform_is_the_identity_with_zero_log_det():
    transform = default_design_transform(_standard_normal((2, 3)).support)
    raw = torch.randn(4, 5, 2, 3)

    log_det = transform.log_abs_det_jacobian(raw, transform(raw))

    assert torch.equal(transform(raw), raw)
    assert log_det.shape == (4, 5)
    assert torch.all(log_det == 0)


def test_box_log_det_matches_autograd():
    box = Box(torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 7.0]), event_shape=(2,))
    transform = default_design_transform(box.support)
    raw = torch.tensor([0.4, -0.9])

    assert torch.allclose(
        transform.log_abs_det_jacobian(raw, transform(raw)),
        _autograd_log_det(transform, raw),
    )


def test_td_log_det_matches_autograd():
    transform = TDDesignTransform(log_k_mean=-3.0)
    raw = torch.tensor([0.7, -0.4])

    assert torch.allclose(
        transform.log_abs_det_jacobian(raw, transform(raw)),
        _autograd_log_det(transform, raw),
    )


# ---------------------------------------------------------------------------
# Standard normal (the default)
# ---------------------------------------------------------------------------


def test_standard_normal_samples_requested_shape():
    prior = _standard_normal((2,))

    assert prior.sample().shape == (2,)
    assert prior.sample((4,)).shape == (4, 2)
    assert prior.sample((4, 3)).shape == (4, 3, 2)


def test_standard_normal_is_the_default_design_prior():
    task = LocationFinding(n_sources=1, dim_theta=2)

    assert isinstance(task.design_prior.dist.base_dist, Normal)
    assert task.design_prior.event_shape == torch.Size([2])


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------


def test_box_samples_with_integer_bounds():
    box = Box(-1, 1, event_shape=(2,))

    sample = box.sample((4,))

    assert torch.is_floating_point(sample)
    assert sample.shape == (4, 2)
    assert in_support(box.support, sample)


def test_box_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        Box(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), event_shape=(2,))


def test_box_support_and_projection():
    box = Box(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]), event_shape=(2,))

    assert box.event_shape == torch.Size([2])
    assert in_support(box.support, torch.tensor([[0.0, 1.0], [1.0, 2.0]]))
    assert not in_support(box.support, torch.tensor([0.0, 3.0]))
    assert torch.equal(
        _project(box.support, torch.tensor([-2.0, 3.0])), torch.tensor([-1.0, 2.0])
    )


def test_box_to_design_lands_in_the_box_and_round_trips():
    box = Box(torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 7.0]), event_shape=(2,))
    transform = default_design_transform(box.support)
    raw = torch.randn(64, 2)

    design = transform(raw)

    bias, scale = torch.tensor([1.0, 5.0]), torch.tensor([3.0, 2.0])
    assert torch.allclose(design, bias + scale * torch.tanh(raw))
    assert in_support(box.support, design)
    assert torch.allclose(transform.inv(design), raw, atol=1e-4)


def test_box_inverse_is_finite_on_the_boundary():
    box = Box(0.0, 24.0, event_shape=(1,))
    transform = default_design_transform(box.support)

    raw = transform.inv(torch.tensor([[0.0], [24.0]]))

    assert torch.isfinite(raw).all()


def test_box_reparametrisation_requires_finite_positive_width():
    degenerate = Box(1.0, 1.0, event_shape=(2,))

    # A zero-width box is still a legitimate space for membership.
    assert in_support(degenerate.support, torch.ones(2))
    with pytest.raises(ValueError, match="finite bounds"):
        default_design_transform(degenerate.support)


# ---------------------------------------------------------------------------
# Finite design sets
# ---------------------------------------------------------------------------


def _finite_set():
    return EmpiricalDistribution(torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]))


def test_finite_set_encodes_and_decodes_physical_designs():
    designs_set = EmpiricalDistribution(torch.tensor([[1.0], [5.0], [10.0]]))

    designs = torch.tensor([[10.0], [1.0], [5.0]])
    indices = designs_set.encode(designs)

    assert in_support(designs_set.support, designs)
    assert torch.equal(indices, torch.tensor([2, 0, 1]))
    assert torch.equal(designs_set.decode(indices), designs)
    assert not in_support(designs_set.support, torch.tensor([[2.0]]))


def test_finite_set_derives_shape_and_discreteness():
    designs_set = _finite_set()

    assert designs_set.event_shape == torch.Size([2])
    assert designs_set.locs.shape[0] == 3
    assert designs_set.support.is_discrete


def test_finite_set_membership():
    support = _finite_set().support

    assert in_support(support, torch.tensor([1.0, 2.0]))
    assert in_support(support, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert not in_support(support, torch.tensor([1.0, 5.0]))


def test_finite_set_encode_decode_handle_unbatched_designs():
    designs_set = _finite_set()

    index = designs_set.encode(torch.tensor([1.0, 2.0]))

    assert index.shape == ()
    assert designs_set.decode(index).shape == (2,)


def test_finite_set_encode_rejects_invalid_designs():
    designs_set = _finite_set()

    with pytest.raises(ValueError, match="trailing shape"):
        designs_set.encode(torch.tensor([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="outside the finite set"):
        designs_set.encode(torch.tensor([9.0, 9.0]))


def test_finite_set_decode_rejects_invalid_indices():
    designs_set = _finite_set()

    with pytest.raises(ValueError, match="integer-valued"):
        designs_set.decode(torch.tensor([0.5]))
    with pytest.raises(ValueError, match="outside the finite set"):
        designs_set.decode(torch.tensor([3]))


def test_finite_set_projects_to_the_nearest_design():
    support = _finite_set().support

    projected = _project(support, torch.tensor([[0.9, 1.9], [2.6, 3.6]]))

    assert torch.equal(projected, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert in_support(support, projected)


def test_finite_set_samples_land_in_the_set():
    torch.manual_seed(0)
    designs_set = _finite_set()

    assert designs_set.sample().shape == (2,)
    assert in_support(designs_set.support, designs_set.sample())
    assert in_support(designs_set.support, designs_set.sample((32,)))
    assert designs_set.sample((4, 3)).shape == (4, 3, 2)


def test_finite_set_reparametrisation_is_the_identity():
    # A finite set admits no smooth reparametrisation, so the transform must
    # pass physical designs straight through -- a designer with the default
    # transform_designs=True runs everything it proposes through it.
    transform = default_design_transform(_finite_set().support)
    design = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    assert torch.equal(transform(design), design)
    assert torch.equal(transform.inv(design), design)
    assert torch.equal(transform.log_abs_det_jacobian(design, design), torch.zeros(2))


def test_finite_set_supports_scalar_designs():
    designs_set = EmpiricalDistribution(torch.tensor([1.0, 2.0, 3.0]))

    assert designs_set.event_shape == torch.Size([])
    assert in_support(designs_set.support, torch.tensor([1.0, 3.0]))
    assert torch.equal(
        designs_set.encode(torch.tensor([3.0, 1.0])), torch.tensor([2, 0])
    )
    assert torch.equal(
        _project(designs_set.support, torch.tensor([2.9])), torch.tensor([3.0])
    )


def test_prey_population_uses_a_finite_design_set():
    task = PreyPopulation()

    assert isinstance(task.design_prior, EmpiricalDistribution)
    assert task.design_prior.support.is_discrete
    assert task.contains_design(torch.tensor([[1.0], [300.0]]))
    assert not task.contains_design(torch.tensor([[0.5]]))


# ---------------------------------------------------------------------------
# Temporal discounting
# ---------------------------------------------------------------------------


def test_td_design_space_domain_is_closed():
    # Closed, matching Step-DAD: exp and sigmoid saturate onto the boundary in
    # float32, and the likelihood is well defined there.
    support = TDDesignPrior().support

    assert in_support(support, torch.tensor([0.0, 50.0]))  # D_1 = 0 is allowed
    assert in_support(support, torch.tensor([1.0, 0.0]))  # R_0 = 0 is allowed
    assert in_support(support, torch.tensor([1.0, 100.0]))  # R_0 = 100 is allowed

    assert not in_support(support, torch.tensor([-1e-6, 50.0]))  # D_1 < 0
    assert not in_support(support, torch.tensor([1.0, -1e-6]))  # R_0 < 0
    assert not in_support(support, torch.tensor([1.0, 100.001]))  # R_0 > 100


def test_td_sample_lands_in_the_space():
    torch.manual_seed(0)
    prior = TDDesignPrior(log_k_mean=-3.0)

    assert in_support(prior.support, prior.sample((256,)))


def test_td_to_design_survives_float_saturation():
    # Regression: exp and 100*sigmoid saturate onto the domain boundary in
    # float32 (from raw around -104 and +17), which a DAD policy net reaches
    # routinely. Those designs must stay valid and finite -- they used to crash
    # training against a strict domain.
    transform = TDDesignTransform(log_k_mean=-3.125)
    extreme = torch.tensor(
        [[-200.0, -200.0], [-120.0, -120.0], [20.0, 20.0], [200.0, 200.0]]
    )

    design = transform(extreme)

    assert in_support(TDDesignPrior().support, design)
    assert torch.isfinite(design).all()
    # Saturation genuinely reaches the bounds; that is allowed, not clamped away.
    assert (design[:, 0] == 0.0).any()
    assert (design[:, 1] == 100.0).any()


def test_td_to_design_gradients_stay_finite_when_saturated():
    # Regression: bounding D_1 by clamping exp's *output* makes the backward
    # pass compute 0 * inf = nan, which silently poisoned the policy net a few
    # thousand steps into DAD training. The bound goes on the exponent instead.
    transform = TDDesignTransform(log_k_mean=-3.125)
    raw = torch.tensor(
        [[-200.0, -200.0], [0.0, 0.0], [200.0, 200.0]], requires_grad=True
    )

    transform(raw).sum().backward()

    assert torch.isfinite(raw.grad).all()


def test_td_project_output_is_contained():
    # Regression: project() clamped R_0 to the closed [0, 100] while contains()
    # required the open (0, 100), so a projected design failed validation --
    # BEDEnv projects and then validates, so this raised on every step whose
    # design fell outside the domain.
    support = TDDesignPrior().support
    outside = torch.tensor([[5.0, -20.0], [1.0, 500.0], [-3.0, 100.0]])

    projected = _project(support, outside)

    assert torch.equal(
        projected, torch.tensor([[5.0, 0.0], [1.0, 100.0], [0.0, 100.0]])
    )
    assert in_support(support, projected)


def test_td_to_design_round_trips():
    transform = TDDesignTransform(log_k_mean=-3.0)
    raw = torch.randn(16, 2)

    assert torch.allclose(transform.inv(transform(raw)), raw, atol=1e-4)


def test_projection_raises_for_a_support_it_cannot_project_onto():
    with pytest.raises(NotImplementedError):
        _project(constraints.simplex, torch.tensor([0.2, 0.9]))


@pytest.mark.parametrize(
    "values", [torch.zeros(0, 2), torch.tensor([[1.0, 2.0], [1.0, 2.0]])]
)
def test_task_rejects_an_empty_or_duplicated_finite_design_set(values):
    with pytest.raises(ValueError, match="each listed once"):
        LocationFinding(
            n_sources=1, dim_theta=2, design_prior=EmpiricalDistribution(values)
        )
