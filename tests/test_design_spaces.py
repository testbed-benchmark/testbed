import pytest
import torch

from testbed.core.design_spaces import (
    BoxDesignSpace,
    DefaultDesignSpace,
    DesignSpace,
    FiniteDesignSpace,
)
from testbed.tasks.temporal_discounting import TDDesignSpace


class PositiveSpace(DesignSpace):
    """Minimal subclass: implements only the required _contains."""

    def __init__(self, event_shape):
        super().__init__(event_shape, is_discrete=False)

    def _contains(self, design):
        return bool(torch.all(design > 0).item())


class ExpSpace(DesignSpace):
    """Positive space via an exp reparametrisation, with its log-det."""

    def __init__(self, event_shape):
        super().__init__(event_shape, is_discrete=False)

    def _contains(self, design):
        return bool(torch.all(design > 0).item())

    def _to_design(self, raw_design):
        return torch.exp(raw_design)

    def _to_raw(self, design):
        return torch.log(design)

    def _log_det_jacobian(self, raw_design):
        return self._sum_event_dims(raw_design)


# ---------------------------------------------------------------------------
# DesignSpace base contract
# ---------------------------------------------------------------------------


def test_subclass_only_needs_contains():
    space = PositiveSpace(event_shape=(2,))

    assert space.contains(torch.tensor([1.0, 2.0]))
    assert not space.contains(torch.tensor([1.0, -2.0]))


def test_unimplemented_optional_methods_raise_rather_than_defaulting():
    # A space that does not project or reparametrise must say so: an identity
    # default would silently hand back an invalid design as if it were valid.
    space = PositiveSpace(event_shape=(2,))
    design = torch.tensor([1.0, -2.0])

    with pytest.raises(NotImplementedError):
        space.project(design)
    with pytest.raises(NotImplementedError):
        space.to_design(design)
    with pytest.raises(NotImplementedError):
        space.to_raw(design)


def test_contains_checks_shape_without_subclass_help():
    # PositiveSpace never looks at shape; the base class does it.
    space = PositiveSpace(event_shape=(2,))

    assert space.contains(torch.ones(5, 4, 2))  # leading dims are batch
    assert not space.contains(torch.ones(3))
    assert not space.contains(torch.ones(5, 1))
    assert not space.contains(torch.tensor(1.0))


def test_contains_is_all_or_nothing_over_a_batch():
    space = PositiveSpace(event_shape=(2,))

    assert space.contains(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    # One bad element in the batch fails the whole batch.
    assert not space.contains(torch.tensor([[1.0, 2.0], [3.0, -4.0]]))


def test_contains_accepts_non_tensor_input():
    space = PositiveSpace(event_shape=(2,))

    assert space.contains([1.0, 2.0])
    assert not space.contains([1.0, -2.0])


def test_subclass_must_implement_contains():
    class NoContains(DesignSpace):
        pass

    with pytest.raises(TypeError):
        NoContains(event_shape=(2,), is_discrete=False)


def test_sample_raises_by_default():
    space = PositiveSpace(event_shape=(2,))

    with pytest.raises(NotImplementedError, match="PositiveSpace"):
        space.sample()


def test_event_shape_is_normalized_to_a_size():
    space = PositiveSpace(event_shape=[2, 3])

    assert space.event_shape == torch.Size([2, 3])
    assert space.is_discrete is False


# ---------------------------------------------------------------------------
# Change of variables
# ---------------------------------------------------------------------------


def _autograd_log_det(space, raw):
    """Log |det J| of space.to_design at a single raw point, via autograd."""
    jacobian = torch.autograd.functional.jacobian(space.to_design, raw)
    return torch.linalg.slogdet(jacobian)[1]


def test_to_design_without_log_prob_returns_a_bare_tensor():
    space = ExpSpace(event_shape=(2,))
    raw = torch.randn(4, 2)

    design = space.to_design(raw)

    assert isinstance(design, torch.Tensor)
    assert torch.allclose(design, torch.exp(raw))


def test_to_design_applies_the_change_of_variables():
    space = ExpSpace(event_shape=(2,))
    raw = torch.randn(4, 2)
    log_prob = torch.randn(4)

    design, new_log_prob = space.to_design(raw, log_prob)

    assert torch.allclose(design, torch.exp(raw))
    assert new_log_prob.shape == (4,)
    assert torch.allclose(new_log_prob, log_prob - raw.sum(-1))


def test_to_raw_inverts_the_change_of_variables():
    space = ExpSpace(event_shape=(2,))
    raw = torch.randn(4, 2)
    log_prob = torch.randn(4)

    design, design_log_prob = space.to_design(raw, log_prob)
    round_trip, round_trip_log_prob = space.to_raw(design, design_log_prob)

    assert torch.allclose(round_trip, raw, atol=1e-6)
    assert torch.allclose(round_trip_log_prob, log_prob, atol=1e-6)


def test_identity_space_change_of_variables_is_a_no_op():
    space = DefaultDesignSpace(event_shape=(3,))
    raw = torch.randn(5, 3)
    log_prob = torch.randn(5)

    design, new_log_prob = space.to_design(raw, log_prob)

    assert torch.equal(design, raw)
    assert torch.equal(new_log_prob, log_prob)


def test_identity_space_log_det_is_zeros_with_batch_shape():
    space = DefaultDesignSpace(event_shape=(2, 3))
    raw = torch.randn(4, 5, 2, 3)

    log_det = space._log_det_jacobian(raw)

    assert log_det.shape == (4, 5)
    assert torch.all(log_det == 0)


def test_transform_without_log_det_jacobian_raises_rather_than_silently_wrong():
    # A subclass that reparametrises but forgets _log_det_jacobian would
    # otherwise get a zero correction, i.e. silently wrong log-probs.
    class Forgetful(DesignSpace):
        def __init__(self):
            super().__init__((2,), is_discrete=False)

        def _contains(self, design):
            return True

        def _to_design(self, raw_design):
            return torch.exp(raw_design)

    space = Forgetful()
    raw = torch.randn(4, 2)

    assert torch.allclose(space.to_design(raw), torch.exp(raw))  # still fine
    with pytest.raises(NotImplementedError):
        space.to_design(raw, torch.zeros(4))


def test_exp_space_log_det_matches_autograd():
    space = ExpSpace(event_shape=(2,))
    raw = torch.tensor([0.3, -1.2])

    assert torch.allclose(space._log_det_jacobian(raw), _autograd_log_det(space, raw))


def test_box_log_det_matches_autograd():
    space = BoxDesignSpace(
        torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 7.0]), event_shape=(2,)
    )
    raw = torch.tensor([0.4, -0.9])

    assert torch.allclose(space._log_det_jacobian(raw), _autograd_log_det(space, raw))


def test_td_log_det_matches_autograd():
    space = TDDesignSpace(log_k_mean=-3.0)
    raw = torch.tensor([0.7, -0.4])

    assert torch.allclose(space._log_det_jacobian(raw), _autograd_log_det(space, raw))


# ---------------------------------------------------------------------------
# DefaultDesignSpace
# ---------------------------------------------------------------------------


def test_default_design_space_constrains_shape_only():
    space = DefaultDesignSpace(event_shape=(2,))

    assert space.contains(torch.tensor([-1e30, 1e30]))
    assert not space.contains(torch.ones(3))


def test_default_design_space_samples_requested_batch_shape():
    space = DefaultDesignSpace(event_shape=(2,))

    assert space.sample().shape == (2,)
    assert space.sample((4,)).shape == (4, 2)
    assert space.sample((4, 3)).shape == (4, 3, 2)


# ---------------------------------------------------------------------------
# BoxDesignSpace
# ---------------------------------------------------------------------------


def test_box_design_space_samples_with_integer_bounds():
    space = BoxDesignSpace(-1, 1, event_shape=(2,))

    sample = space.sample((4,))

    assert torch.is_floating_point(sample)
    assert sample.shape == (4, 2)
    assert space.contains(sample)


def test_box_design_space_rejects_invalid_bounds_and_shapes():
    with pytest.raises(ValueError):
        BoxDesignSpace(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), event_shape=(2,)
        )

    space = BoxDesignSpace(
        torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]), event_shape=(2,)
    )

    assert space.event_shape == torch.Size([2])
    assert space.contains(torch.tensor([[0.0, 1.0], [1.0, 2.0]]))
    assert not space.contains(torch.tensor([0.0, 1.0, 2.0]))
    assert not space.contains(torch.tensor([0.0, 3.0]))
    assert torch.equal(
        space.project(torch.tensor([-2.0, 3.0])),
        torch.tensor([-1.0, 2.0]),
    )


# ---------------------------------------------------------------------------
# FiniteDesignSpace
# ---------------------------------------------------------------------------


def _finite_space():
    return FiniteDesignSpace(torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]]))


def test_finite_design_space_encodes_and_decodes_physical_designs():
    space = FiniteDesignSpace(torch.tensor([[1.0], [5.0], [10.0]]))

    designs = torch.tensor([[10.0], [1.0], [5.0]])
    indices = space.encode(designs)

    assert space.n_designs == 3
    assert space.contains(designs)
    assert torch.equal(indices, torch.tensor([2, 0, 1]))
    assert torch.equal(space.decode(indices), designs)
    assert not space.contains(torch.tensor([[2.0]]))
    with pytest.raises(ValueError, match="integer-valued"):
        space.decode(torch.tensor([0.5]))


@pytest.mark.parametrize("values", [torch.tensor([]), torch.tensor([[1.0], [1.0]])])
def test_finite_design_space_rejects_empty_or_duplicate_actions(values):
    with pytest.raises(ValueError):
        FiniteDesignSpace(values)


def test_finite_design_space_derives_shape_and_discreteness():
    space = _finite_space()

    assert space.event_shape == torch.Size([2])
    assert space.n_designs == 3
    assert space.is_discrete is True


def test_finite_design_space_rejects_empty_and_duplicate_values():
    with pytest.raises(ValueError, match="at least one design"):
        FiniteDesignSpace(torch.zeros(0, 2))
    with pytest.raises(ValueError, match="unique"):
        FiniteDesignSpace(torch.tensor([[1.0, 2.0], [1.0, 2.0]]))


def test_finite_design_space_contains_uses_the_base_shape_check():
    space = _finite_space()

    assert space.contains(torch.tensor([1.0, 2.0]))
    assert space.contains(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert not space.contains(torch.tensor([1.0, 5.0]))  # not in the set
    assert not space.contains(torch.tensor([1.0, 2.0, 3.0]))  # wrong event shape


def test_finite_design_space_encode_decode_round_trip():
    space = _finite_space()
    designs = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    index = space.encode(designs)

    assert torch.equal(index, torch.tensor([2, 0]))
    assert torch.equal(space.decode(index), designs)


def test_finite_design_space_encode_decode_handle_unbatched_designs():
    space = _finite_space()

    index = space.encode(torch.tensor([1.0, 2.0]))

    assert index.shape == ()
    assert space.decode(index).shape == (2,)


def test_finite_design_space_encode_rejects_invalid_designs():
    space = _finite_space()

    with pytest.raises(ValueError, match="trailing shape"):
        space.encode(torch.tensor([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="outside the finite design space"):
        space.encode(torch.tensor([9.0, 9.0]))


def test_finite_design_space_decode_rejects_invalid_indices():
    space = _finite_space()

    with pytest.raises(ValueError, match="integer-valued"):
        space.decode(torch.tensor([0.5]))
    with pytest.raises(ValueError, match="outside the finite design space"):
        space.decode(torch.tensor([3]))


def test_finite_design_space_projects_to_the_nearest_design():
    space = _finite_space()

    projected = space.project(torch.tensor([[0.9, 1.9], [2.6, 3.6]]))

    assert torch.equal(projected, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert space.contains(projected)


def test_finite_design_space_samples_land_in_the_set():
    torch.manual_seed(0)
    space = _finite_space()

    assert space.sample().shape == (2,)
    assert space.contains(space.sample())
    assert space.contains(space.sample((32,)))
    assert space.sample((4, 3)).shape == (4, 3, 2)


def test_finite_design_space_reparametrisation_is_the_identity():
    # A finite set admits no smooth reparametrisation, so to_design/to_raw must
    # pass physical designs straight through -- a designer with the default
    # transform_designs=True runs everything it proposes through them.
    space = _finite_space()
    design = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    log_prob = torch.randn(2)

    assert torch.equal(space.to_design(design), design)
    assert torch.equal(space.to_raw(design), design)

    transformed, new_log_prob = space.to_design(design, log_prob)
    assert torch.equal(transformed, design)
    assert torch.equal(new_log_prob, log_prob)


def test_finite_design_space_supports_scalar_designs():
    space = FiniteDesignSpace(torch.tensor([1.0, 2.0, 3.0]))

    assert space.event_shape == torch.Size([])
    assert space.contains(torch.tensor([1.0, 3.0]))
    assert torch.equal(space.encode(torch.tensor([3.0, 1.0])), torch.tensor([2, 0]))
    assert torch.equal(space.project(torch.tensor([2.9])), torch.tensor([3.0]))


# ---------------------------------------------------------------------------
# TDDesignSpace
# ---------------------------------------------------------------------------


def test_td_design_space_inherits_the_base_shape_check():
    space = TDDesignSpace()

    assert space.event_shape == torch.Size([2])
    assert space.contains(torch.tensor([[1.0, 50.0], [2.0, 25.0]]))
    assert not space.contains(torch.tensor([1.0, 50.0, 25.0]))


def test_td_design_space_domain_is_closed():
    # Closed, matching Step-DAD: exp and sigmoid saturate onto the boundary in
    # float32, and the likelihood is well defined there.
    space = TDDesignSpace()

    assert space.contains(torch.tensor([0.0, 50.0]))  # D_1 = 0 is allowed
    assert space.contains(torch.tensor([1.0, 0.0]))  # R_0 = 0 is allowed
    assert space.contains(torch.tensor([1.0, 100.0]))  # R_0 = 100 is allowed

    assert not space.contains(torch.tensor([-1e-6, 50.0]))  # D_1 < 0
    assert not space.contains(torch.tensor([1.0, -1e-6]))  # R_0 < 0
    assert not space.contains(torch.tensor([1.0, 100.001]))  # R_0 > 100


def test_td_sample_lands_in_the_space():
    torch.manual_seed(0)
    space = TDDesignSpace(log_k_mean=-3.0)

    assert space.contains(space.sample((256,)))


def test_td_to_design_survives_float_saturation():
    # Regression: exp and 100*sigmoid saturate onto the domain boundary in
    # float32 (from raw around -104 and +17), which a DAD policy net reaches
    # routinely. Those designs must stay valid and finite -- they used to crash
    # training against a strict domain.
    space = TDDesignSpace(log_k_mean=-3.125)
    extreme = torch.tensor(
        [[-200.0, -200.0], [-120.0, -120.0], [20.0, 20.0], [200.0, 200.0]]
    )

    design = space.to_design(extreme)

    assert space.contains(design)
    assert torch.isfinite(design).all()
    # Saturation genuinely reaches the bounds; that is allowed, not clamped away.
    assert (design[:, 0] == 0.0).any()
    assert (design[:, 1] == 100.0).any()


def test_td_to_design_gradients_stay_finite_when_saturated():
    # Regression: bounding D_1 by clamping exp's *output* makes the backward
    # pass compute 0 * inf = nan, which silently poisoned the policy net a few
    # thousand steps into DAD training. The bound goes on the exponent instead.
    space = TDDesignSpace(log_k_mean=-3.125)
    raw = torch.tensor(
        [[-200.0, -200.0], [0.0, 0.0], [200.0, 200.0]], requires_grad=True
    )

    space.to_design(raw).sum().backward()

    assert torch.isfinite(raw.grad).all()


def test_td_project_output_is_contained():
    # Regression: project() clamped R_0 to the closed [0, 100] while contains()
    # requires the open (0, 100), so a projected design failed validation --
    # BEDEnv projects and then validates, so this raised on every step whose
    # design fell outside the domain.
    space = TDDesignSpace()
    outside = torch.tensor([[5.0, -20.0], [1.0, 500.0], [-3.0, 100.0]])

    projected = space.project(outside)

    assert space.contains(projected)


def test_td_to_design_round_trips():
    space = TDDesignSpace(log_k_mean=-3.0)
    raw = torch.randn(16, 2)

    assert torch.allclose(space.to_raw(space.to_design(raw)), raw, atol=1e-4)


# ---------------------------------------------------------------------------
# BoxDesignSpace tanh squash
# ---------------------------------------------------------------------------


def test_box_to_design_corrects_log_prob_for_tanh_affine_jacobian():
    space = BoxDesignSpace(
        torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 7.0]), event_shape=(2,)
    )
    raw = torch.randn(8, 2)
    raw_log_prob = torch.randn(8)

    design, log_prob = space.to_design(raw, raw_log_prob)

    assert torch.allclose(design, space.to_design(raw))
    assert torch.allclose(log_prob, raw_log_prob - space._log_det_jacobian(raw))


def test_box_to_design_lands_in_the_box_and_round_trips():
    space = BoxDesignSpace(
        torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 7.0]), event_shape=(2,)
    )
    raw = torch.randn(64, 2)

    design = space.to_design(raw)

    assert space.contains(design)
    assert torch.allclose(space.to_raw(design), raw, atol=1e-4)


def test_box_reparametrisation_requires_finite_positive_width():
    degenerate = BoxDesignSpace(1.0, 1.0, event_shape=(2,))

    # A zero-width box is still a legitimate space for contains/project.
    assert degenerate.contains(torch.ones(2))
    with pytest.raises(ValueError, match="finite bounds"):
        degenerate.to_design(torch.zeros(2))
