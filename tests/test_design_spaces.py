import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace, DefaultDesignSpace, DesignSpace
from testbed.design_nets.stochastic import TanhGaussianPolicyNet
from testbed.tasks.hyperbolic_temporal_discounting import HTDDesignSpace


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

    def log_det_jacobian(self, raw_design):
        return self._sum_event_dims(raw_design)


# ---------------------------------------------------------------------------
# DesignSpace base contract
# ---------------------------------------------------------------------------


def test_subclass_only_needs_contains():
    space = PositiveSpace(event_shape=(2,))

    assert space.contains(torch.tensor([1.0, 2.0]))
    assert not space.contains(torch.tensor([1.0, -2.0]))

    # Everything else falls back to a working default.
    design = torch.tensor([1.0, -2.0])
    assert torch.equal(space.project(design), design)
    assert torch.equal(space.to_design(design), design)
    assert torch.equal(space.to_raw(design), design)


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


def test_log_det_jacobian_defaults_to_zeros_with_batch_shape():
    space = DefaultDesignSpace(event_shape=(2, 3))
    raw = torch.randn(4, 5, 2, 3)

    log_det = space.log_det_jacobian(raw)

    assert log_det.shape == (4, 5)
    assert torch.all(log_det == 0)


def test_transform_without_log_det_jacobian_raises_rather_than_silently_wrong():
    # A subclass that reparametrises but forgets log_det_jacobian would
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
    with pytest.raises(NotImplementedError, match="log_det_jacobian"):
        space.to_design(raw, torch.zeros(4))


def test_exp_space_log_det_matches_autograd():
    space = ExpSpace(event_shape=(2,))
    raw = torch.tensor([0.3, -1.2])

    assert torch.allclose(space.log_det_jacobian(raw), _autograd_log_det(space, raw))


def test_box_log_det_matches_autograd():
    space = BoxDesignSpace(
        torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 7.0]), event_shape=(2,)
    )
    raw = torch.tensor([0.4, -0.9])

    assert torch.allclose(space.log_det_jacobian(raw), _autograd_log_det(space, raw))


def test_htd_log_det_matches_autograd():
    space = HTDDesignSpace(log_k_mean=-3.0)
    raw = torch.tensor([0.7, -0.4])

    assert torch.allclose(space.log_det_jacobian(raw), _autograd_log_det(space, raw))


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
# HTDDesignSpace
# ---------------------------------------------------------------------------


def test_htd_design_space_inherits_the_base_shape_check():
    space = HTDDesignSpace()

    assert space.event_shape == torch.Size([2])
    assert space.contains(torch.tensor([[1.0, 50.0], [2.0, 25.0]]))
    assert not space.contains(torch.tensor([1.0, 50.0, 25.0]))


def test_htd_design_space_domain():
    space = HTDDesignSpace()

    assert not space.contains(torch.tensor([0.0, 50.0]))  # D_1 > 0 is strict
    assert not space.contains(torch.tensor([1.0, 0.0]))  # R_0 > 0 is strict
    assert not space.contains(torch.tensor([1.0, 100.0]))  # R_0 < 100 is strict


def test_htd_sample_lands_in_the_space():
    torch.manual_seed(0)
    space = HTDDesignSpace(log_k_mean=-3.0)

    assert space.contains(space.sample((256,)))


def test_htd_to_design_round_trips():
    space = HTDDesignSpace(log_k_mean=-3.0)
    raw = torch.randn(16, 2)

    assert torch.allclose(space.to_raw(space.to_design(raw)), raw, atol=1e-4)


# ---------------------------------------------------------------------------
# BoxDesignSpace squash agrees with the copy inside TanhGaussianPolicyNet
# ---------------------------------------------------------------------------


def test_box_squash_matches_tanh_gaussian_policy_net():
    # The net predates BoxDesignSpace.to_design and still owns its own copy of
    # this transform; the two must not drift apart.
    space = BoxDesignSpace(
        torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 7.0]), event_shape=(2,)
    )
    policy = TanhGaussianPolicyNet(space)
    raw = torch.randn(8, 2)

    assert torch.allclose(space.to_design(raw), policy._squash(raw))
    assert torch.allclose(
        space.log_det_jacobian(raw), policy._squash_log_det(raw).sum(-1)
    )


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
