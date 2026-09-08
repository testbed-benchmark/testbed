import pytest
import torch

from testbed.core.util import (
    broadcast_and_flatten,
    matches_event_shape,
    normal_log_prob,
    sort_by_norm,
)


def test_matches_event_shape_ignores_leading_batch_dims():
    t = torch.randn(4, 6, 3, 2)
    assert matches_event_shape(t, (3, 2))
    assert matches_event_shape(t, (2,))
    assert not matches_event_shape(t, (6, 2))


def test_matches_event_shape_edge_cases():
    # An empty event shape matches anything: the whole tensor is batch.
    assert matches_event_shape(torch.randn(4, 2), ())
    assert matches_event_shape(torch.tensor(1.0), ())

    # Too few dims to hold the event shape at all.
    assert not matches_event_shape(torch.randn(2), (3, 2))
    assert not matches_event_shape(torch.tensor(1.0), (2,))

    # A size-1 trailing dim is not a match, even though it would broadcast.
    assert not matches_event_shape(torch.randn(5, 1), (2,))


def test_sort_by_norm_no_batchdim():
    input = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0]])

    expected = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])

    output = sort_by_norm(input)

    assert torch.equal(output, expected)
    assert output.shape == (3, 3)


def test_sort_by_norm_one_batch():
    input = torch.tensor([[[1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0]]])

    expected = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]])

    output = sort_by_norm(input)

    assert torch.equal(output, expected)
    assert output.shape == (1, 3, 3)


def test_sort_by_norm_many_batch():
    input = torch.randn((10, 3, 4))
    output = sort_by_norm(input)
    assert output.shape == (10, 3, 4)


def test_broadcast_and_flatten_matches_expected_event_shape():
    t = torch.randn(5, 3, 2)
    (flat,), batch_shape = broadcast_and_flatten((t, (3, 2)))
    assert batch_shape == (5,)
    assert flat.shape == (5, 3, 2)


def test_broadcast_and_flatten_broadcasts_batch_dims():
    a = torch.randn(4, 1, 2)
    b = torch.randn(1, 6, 2)
    (a_flat, b_flat), batch_shape = broadcast_and_flatten((a, (2,)), (b, (2,)))
    assert batch_shape == (4, 6)
    assert a_flat.shape == (24, 2)
    assert b_flat.shape == (24, 2)


def test_broadcast_and_flatten_rejects_wrong_trailing_shape():
    # Regression test: a tensor whose last dim is 1 instead of the expected
    # event dim used to silently pass through, since a size-1 trailing dim
    # broadcasts against any size -- e.g. theta=[b, 1] against
    # event_shape_theta=(2,) got expanded to [b, 2] by duplicating the single
    # value, corrupting the two "components" of theta instead of raising.
    t = torch.randn(5, 1)
    with pytest.raises(ValueError, match="trailing"):
        broadcast_and_flatten((t, (2,)))


def test_broadcast_and_flatten_rejects_too_few_dims():
    t = torch.randn(5)
    with pytest.raises(ValueError, match="trailing"):
        broadcast_and_flatten((t, (2,)))


def test_normal_log_prob_matches_reference():
    y = torch.randn(64, 3)
    loc = torch.randn(64, 3)
    scale = torch.rand(64, 3) + 0.1

    expected = torch.distributions.Normal(loc, scale).log_prob(y)
    assert torch.allclose(normal_log_prob(y, loc, scale), expected, atol=1e-6)


def test_normal_log_prob_backward_finite_at_tiny_scale():
    # Regression test: forming (y - loc)^2 / var made autograd materialise
    # (y - loc)^2 / scale^4 in the backward, which overflows float32 here even
    # though the true gradient, ~4e29, is representable. The z = (y - loc) /
    # scale parametrisation keeps the largest intermediate cubic in 1 / scale.
    delta, scale_value = 20.0, 1e-9
    loc = torch.zeros(1, requires_grad=True)
    scale = torch.full((1,), scale_value, requires_grad=True)

    logp = normal_log_prob(torch.full((1,), delta), loc, scale)
    logp.backward()

    z = delta / scale_value
    assert torch.isfinite(logp).all()
    assert torch.isfinite(scale.grad).all()
    assert torch.isfinite(loc.grad).all()
    # d/dscale = (z^2 - 1) / scale, d/dloc = z / scale.
    assert scale.grad.item() == pytest.approx((z**2 - 1) / scale_value, rel=1e-5)
    assert loc.grad.item() == pytest.approx(z / scale_value, rel=1e-5)


def test_normal_log_prob_backward_finite_at_large_scale():
    # The mirror case: log(scale) as the normaliser stops var = scale^2 from
    # overflowing, where log(2 * pi * var) would have.
    scale = torch.full((1,), 1e30, requires_grad=True)

    logp = normal_log_prob(torch.ones(1), torch.zeros(1), scale)
    logp.backward()

    assert torch.isfinite(logp).all()
    assert torch.isfinite(scale.grad).all()
