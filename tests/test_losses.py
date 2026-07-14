import itertools

import pytest
import torch
from torch.nn import functional as F

from testbed.core.util import sort_by_norm
from testbed.design_objectives import (
    LogMSELoss,
    LotDecisionLoss,
    PermutationInvariantMSELoss,
    SortedLogMSELoss,
    SortedMSELoss,
)

LOSS_CLASSES = [
    LogMSELoss,
    SortedMSELoss,
    SortedLogMSELoss,
    PermutationInvariantMSELoss,
]


def test_lot_decision_loss_matches_asymmetric_terminal_cost():
    logits = torch.zeros(2, 1, requires_grad=True)
    theta = torch.tensor([[0.0, 3.0, 0.2], [1.0, 7.0, 0.08]])

    loss = LotDecisionLoss(
        false_accept_cost=4.0,
        false_reject_cost=1.0,
        reduction="none",
    )(logits, theta)

    torch.testing.assert_close(loss, torch.tensor([4.0, 0.0]))
    loss.mean().backward()
    assert logits.grad is not None


def test_lot_decision_loss_rejects_invalid_action_shape():
    with pytest.raises(ValueError, match="one acceptance logit"):
        LotDecisionLoss()(torch.zeros(2, 3), torch.zeros(2, 3))


def _brute_force_permutation_mse(input, target):
    """Reference implementation for a single unbatched [K, D] pair."""
    k = input.shape[0]
    costs = [
        sum((input[i] - target[j]).pow(2).sum().item() for i, j in enumerate(perm))
        for perm in itertools.permutations(range(k))
    ]
    return min(costs)


@pytest.mark.parametrize("loss_cls", LOSS_CLASSES)
def test_rejects_invalid_reduction(loss_cls):
    with pytest.raises(ValueError, match="Unsupported reduction"):
        loss_cls(reduction="max")


@pytest.mark.parametrize("loss_cls", LOSS_CLASSES)
def test_rejects_mismatched_shapes(loss_cls):
    loss_fn = loss_cls()
    with pytest.raises(ValueError, match="same shape"):
        loss_fn(torch.randn(2, 3, 4), torch.randn(2, 3, 5))


@pytest.mark.parametrize("loss_cls", LOSS_CLASSES)
def test_gradient_flows_through_loss(loss_cls):
    input = torch.randn(2, 3, 2, requires_grad=True)
    target = torch.randn(2, 3, 2)

    loss_cls()(input, target).backward()

    assert input.grad is not None
    assert torch.isfinite(input.grad).all()


def test_log_mse_loss_shapes_across_reductions_and_batch_dims():
    for shape in [(3,), (4, 3), (2, 4, 3)]:
        input = torch.randn(shape)
        target = torch.randn(shape)

        assert LogMSELoss(reduction="none")(input, target).shape == shape
        assert LogMSELoss(reduction="mean")(input, target).shape == ()
        assert LogMSELoss(reduction="sum")(input, target).shape == ()


def test_log_mse_loss_matches_hand_computed_value():
    input = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 2.0, 5.0])  # squared errors: 1, 0, 4
    epsilon = 1e-6
    elementwise_mse = torch.tensor([1.0, 0.0, 4.0])
    expected = (elementwise_mse + torch.log(elementwise_mse + epsilon)).mean()

    loss = LogMSELoss(epsilon=epsilon)(input, target)

    torch.testing.assert_close(loss, expected)


def test_log_mse_loss_zero_error_equals_log_epsilon():
    x = torch.randn(5, 2)
    epsilon = 1e-4
    loss = LogMSELoss(epsilon=epsilon)(x, x.clone())

    torch.testing.assert_close(loss, torch.log(torch.tensor(epsilon)))


def test_log_mse_loss_applies_log_before_reduction():
    """[Warning] in the source: the log is applied element-wise, per the ActionBED paper."""
    torch.manual_seed(4)
    input = torch.randn(4, 3)
    target = torch.randn(4, 3)
    epsilon = 1e-6

    loss = LogMSELoss(epsilon=epsilon, reduction="mean")(input, target)

    elementwise_mse = F.mse_loss(input, target, reduction="none")
    expected = (elementwise_mse + torch.log(elementwise_mse + epsilon)).mean()
    torch.testing.assert_close(loss, expected)

    # Applying the log after reduction instead (the pre-fix behavior) gives a
    # different value; guards against regressing to that formula.
    reduced_mse = elementwise_mse.mean()
    log_after_reduction = reduced_mse + torch.log(reduced_mse + epsilon)
    assert not torch.allclose(loss, log_after_reduction)


def test_log_mse_loss_none_reduction_is_consistent_with_mean_and_sum():
    # The elementwise log means mean/sum of the 'none' output exactly reproduce
    # the 'mean'/'sum' reductions (same design as SortedLogMSELoss).
    torch.manual_seed(11)
    input = torch.randn(4, 3)
    target = torch.randn(4, 3)

    none_loss = LogMSELoss(reduction="none")(input, target)
    mean_loss = LogMSELoss(reduction="mean")(input, target)
    sum_loss = LogMSELoss(reduction="sum")(input, target)

    torch.testing.assert_close(none_loss.mean(), mean_loss)
    torch.testing.assert_close(none_loss.sum(), sum_loss)


def test_sorted_mse_loss_supports_unbatched_and_batched_inputs():
    for shape in [(3, 2), (4, 3, 2)]:
        input = torch.randn(shape)
        target = torch.randn(shape)

        assert SortedMSELoss(reduction="none")(input, target).shape == shape
        assert SortedMSELoss(reduction="mean")(input, target).shape == ()
        assert SortedMSELoss(reduction="sum")(input, target).shape == ()


def test_sorted_mse_loss_sorts_target_but_not_input_by_default():
    # target sorted by norm is exactly [[0,0,1],[0,1,2],[1,2,3]]; input is given
    # already in that order, so the (unsorted) input should match the sorted target.
    input = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    target = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0]])

    loss = SortedMSELoss()(input, target)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_sorted_mse_loss_is_invariant_to_target_object_order():
    torch.manual_seed(0)
    input = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 3)
    permutation = torch.tensor([2, 0, 3, 1])

    loss_fn = SortedMSELoss()
    loss = loss_fn(input, target)
    permuted_loss = loss_fn(input, target[:, permutation])

    torch.testing.assert_close(loss, permuted_loss)


def test_sorted_mse_loss_default_is_sensitive_to_input_object_order():
    torch.manual_seed(2)
    input = torch.randn(4, 3)
    target = torch.randn(4, 3)
    permutation = torch.tensor([3, 1, 0, 2])

    loss_fn = SortedMSELoss()  # sort_input=False
    loss = loss_fn(input, target)
    permuted_loss = loss_fn(input[permutation], target)

    assert not torch.allclose(loss, permuted_loss)


def test_sorted_mse_loss_sort_input_true_is_invariant_to_input_object_order():
    torch.manual_seed(1)
    input = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 3)
    permutation = torch.tensor([3, 1, 0, 2])

    loss_fn = SortedMSELoss(sort_input=True)
    loss = loss_fn(input, target)
    permuted_loss = loss_fn(input[:, permutation], target)

    torch.testing.assert_close(loss, permuted_loss)


def test_sorted_mse_loss_matches_manual_computation():
    torch.manual_seed(3)
    input = torch.randn(5, 3, 2)
    target = torch.randn(5, 3, 2)

    loss = SortedMSELoss(sort_input=True, reduction="sum")(input, target)
    expected = F.mse_loss(sort_by_norm(input), sort_by_norm(target), reduction="sum")

    torch.testing.assert_close(loss, expected)


def test_sorted_mse_loss_gradient_flows_when_sorting_input():
    input = torch.randn(2, 3, 2, requires_grad=True)
    target = torch.randn(2, 3, 2)

    SortedMSELoss(sort_input=True)(input, target).backward()

    assert input.grad is not None
    assert torch.isfinite(input.grad).all()


def test_sorted_log_mse_loss_supports_unbatched_and_batched_inputs():
    for shape in [(3, 2), (4, 3, 2)]:
        input = torch.randn(shape)
        target = torch.randn(shape)

        assert SortedLogMSELoss(reduction="none")(input, target).shape == shape
        assert SortedLogMSELoss(reduction="mean")(input, target).shape == ()
        assert SortedLogMSELoss(reduction="sum")(input, target).shape == ()


def test_sorted_log_mse_loss_applies_log_before_reduction():
    """[Warning] in the source: the log is applied element-wise before reduction."""
    torch.manual_seed(4)
    input = torch.randn(4, 3, 2)
    target = torch.randn(4, 3, 2)
    epsilon = 1e-6

    loss = SortedLogMSELoss(epsilon=epsilon, reduction="mean")(input, target)

    elementwise_mse = F.mse_loss(input, sort_by_norm(target), reduction="none")
    expected = (elementwise_mse + torch.log(elementwise_mse + epsilon)).mean()
    torch.testing.assert_close(loss, expected)

    reduced_mse = elementwise_mse.mean()
    log_after_reduction = reduced_mse + torch.log(reduced_mse + epsilon)
    assert not torch.allclose(loss, log_after_reduction)


def test_sorted_log_mse_loss_none_reduction_is_consistent_with_mean_and_sum():
    # The elementwise log means mean/sum of the 'none' output exactly reproduce
    # the 'mean'/'sum' reductions (LogMSELoss has the same property).
    torch.manual_seed(12)
    input = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 3)

    none_loss = SortedLogMSELoss(reduction="none")(input, target)
    mean_loss = SortedLogMSELoss(reduction="mean")(input, target)
    sum_loss = SortedLogMSELoss(reduction="sum")(input, target)

    torch.testing.assert_close(none_loss.mean(), mean_loss)
    torch.testing.assert_close(none_loss.sum(), sum_loss)


def test_sorted_log_mse_loss_is_invariant_to_target_object_order():
    torch.manual_seed(6)
    input = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 3)
    permutation = torch.tensor([2, 0, 3, 1])

    loss_fn = SortedLogMSELoss()
    loss = loss_fn(input, target)
    permuted_loss = loss_fn(input, target[:, permutation])

    torch.testing.assert_close(loss, permuted_loss)


def test_permutation_invariant_loss_rejects_1d_input():
    loss_fn = PermutationInvariantMSELoss()
    with pytest.raises(ValueError, match="dim_object"):
        loss_fn(torch.randn(4), torch.randn(4))


def test_permutation_invariant_loss_shapes_across_batch_dims_and_reductions():
    shapes_and_none_shapes = [
        ((3, 2), ()),
        ((5, 3, 2), (5,)),
        ((5, 6, 3, 2), (5, 6)),
    ]

    for shape, none_shape in shapes_and_none_shapes:
        input = torch.randn(shape)
        target = torch.randn(shape)

        assert (
            PermutationInvariantMSELoss(reduction="none")(input, target).shape
            == none_shape
        )
        assert PermutationInvariantMSELoss(reduction="mean")(input, target).shape == ()
        assert PermutationInvariantMSELoss(reduction="sum")(input, target).shape == ()


def test_permutation_invariant_loss_is_invariant_to_target_permutation():
    torch.manual_seed(8)
    input = torch.randn(3, 5, 4)
    target = torch.randn(3, 5, 4)
    permutation = torch.tensor([2, 0, 4, 1, 3])

    loss_fn = PermutationInvariantMSELoss(reduction="none")
    loss = loss_fn(input, target)
    permuted_loss = loss_fn(input, target[:, permutation])

    torch.testing.assert_close(loss, permuted_loss)


def test_permutation_invariant_loss_is_invariant_to_input_permutation():
    torch.manual_seed(9)
    input = torch.randn(3, 5, 4)
    target = torch.randn(3, 5, 4)
    permutation = torch.tensor([2, 0, 4, 1, 3])

    loss_fn = PermutationInvariantMSELoss(reduction="none")
    loss = loss_fn(input, target)
    permuted_loss = loss_fn(input[:, permutation], target)

    torch.testing.assert_close(loss, permuted_loss)


def test_permutation_invariant_loss_is_zero_for_permuted_copy():
    torch.manual_seed(10)
    target = torch.randn(6, 3)  # unbatched: n_objects=6, dim_object=3
    permutation = torch.tensor([5, 2, 0, 4, 1, 3])
    input = target[permutation]

    loss = PermutationInvariantMSELoss(reduction="none")(input, target)

    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_permutation_invariant_loss_matches_brute_force_reference():
    torch.manual_seed(7)
    loss_fn = PermutationInvariantMSELoss(reduction="none")

    for k in (1, 2, 3, 4):
        input = torch.randn(5, k, 3)
        target = torch.randn(5, k, 3)

        actual = loss_fn(input, target)
        expected = torch.tensor(
            [_brute_force_permutation_mse(input[b], target[b]) for b in range(5)]
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_permutation_invariant_loss_reduction_semantics():
    # B=2, K=2, D=1, hand-computed optimal assignment costs per sample: 2 and 50.
    input = torch.tensor([[[0.0], [10.0]], [[0.0], [0.0]]])
    target = torch.tensor([[[1.0], [9.0]], [[5.0], [5.0]]])

    none_loss = PermutationInvariantMSELoss(reduction="none")(input, target)
    mean_loss = PermutationInvariantMSELoss(reduction="mean")(input, target)
    sum_loss = PermutationInvariantMSELoss(reduction="sum")(input, target)

    torch.testing.assert_close(none_loss, torch.tensor([2.0, 50.0]))
    torch.testing.assert_close(mean_loss, torch.tensor(13.0))  # mean(none / (D*K))
    torch.testing.assert_close(
        sum_loss, torch.tensor(52.0)
    )  # sum(none), no D*K scaling
