"""Shared input/output contracts for history-based policy networks."""

import pytest
import torch

from testbed.core.design_spaces import BoxDesignSpace, FiniteDesignSpace
from testbed.design_nets import (
    AttentivePolicyNet,
    CategoricalHead,
    CategoricalPolicyNet,
    DeepSetEncoder,
    DeepSetPolicyNet,
    GatedDeepSetEncoder,
    GatedDeepSetPolicyNet,
    GaussianHead,
    GaussianPolicyNet,
    TimeEmbeddedDeepSetEncoder,
)

EVENT_SHAPE = (2, 2)
BATCH_SHAPE = (2, 3)

# The gated encoder mixes two heads with a scalar outcome, so it needs its own
# shapes rather than the shared EVENT_SHAPE.
GATED_SHAPE_Y = (1,)
GATED_SHAPE_DESIGN = (2,)


def make_gated_encoder(dim_emb=4):
    return GatedDeepSetEncoder(
        GATED_SHAPE_Y, GATED_SHAPE_DESIGN, dim_emb=dim_emb, dim_h=4
    )


def make_gated_histories(batch_shape=BATCH_SHAPE, history_length=4):
    return (
        torch.randint(0, 2, (*batch_shape, history_length, *GATED_SHAPE_Y)).float(),
        torch.randn(*batch_shape, history_length, *GATED_SHAPE_DESIGN),
    )


def make_histories(batch_shape=BATCH_SHAPE, history_length=4):
    return (
        torch.randn(*batch_shape, history_length, *EVENT_SHAPE),
        torch.randn(*batch_shape, history_length, *EVENT_SHAPE),
    )


def make_deepset_policy():
    return DeepSetPolicyNet(
        EVENT_SHAPE,
        EVENT_SHAPE,
        event_shape_out=EVENT_SHAPE,
        dim_emb=7,
        encoder_dim_h=8,
        decoder_dim_h=8,
    )


def make_gaussian_policy(actor=None):
    space = BoxDesignSpace(-1.0, 1.0, event_shape=EVENT_SHAPE)
    if actor is not None:
        return GaussianPolicyNet(space, actor=actor)
    encoder = DeepSetEncoder(EVENT_SHAPE, EVENT_SHAPE, dim_emb=7, dim_h=8)
    return GaussianPolicyNet(space, encoder=encoder, head=GaussianHead(7, 4, dim_h=8))


def make_categorical_policy(actor=None):
    values = torch.arange(20, dtype=torch.float32).reshape(5, *EVENT_SHAPE)
    space = FiniteDesignSpace(values)
    if actor is not None:
        return CategoricalPolicyNet(space, actor=actor)
    encoder = DeepSetEncoder(EVENT_SHAPE, EVENT_SHAPE, dim_emb=7, dim_h=8)
    return CategoricalPolicyNet(
        space, encoder=encoder, head=CategoricalHead(7, 5, dim_h=8)
    )


@pytest.mark.parametrize(
    "make_policy", [make_deepset_policy, make_gaussian_policy, make_categorical_policy]
)
def test_policies_preserve_arbitrary_batch_and_event_shapes(make_policy):
    policy = make_policy()
    y_history, design_history = make_histories()
    mask = torch.tensor([[True, False, True, False]]).expand(*BATCH_SHAPE, -1)

    result = policy(y_history, design_history, mask=mask)

    assert result.shape == (*BATCH_SHAPE, *EVENT_SHAPE)


@pytest.mark.parametrize(
    "make_policy", [make_deepset_policy, make_gaussian_policy, make_categorical_policy]
)
def test_policies_support_scalar_batches_and_empty_histories(make_policy):
    policy = make_policy()
    y_history, design_history = make_histories(batch_shape=(), history_length=0)

    result = policy(y_history, design_history)

    assert result.shape == EVENT_SHAPE


@pytest.mark.parametrize(
    "make_policy", [make_deepset_policy, make_gaussian_policy, make_categorical_policy]
)
def test_policies_treat_fully_masked_histories_as_null_histories(make_policy):
    policy = make_policy()
    y_history, design_history = make_histories()
    fully_masked = policy(
        y_history,
        design_history,
        mask=torch.zeros(*BATCH_SHAPE, 4, dtype=torch.bool),
    )
    empty = policy(
        torch.empty(*BATCH_SHAPE, 0, *EVENT_SHAPE),
        torch.empty(*BATCH_SHAPE, 0, *EVENT_SHAPE),
    )

    torch.testing.assert_close(fully_masked, empty)


def test_deepset_learned_null_history_agrees_for_empty_and_fully_masked_rows():
    encoder = DeepSetEncoder((1,), (1,), dim_emb=4, dim_h=4)
    y_history = torch.randn(2, 3, 1)
    design_history = torch.randn(2, 3, 1)

    empty = encoder(torch.empty(2, 0, 1), torch.empty(2, 0, 1))
    fully_masked = encoder(
        y_history, design_history, mask=torch.zeros(2, 3, dtype=torch.bool)
    )

    torch.testing.assert_close(empty, fully_masked)


def test_deepset_partially_masked_rows_pool_only_valid_tokens():
    encoder = DeepSetEncoder((1,), (1,), dim_emb=4, dim_h=4)
    y_history = torch.randn(2, 3, 1)
    design_history = torch.randn(2, 3, 1)
    mask = torch.tensor([[True, False, True], [False, True, False]])

    masked = encoder(y_history, design_history, mask=mask)
    expected = torch.stack(
        [
            encoder(y_history[0, [0, 2]], design_history[0, [0, 2]]),
            encoder(
                y_history[1, [1]].unsqueeze(0), design_history[1, [1]].unsqueeze(0)
            )[0],
        ]
    )

    torch.testing.assert_close(masked, expected)


def test_gated_deepset_policy_preserves_batch_shapes_and_empty_histories():
    policy = GatedDeepSetPolicyNet(
        GATED_SHAPE_Y,
        GATED_SHAPE_DESIGN,
        event_shape_out=GATED_SHAPE_DESIGN,
        dim_emb=4,
        encoder_dim_h=4,
        decoder_dim_h=4,
    )
    y_history, design_history = make_gated_histories()
    mask = torch.tensor([[True, False, True, False]]).expand(*BATCH_SHAPE, -1)

    assert policy(y_history, design_history, mask=mask).shape == (
        *BATCH_SHAPE,
        *GATED_SHAPE_DESIGN,
    )

    empty_y, empty_design = make_gated_histories(batch_shape=(), history_length=0)
    assert policy(empty_y, empty_design).shape == GATED_SHAPE_DESIGN


def test_gated_deepset_learned_null_history_agrees_for_empty_and_masked_rows():
    encoder = make_gated_encoder()
    y_history, design_history = make_gated_histories(batch_shape=(2,), history_length=3)

    empty = encoder(torch.empty(2, 0, 1), torch.empty(2, 0, 2))
    fully_masked = encoder(
        y_history, design_history, mask=torch.zeros(2, 3, dtype=torch.bool)
    )

    torch.testing.assert_close(empty, fully_masked)


def test_gated_deepset_partially_masked_rows_pool_only_valid_tokens():
    encoder = make_gated_encoder()
    y_history, design_history = make_gated_histories(batch_shape=(1,), history_length=3)
    mask = torch.tensor([[True, False, True]])

    masked = encoder(y_history, design_history, mask=mask)
    expected = encoder(y_history[:, [0, 2]], design_history[:, [0, 2]])

    torch.testing.assert_close(masked, expected)


def test_gated_deepset_is_permutation_invariant():
    encoder = make_gated_encoder()
    y_history, design_history = make_gated_histories(batch_shape=(1,), history_length=3)
    permutation = [2, 0, 1]

    torch.testing.assert_close(
        encoder(y_history, design_history),
        encoder(y_history[:, permutation], design_history[:, permutation]),
    )


def test_gated_deepset_encodes_the_two_outcomes_differently():
    encoder = make_gated_encoder()
    design_history = torch.randn(1, 1, *GATED_SHAPE_DESIGN)

    zero = encoder(torch.zeros(1, 1, 1), design_history)
    one = encoder(torch.ones(1, 1, 1), design_history)

    assert not torch.allclose(zero, one)


def test_gated_deepset_encoder_sees_the_design_alone_with_two_heads():
    encoder = make_gated_encoder(dim_emb=16)
    linear_layers = [
        layer for layer in encoder.encoder_net.net if isinstance(layer, torch.nn.Linear)
    ]

    assert linear_layers[0].in_features == 2  # design only, not [y, design]
    assert linear_layers[-1].out_features == 32  # two dim_emb heads
    assert encoder.dim_token == 3  # tokens still carry y through to the gate


def test_gated_deepset_rejects_non_scalar_outcomes():
    with pytest.raises(ValueError, match="scalar outcome"):
        GatedDeepSetEncoder(EVENT_SHAPE, GATED_SHAPE_DESIGN)


def make_time_embedded_encoder(dim_emb=7, dim_time=None, max_T=8):
    return TimeEmbeddedDeepSetEncoder(
        EVENT_SHAPE, EVENT_SHAPE, dim_emb=dim_emb, dim_h=8, dim_time=dim_time,
        max_T=max_T,
    )


def test_deepset_is_permutation_invariant():
    """The baseline the time-embedded encoder is meant to break."""
    encoder = DeepSetEncoder(EVENT_SHAPE, EVENT_SHAPE, dim_emb=7, dim_h=8)
    y_history, design_history = make_histories(batch_shape=(1,), history_length=3)
    permutation = [2, 0, 1]

    torch.testing.assert_close(
        encoder(y_history, design_history),
        encoder(y_history[:, permutation], design_history[:, permutation]),
    )


def test_time_embedded_deepset_is_not_permutation_invariant():
    """The whole point of the class, and easy to get wrong.

    Adding the embedding to the per-step encoding instead of concatenating it
    onto the token would leave invariance intact, since
    ``sum_t [phi(x_t) + e_t]`` splits into ``sum_t phi(x_t)`` plus a constant
    that does not depend on the ordering.
    """
    encoder = make_time_embedded_encoder()
    y_history, design_history = make_histories(batch_shape=(1,), history_length=3)
    permutation = [2, 0, 1]

    assert not torch.allclose(
        encoder(y_history, design_history),
        encoder(y_history[:, permutation], design_history[:, permutation]),
    )


def test_time_embedded_deepset_token_carries_the_time_embedding():
    encoder = make_time_embedded_encoder(dim_emb=16, dim_time=5)
    linear_layers = [
        layer for layer in encoder.encoder_net.net if isinstance(layer, torch.nn.Linear)
    ]

    # y + design + time, so the time enters inside the per-step nonlinearity.
    assert encoder.dim_token == 4 + 4 + 5
    assert linear_layers[0].in_features == 13
    assert encoder.time_emb.num_embeddings == 9  # max_T + 1


def test_time_embedded_deepset_dim_time_defaults_to_dim_emb():
    encoder = make_time_embedded_encoder(dim_emb=7)

    assert encoder.dim_time == 7


def test_time_embedded_deepset_rejects_histories_longer_than_max_T():
    encoder = make_time_embedded_encoder(max_T=3)
    y_history, design_history = make_histories(batch_shape=(1,), history_length=4)

    with pytest.raises(ValueError, match="exceeds max_T"):
        encoder(y_history, design_history)


def test_attentive_policy_without_time_embedding_is_permutation_invariant():
    """use_time_embedding is the only thing separating these two."""
    torch.manual_seed(0)
    timed = AttentivePolicyNet(
        EVENT_SHAPE, EVENT_SHAPE, event_shape_out=EVENT_SHAPE, dim_emb=8,
        encoder_dim_h=8, num_heads=2, num_blocks=1,
    )
    torch.manual_seed(0)
    untimed = AttentivePolicyNet(
        EVENT_SHAPE, EVENT_SHAPE, event_shape_out=EVENT_SHAPE, dim_emb=8,
        encoder_dim_h=8, num_heads=2, num_blocks=1, use_time_embedding=False,
    )
    y_history, design_history = make_histories(batch_shape=(1,), history_length=3)
    permutation = [2, 0, 1]

    assert not untimed.encoder.use_time_embedding
    torch.testing.assert_close(
        untimed.encoder(y_history, design_history),
        untimed.encoder(y_history[:, permutation], design_history[:, permutation]),
    )
    assert not torch.allclose(
        timed.encoder(y_history, design_history),
        timed.encoder(y_history[:, permutation], design_history[:, permutation]),
    )


@pytest.mark.parametrize("make_policy", [make_gaussian_policy, make_categorical_policy])
def test_stochastic_forward_matches_deterministic_proposal(make_policy):
    policy = make_policy()
    y_history, design_history = make_histories()
    mask = torch.tensor([[True, True, False, False]]).expand(*BATCH_SHAPE, -1)

    deterministic = policy(y_history, design_history, mask=mask)
    proposal = policy.propose(y_history, design_history, mask=mask)

    torch.testing.assert_close(deterministic, proposal.deterministic_raw_design)
    assert proposal.raw_design.shape == (*BATCH_SHAPE, *EVENT_SHAPE)
    assert proposal.log_prob.shape == BATCH_SHAPE
