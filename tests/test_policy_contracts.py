"""Shared input/output contracts for history-based policy networks."""

import pytest
import torch
from torch import nn

from testbed.core.design_spaces import BoxDesignSpace, FiniteDesignSpace
from testbed.design_nets import (
    CategoricalHead,
    CategoricalPolicyNet,
    DeepSetEncoder,
    DeepSetPolicyNet,
    GaussianHead,
    GaussianPolicyNet,
)

EVENT_SHAPE = (2, 2)
BATCH_SHAPE = (2, 3)


class RecordingGaussianActor(nn.Module):
    """Minimal custom actor used to assert the flattened actor contract."""

    def __init__(self, dim_design):
        super().__init__()
        self.dim_design = dim_design
        self.last_inputs = None
        self.register_buffer("mean", torch.arange(dim_design, dtype=torch.float32))

    def forward(self, y_history, design_history, *, mask=None):
        self.last_inputs = (y_history, design_history, mask)
        return self.mean.expand(design_history.shape[0], -1), torch.zeros_like(
            design_history[:, 0]
        )


class RecordingCategoricalActor(nn.Module):
    """Minimal custom categorical actor with no implementation metadata."""

    def __init__(self, n_designs):
        super().__init__()
        self.n_designs = n_designs
        self.last_inputs = None

    def forward(self, y_history, design_history, *, mask=None):
        self.last_inputs = (y_history, design_history, mask)
        return torch.arange(
            self.n_designs, device=design_history.device, dtype=design_history.dtype
        ).expand(design_history.shape[0], -1)


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
    assert "null_enc" in dict(encoder.named_parameters())


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


def test_gaussian_sampling_and_reparameterized_sampling_contracts():
    policy = make_gaussian_policy()
    y_history, design_history = make_histories()

    sampled = policy.propose(y_history, design_history)
    reparameterized = policy.propose(
        y_history, design_history, reparameterize_policy=True
    )

    assert not sampled.raw_design.requires_grad
    assert reparameterized.raw_design.requires_grad
    assert reparameterized.log_prob.requires_grad


@pytest.mark.parametrize(
    ("make_policy", "actor"),
    [
        (make_gaussian_policy, RecordingGaussianActor(4)),
        (make_categorical_policy, RecordingCategoricalActor(5)),
    ],
)
def test_custom_actors_receive_flattened_histories_and_propagated_dtype(
    make_policy, actor
):
    policy = make_policy(actor=actor).double()
    y_history, design_history = make_histories()
    mask = torch.tensor([[True, False, True, False]]).expand(*BATCH_SHAPE, -1)

    result = policy(y_history.float(), design_history.float(), mask=mask)
    y_flat, design_flat, mask_flat = policy.actor.last_inputs

    assert result.dtype == torch.float64
    assert result.device == policy.design_low.device
    assert y_flat.shape == (BATCH_SHAPE[0] * BATCH_SHAPE[1], 4, 4)
    assert design_flat.shape == (BATCH_SHAPE[0] * BATCH_SHAPE[1], 4, 4)
    assert y_flat.dtype == design_flat.dtype == torch.float64
    assert mask_flat.shape == (BATCH_SHAPE[0] * BATCH_SHAPE[1], 4)
    if isinstance(actor, RecordingCategoricalActor):
        assert not hasattr(policy.actor, "head")


@pytest.mark.parametrize(
    ("make_policy", "actor"),
    [
        (make_gaussian_policy, RecordingGaussianActor(4)),
        (make_categorical_policy, RecordingCategoricalActor(5)),
    ],
)
def test_custom_actors_infer_scalar_outcome_events(make_policy, actor):
    policy = make_policy(actor=actor)
    design_history = torch.randn(*BATCH_SHAPE, 4, *EVENT_SHAPE)

    policy(torch.randn(*BATCH_SHAPE, 4), design_history)

    y_flat, _, _ = policy.actor.last_inputs
    assert y_flat.shape == (BATCH_SHAPE[0] * BATCH_SHAPE[1], 4, 1)


def test_gaussian_policy_flattens_multidimensional_outcome_bounds():
    actor = RecordingGaussianActor(4)
    policy = make_gaussian_policy(actor=actor)
    outcome_low = torch.tensor([[0.0, 10.0], [20.0, 30.0]])
    outcome_high = outcome_low + 10.0
    policy = GaussianPolicyNet(
        policy.design_space,
        actor=actor,
        outcome_bounds=(outcome_low, outcome_high),
    )
    y_history = torch.stack((outcome_low, outcome_high)).reshape(1, 2, 2, 2)
    design_history = torch.randn(1, 2, *EVENT_SHAPE)

    policy(y_history, design_history)

    y_flat, _, _ = actor.last_inputs
    torch.testing.assert_close(
        y_flat,
        torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]]),
    )


@pytest.mark.parametrize(
    "make_policy", [make_deepset_policy, make_gaussian_policy, make_categorical_policy]
)
@pytest.mark.parametrize(
    ("y_shape", "design_shape", "mask_shape"),
    [
        ((2, 3, 4, 2), (2, 3, 4, 2, 2), (2, 3, 4)),
        ((2, 3, 4, 2, 2), (2, 3, 4, 2), (2, 3, 4)),
        ((2, 3, 4, 2, 2), (2, 3, 4, 2, 2), (2, 3, 3)),
    ],
)
def test_policies_reject_invalid_history_and_mask_shapes(
    make_policy, y_shape, design_shape, mask_shape
):
    policy = make_policy()

    with pytest.raises(ValueError):
        policy(
            torch.randn(*y_shape),
            torch.randn(*design_shape),
            mask=torch.ones(*mask_shape, dtype=torch.bool),
        )
