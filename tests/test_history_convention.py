"""Every history-consuming net shares the ``prepare_history`` convention.

Histories are ``[*batch_shape, T, *event_shape]`` with an optional
``[*batch_shape, T]`` boolean mask, and outputs keep ``batch_shape``.
"""

from dataclasses import fields

import pytest
import torch

from testbed.design_nets import (
    AlineModel,
    DeepSetVariationalPosteriorNet,
    RLBOEDDiscreteQNetwork,
    RLBOEDQNetwork,
)

BATCH_SHAPE = (2, 3)
T = 4
EVENT_Y = (1,)
EVENT_DESIGN = (2,)
EVENT_THETA = (2, 2)


def _history(batch_shape=BATCH_SHAPE):
    torch.manual_seed(0)
    y = torch.randn(*batch_shape, T, *EVENT_Y)
    designs = torch.randn(*batch_shape, T, *EVENT_DESIGN)
    mask = torch.rand(*batch_shape, T) > 0.3
    return y, designs, mask


def _flat(x, n_batch_dims=len(BATCH_SHAPE)):
    return x.reshape(-1, *x.shape[n_batch_dims:])


def _critic_kwargs():
    return {
        "event_shape_y": EVENT_Y,
        "event_shape_design": EVENT_DESIGN,
        "design_low": -1.0,
        "design_high": 1.0,
        "outcome_low": -3.0,
        "outcome_high": 3.0,
        "dim_h": 16,
        "dim_emb": 8,
    }


@pytest.mark.parametrize("discrete", [False, True])
def test_critics_accept_leading_batch_dims(discrete):
    y, designs, mask = _history()
    if discrete:
        critic = RLBOEDDiscreteQNetwork(n_designs=5, **_critic_kwargs())
        candidate = ()
    else:
        critic = RLBOEDQNetwork(**_critic_kwargs())
        candidate = (torch.rand(*BATCH_SHAPE, *EVENT_DESIGN),)

    q = critic(y, designs, *candidate, mask=mask)
    flat = critic(_flat(y), _flat(designs), *map(_flat, candidate), mask=_flat(mask))

    assert q.shape == (*BATCH_SHAPE, 5 if discrete else 1)
    torch.testing.assert_close(q.flatten(0, 1), flat)


def test_critic_rejects_mismatched_design_batch():
    critic = RLBOEDQNetwork(**_critic_kwargs())
    y, designs, mask = _history()
    with pytest.raises(ValueError, match="design must have shape"):
        critic(y, designs, torch.zeros(6, *EVENT_DESIGN), mask=mask)


def test_posterior_accepts_leading_batch_dims():
    net = DeepSetVariationalPosteriorNet(
        EVENT_Y, EVENT_DESIGN, EVENT_THETA, encoder_dim_h=[16], dim_emb=8
    )
    y, designs, mask = _history()
    theta = torch.randn(*BATCH_SHAPE, *EVENT_THETA)

    log_prob = net.log_prob(theta, y, designs, mask=mask)
    flat = net.log_prob(_flat(theta), _flat(y), _flat(designs), mask=_flat(mask))
    samples = net.sample(y, designs, sample_shape=7, mask=mask)

    assert log_prob.shape == BATCH_SHAPE
    torch.testing.assert_close(log_prob, flat.reshape(BATCH_SHAPE))
    assert samples.shape == (7, *BATCH_SHAPE, *EVENT_THETA)


def _aline():
    torch.manual_seed(0)
    return AlineModel(
        event_shape_y=EVENT_Y,
        event_shape_theta=EVENT_THETA,
        event_shape_design=EVENT_DESIGN,
        embedding_dim=8,
        feedforward_dim=16,
        n_heads=2,
        n_layers=1,
        n_components=3,
    ).eval()


def _assert_outputs_close(actual, expected, *, flatten_batch=False):
    for field in fields(actual):
        value = getattr(actual, field.name)
        if flatten_batch:
            value = value.flatten(0, 1)
        torch.testing.assert_close(value, getattr(expected, field.name))


def test_aline_accepts_leading_batch_dims():
    model = _aline()
    y, designs, _ = _history()
    candidates = torch.randn(*BATCH_SHAPE, 5, *EVENT_DESIGN)
    available = torch.rand(*BATCH_SHAPE, 5) > 0.2
    available[..., 0] = True

    output = model(y, designs, candidates, available_mask=available)
    flat = model(
        _flat(y), _flat(designs), _flat(candidates), available_mask=_flat(available)
    )

    assert output.acquisition_logits.shape == (*BATCH_SHAPE, 5)
    assert output.mixture_means.shape == (*BATCH_SHAPE, 4, 3)
    _assert_outputs_close(output, flat, flatten_batch=True)


@pytest.mark.parametrize("n_valid", [0, 2])
def test_aline_ignores_masked_context(n_valid):
    model = _aline()
    y, designs, _ = _history(batch_shape=(2,))
    candidates = torch.randn(2, 5, *EVENT_DESIGN)
    mask = torch.zeros(2, T, dtype=torch.bool)
    mask[:, :n_valid] = True

    padded = model(y, designs, candidates, mask=mask)
    unpadded = model(y[:, :n_valid], designs[:, :n_valid], candidates)

    _assert_outputs_close(padded, unpadded)
