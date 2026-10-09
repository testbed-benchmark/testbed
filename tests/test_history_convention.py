"""Every history-consuming net shares the ``prepare_history`` convention.

Histories are ``[*batch_shape, T, *event_shape]`` with an optional
``[*batch_shape, T]`` boolean mask, and outputs keep ``batch_shape``.
"""

import pytest
import torch

from testbed.design_nets import DeepSetVariationalPosteriorNet
from testbed.method_nets import AlineModel, RLBOEDDiscreteQNetwork, RLBOEDQNetwork

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


def test_continuous_critic_accepts_leading_batch_dims():
    critic = RLBOEDQNetwork(**_critic_kwargs())
    y, designs, mask = _history()
    design = torch.rand(*BATCH_SHAPE, *EVENT_DESIGN)

    q = critic(y, designs, design, mask=mask)
    flat = critic(_flat(y), _flat(designs), _flat(design), mask=_flat(mask))

    assert q.shape == (*BATCH_SHAPE, 1)
    torch.testing.assert_close(q, flat.reshape(*BATCH_SHAPE, 1))


def test_discrete_critic_accepts_leading_batch_dims():
    critic = RLBOEDDiscreteQNetwork(n_designs=5, **_critic_kwargs())
    y, designs, mask = _history()

    q = critic(y, designs, mask=mask)
    flat = critic(_flat(y), _flat(designs), mask=_flat(mask))

    assert q.shape == (*BATCH_SHAPE, 5)
    torch.testing.assert_close(q, flat.reshape(*BATCH_SHAPE, 5))


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
    samples = net.sample(y, designs, n_samples=7, mask=mask)

    assert log_prob.shape == BATCH_SHAPE
    torch.testing.assert_close(log_prob, flat.reshape(BATCH_SHAPE))
    assert samples.shape == (7, *BATCH_SHAPE, *EVENT_THETA)


def test_posterior_rejects_theta_with_other_batch_shape():
    net = DeepSetVariationalPosteriorNet(EVENT_Y, EVENT_DESIGN, EVENT_THETA)
    y, designs, mask = _history()
    with pytest.raises(ValueError, match="theta must have shape"):
        net.log_prob(torch.randn(6, *EVENT_THETA), y, designs, mask=mask)


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


def _assert_outputs_close(actual, expected):
    for name in (
        "acquisition_logits",
        "mixture_logits",
        "mixture_means",
        "mixture_scales",
    ):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


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
    expected = type(flat)(
        **{
            name: getattr(flat, name).reshape(
                *BATCH_SHAPE, *getattr(flat, name).shape[1:]
            )
            for name in (
                "acquisition_logits",
                "mixture_logits",
                "mixture_means",
                "mixture_scales",
            )
        }
    )
    _assert_outputs_close(output, expected)


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
