import pytest
import torch

from testbed.design_nets import DADNet, TransformerDADNet
from testbed.designers import PolicyDesigner
from testbed.tasks import LocationFinding


@pytest.mark.parametrize("t", [0, 3])
def test_dad_net_outputs_design_shape(t):
    b = 5
    dim_y = 2
    dim_design = 3
    net = DADNet(dim_y=dim_y, dim_design=dim_design, dim_h=16, dim_emb=8)

    y_history = torch.randn(b, t, dim_y)
    design_history = torch.randn(b, t, dim_design)

    design = net(y_history, design_history)

    assert design.shape == (b, dim_design)


def test_dad_net_supports_deeper_config():
    b = 4
    dim_y = 1
    dim_design = 5
    net = DADNet(
        dim_y=dim_y,
        dim_design=dim_design,
        dim_h=16,
        dim_emb=8,
        encoder_depth=2,
        emitter_dims=[12, 6],
    )

    y_history = torch.randn(b, 2, dim_y)
    design_history = torch.randn(b, 2, dim_design)

    design = net(y_history, design_history)

    assert design.shape == (b, dim_design)


@pytest.mark.parametrize("pooling", ["mean", "weighted", "attention", "cls"])
@pytest.mark.parametrize("t", [0, 3])
def test_transformer_dad_net_outputs_design_shape(pooling, t):
    b = 5
    dim_y = 2
    dim_design = 4
    net = TransformerDADNet(
        dim_y=dim_y,
        dim_design=dim_design,
        dim_h=16,
        dim_emb=8,
        num_heads=2,
        num_blocks=1,
        pooling=pooling,
    )

    y_history = torch.randn(b, t, dim_y)
    design_history = torch.randn(b, t, dim_design)

    design = net(y_history, design_history)

    assert design.shape == (b, dim_design)


def test_policy_designer_rollout_with_dad_net():
    task = LocationFinding(n_sources=1, dim_theta=2)
    net = DADNet(dim_y=1, dim_design=2, dim_h=16, dim_emb=8)
    designer = PolicyDesigner(task, T=2, policy_net=net)
    theta = task.sample_prior(3)

    y_history, design_history = designer.rollout(theta)

    assert y_history.shape == (3, 2, *task.event_shape_y)
    assert design_history.shape == (3, 2, *task.event_shape_design)
