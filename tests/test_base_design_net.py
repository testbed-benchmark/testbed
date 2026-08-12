"""Tests for the common design-network contract."""

import torch

from testbed.design_nets import DesignNet, DesignProposal


class RecordingDesignNet(DesignNet):
    """Minimal deterministic design network that records its mask."""

    def __init__(self):
        super().__init__()
        self.last_mask = None

    def forward(self, y_history, design_history, *, mask=None):
        self.last_mask = mask
        return design_history.sum(dim=1)


def test_design_net_propose_wraps_forward_output_and_forwards_mask():
    net = RecordingDesignNet()
    y_history = torch.randn(2, 3, 1)
    design_history = torch.randn(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    proposal = net.propose(
        y_history,
        design_history,
        mask=mask,
        reparameterize_policy=True,
    )

    assert isinstance(proposal, DesignProposal)
    assert torch.equal(proposal.raw_design, design_history.sum(dim=1))
    assert proposal.log_prob is None
    assert proposal.deterministic_raw_design is None
    assert proposal.info == {}
    assert net.last_mask is mask


def test_design_proposal_info_has_independent_defaults():
    first = DesignProposal(raw_design=torch.tensor(1.0))
    second = DesignProposal(raw_design=torch.tensor(2.0))

    first.info["source"] = "first"

    assert second.info == {}
