"""Tests for design spaces."""

import pytest
import torch

from testbed.envs import CategoricalDesignSpace


def test_categorical_design_space_encodes_and_decodes_physical_designs():
    space = CategoricalDesignSpace(torch.tensor([[1.0], [5.0], [10.0]]))

    designs = torch.tensor([[10.0], [1.0], [5.0]])
    indices = space.encode(designs)

    assert space.n_actions == 3
    assert space.contains(designs)
    assert torch.equal(indices, torch.tensor([2, 0, 1]))
    assert torch.equal(space.decode(indices), designs)
    assert not space.contains(torch.tensor([[2.0]]))
    with pytest.raises(ValueError, match="integer-valued"):
        space.decode(torch.tensor([0.5]))


@pytest.mark.parametrize("values", [torch.tensor([]), torch.tensor([[1.0], [1.0]])])
def test_categorical_design_space_rejects_empty_or_duplicate_actions(values):
    with pytest.raises(ValueError):
        CategoricalDesignSpace(values)
