import pytest
import torch

from testbed.core.util import sort_by_norm


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
