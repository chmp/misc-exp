import numpy as np
import torch
import pytest

# NOTE: also registers the KL divergence
from . import factorized_quadratic, masked_softmax, linear


def test_linear_shape():
    weights = torch.zeros(10, 5)
    assert linear(torch.zeros(20, 10), weights).shape == (20, 5)


def test_factorized_quadratic_shape():
    weights = torch.zeros(2, 10, 5)
    assert factorized_quadratic(torch.zeros(20, 10), weights).shape == (20, 5)


def test_masked_softmax():
    actual = masked_softmax(
        torch.tensor([1., 2., 3.]), torch.tensor([1, 1, 0], dtype=torch.uint8)
    )
    actual = np.asarray(actual)

    expected = np.asarray([np.exp(1), np.exp(2), 0])
    expected = expected / expected.sum()

    np.testing.assert_allclose(actual, expected)