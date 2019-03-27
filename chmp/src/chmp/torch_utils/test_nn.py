import numpy as np
import torch

from .nn import factorized_quadratic, masked_softmax, linear, DiagonalScaleShift, call_torch


def test_linear_shape():
    weights = torch.zeros(10, 5)
    assert linear(torch.zeros(20, 10), weights).shape == (20, 5)


def test_factorized_quadratic_shape():
    weights = torch.zeros(2, 10, 5)
    assert factorized_quadratic(torch.zeros(20, 10), weights).shape == (20, 5)


def test_masked_softmax():
    actual = masked_softmax(
        torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1, 1, 0], dtype=torch.uint8)
    )
    actual = np.asarray(actual)

    expected = np.asarray([np.exp(1), np.exp(2), 0])
    expected = expected / expected.sum()

    np.testing.assert_allclose(actual, expected)


def test_diagonal_scale_shift():
    m = DiagonalScaleShift(shift=torch.ones(10), scale=2.0 * torch.ones(10))
    assert m(torch.zeros(20, 10)).shape == (20, 10)


def test_call_torch():
    np.testing.assert_almost_equal(
        call_torch(torch.sqrt, np.asarray([1, 4, 9], dtype='float')),
        [1, 2, 3],
    )

    np.testing.assert_almost_equal(
        call_torch(
            torch.add,
            np.asarray([1, 2, 3], dtype='float'),
            np.asarray([4, 5, 6], dtype='float')
        ),
        [5, 7, 9],
    )


def test_call_torch_structured():
    a, b = call_torch(
        lambda t: (t[0] + t[1], t[1] - t[0]),
        (np.asarray([1, 2, 3], dtype='float'), np.asarray([4, 5, 6], dtype='float')),
    )

    np.testing.assert_almost_equal(a, [5, 7, 9])
    np.testing.assert_almost_equal(b, [3, 3, 3])


def test_call_torch_batched():
    np.testing.assert_almost_equal(
        call_torch(torch.sqrt, np.arange(1024).astype('float'), batch_size=128),
        np.arange(1024) ** 0.5,
    )
