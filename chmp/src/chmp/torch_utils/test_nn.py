import numpy as np
import pandas as pd
import pytest
import torch

from chmp.ds import assert_has_schema
from chmp.torch_utils import (
    factorized_quadratic,
    masked_softmax,
    linear,
    DiagonalScaleShift,
    call_torch,
    t2n,
    NumpyDataset,
    padded_collate_fn,
)


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
        call_torch(torch.sqrt, np.asarray([1, 4, 9], dtype="float")), [1, 2, 3]
    )

    np.testing.assert_almost_equal(
        call_torch(
            torch.add,
            np.asarray([1, 2, 3], dtype="float"),
            np.asarray([4, 5, 6], dtype="float"),
        ),
        [5, 7, 9],
    )


def test_call_torch_structured():
    a, b = call_torch(
        lambda t: (t[0] + t[1], t[1] - t[0]),
        (np.asarray([1, 2, 3], dtype="float"), np.asarray([4, 5, 6], dtype="float")),
    )

    np.testing.assert_almost_equal(a, [5, 7, 9])
    np.testing.assert_almost_equal(b, [3, 3, 3])


def test_call_torch_batched():
    np.testing.assert_almost_equal(
        call_torch(torch.sqrt, np.arange(1024).astype("float"), batch_size=128),
        np.arange(1024) ** 0.5,
    )


def test_t2n_examples():
    t2n(torch.zeros(10))
    t2n((torch.zeros(10, 2), torch.zeros(10)))


def test_numpy_dataset():
    ds = NumpyDataset(pd.DataFrame({"a": np.zeros(10), "b": np.zeros(10)}))
    assert len(ds) == 10
    ds[0]


def test_padded_collate_fn__empty_item():
    batch = [(), ()]
    assert padded_collate_fn(batch) == ()


def test_padded_collate_fn__empty_batch():
    with pytest.raises(ValueError):
        padded_collate_fn([])


def test_padded_collate_fn__example_scalars():
    batch = [1, 2, 3]
    actual = padded_collate_fn(batch)

    assert_has_schema(actual, None)
    np.testing.assert_allclose(actual, [1, 2, 3])


def test_padded_collate_fn__example_sequences():
    batch = [[1, 2, 3], [4, 5], [6]]
    actual = padded_collate_fn(batch)

    assert_has_schema(actual, None)
    np.testing.assert_allclose(actual, [[1, 2, 3], [4, 5, 0], [6, 0, 0]])


def test_padded_collate_fn__example_matrices():
    batch = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2]], [[3], [4], [5]]]
    actual = padded_collate_fn(batch)

    assert_has_schema(actual, None)
    np.testing.assert_allclose(
        actual,
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[3, 0, 0], [4, 0, 0], [5, 0, 0]],
        ],
    )


def test_padded_collate_fn__example_mixed_flat():
    batch = [
        (1, [1, 2, 3], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        (2, [4, 5], [[1, 2]]),
        (3, [6], [[3], [4], [5]]),
    ]

    actual = padded_collate_fn(batch)

    assert_has_schema(actual, (None, None, None))

    np.testing.assert_allclose(actual[0], [1, 2, 3])
    np.testing.assert_allclose(actual[1], [[1, 2, 3], [4, 5, 0], [6, 0, 0]])
    np.testing.assert_allclose(
        actual[2],
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[3, 0, 0], [4, 0, 0], [5, 0, 0]],
        ],
    )


def test_padded_collate_fn__example_mixed_nested():
    batch = [
        ((1, [1, 2, 3]), [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ((2, [4, 5]), [[1, 2]]),
        ((3, [6]), [[3], [4], [5]]),
    ]

    actual = padded_collate_fn(batch)

    assert_has_schema(actual, ((None, None), None))

    np.testing.assert_allclose(actual[0][0], [1, 2, 3])
    np.testing.assert_allclose(actual[0][1], [[1, 2, 3], [4, 5, 0], [6, 0, 0]])
    np.testing.assert_allclose(
        actual[1],
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[3, 0, 0], [4, 0, 0], [5, 0, 0]],
        ],
    )


def test_padded_collate_fn__example_mixed_dict():
    batch = [
        ({"s": 1, "a": [1, 2, 3]}, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ({"s": 2, "a": [4, 5]}, [[1, 2]]),
        ({"s": 3, "a": [6]}, [[3], [4], [5]]),
    ]

    actual = padded_collate_fn(batch)

    assert_has_schema(actual, ({"s": None, "a": None}, None))

    np.testing.assert_allclose(actual[0]["s"], [1, 2, 3])
    np.testing.assert_allclose(actual[0]["a"], [[1, 2, 3], [4, 5, 0], [6, 0, 0]])
    np.testing.assert_allclose(
        actual[1],
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 0], [0, 0, 0], [0, 0, 0]],
            [[3, 0, 0], [4, 0, 0], [5, 0, 0]],
        ],
    )
