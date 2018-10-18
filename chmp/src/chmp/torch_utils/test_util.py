import numpy as np
import pandas as pd
import pytest

from ._util import pack, unpack, apply_dtype


@pytest.mark.parametrize(
    "input, expected",
    [
        [(None,), (None,)],
        [(0,), (0,)],
        [(0, (1, 2)), (0, (1, 2))],
        [(None, (1, 2)), (None, (1, 2))],
        [(0, (1, 2), {"a": 3, "b": 4}), (0, (1, 2), {"a": 3, "b": 4})],
    ],
)
def test_pack_unpack(input, expected):
    assert unpack(*pack(input)) == expected


@pytest.mark.parametrize(
    "obj",
    [
        2,
        (1, 2),
        (1, (2, 3)),
        (1, (2, (3, 4))),
        (((1, 2), 3), 4),
        (1, (2, 3), 4),
        {},
        {1: 2},
        {1: (2, 3)},
        {1: 2, 3: 4},
        {1: (2, {3: 4})},
    ],
)
def test_pack_unpack_roundtrip(obj):
    key, values = pack(obj)
    roundtripped = unpack(key, values)

    assert roundtripped == obj

    # key is hashable
    hash(key)


def test_apply_dtype__single_type():
    x = apply_dtype("float32", [[1, 2], [3, 4], [5, 6]])

    assert x.shape == (3, 2)
    assert x.dtype == np.float32


def test_apply_dtype__single_type_y():
    y = apply_dtype("float32", [1, 2, 3])

    assert y.shape == (3,)
    assert y.dtype == np.float32


def test_apply_dtype__separate_types():
    x, y = apply_dtype(("float16", "float64"), ([[1, 2], [3, 4], [5, 6]], [7, 8, 9]))

    assert x.shape == (3, 2)
    assert x.dtype == np.float16

    assert y.shape == (3,)
    assert y.dtype == np.float64


def test_apply_dtype__pandas_dataframe():
    x = pd.DataFrame().assign(a=[1, 2, 3], b=[1, 2, 3])
    x = apply_dtype("float32", x)

    assert x.shape == (3, 2)
    assert x.dtype == np.float32


def test_apply_dtype__pandas_series():
    y = pd.Series([1, 2, 3])
    y = apply_dtype("float32", y)

    assert y.shape == (3,)
    assert y.dtype == np.float32


def test_apply_dtype__noop():
    x = apply_dtype(None, [[1, 2], [3, 4], [5, 6]])

    assert isinstance(x, list)


def test_apply_dtype__dict():
    x = apply_dtype("float32", {"a": [1, 2, 3], "b": [4, 5, 6]})

    assert isinstance(x, dict)
    assert x["a"].dtype == np.float32
    assert x["b"].dtype == np.float32


def test_apply_dtype__pandas_dict():
    x = pd.DataFrame().assign(a=[1, 2, 3], b=[1, 2, 3])
    x = apply_dtype({"a": "float16", "b": "float32"}, x)

    assert isinstance(x, dict)
    assert x["a"].dtype == np.float16
    assert x["b"].dtype == np.float32


def test_apply_dtype__tuple_arg():
    x = apply_dtype("float32", ([1, 2, 3], [4, 5, 6]))

    assert isinstance(x, tuple)
    assert x[0].dtype == np.float32
    assert x[1].dtype == np.float32


def test_apply_dtype__tuple():
    x = apply_dtype(("float16", "float32"), ([1, 2, 3], [4, 5, 6]))

    assert isinstance(x, tuple)
    assert x[0].dtype == np.float16
    assert x[1].dtype == np.float32
