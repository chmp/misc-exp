import numpy as np
import pandas as pd
import pytest

from ._batched import (
    get_number_of_samples,
    pack,
    unpack,
    sized_generator,
    parallel_concat,
    apply_dtype,
    iter_batch_indices,
    iter_batched,
)


def test_iter_batched__example():
    data = [np.arange(96), np.random.randint(0, 1024, size=(96, 3))]
    actual = list(iter_batched(data, batch_size=32))

    assert len(actual) == 3
    assert actual[0] == (pytest.approx(data[0][0:32]), pytest.approx(data[1][0:32]))
    assert actual[1] == (pytest.approx(data[0][32:64]), pytest.approx(data[1][32:64]))
    assert actual[2] == (pytest.approx(data[0][64:96]), pytest.approx(data[1][64:96]))


def test_iter_batched__partial_complete():
    data = [np.arange(96), np.random.randint(0, 1024, size=(96, 3))]
    actual = list(iter_batched(data, batch_size=50))

    assert len(actual) == 1
    assert actual[0] == (pytest.approx(data[0][0:50]), pytest.approx(data[1][0:50]))


def test_iter_batched__partial_incomplete():
    data = [np.arange(96), np.random.randint(0, 1024, size=(96, 3))]
    actual = list(iter_batched(data, batch_size=50, only_complete=False))

    assert len(actual) == 2
    assert actual[0] == (pytest.approx(data[0][0:50]), pytest.approx(data[1][0:50]))
    assert actual[1] == (pytest.approx(data[0][50:96]), pytest.approx(data[1][50:96]))


def test_iter_batched__reversed():
    data = [np.arange(96), np.random.randint(0, 1024, size=(96, 3))]
    indices = np.arange(96)[::-1]

    actual = list(iter_batched(data, batch_size=32, indices=indices))

    assert len(actual) == 3

    # NOTE: note the inversion of the data via [::-1]
    assert actual[0] == (
        pytest.approx(data[0][::-1][0:32]),
        pytest.approx(data[1][::-1][0:32]),
    )
    assert actual[1] == (
        pytest.approx(data[0][::-1][32:64]),
        pytest.approx(data[1][::-1][32:64]),
    )
    assert actual[2] == (
        pytest.approx(data[0][::-1][64:96]),
        pytest.approx(data[1][::-1][64:96]),
    )


def test_iter_batched__no_data():
    with pytest.raises(ValueError):
        list(iter_batched({}))


def test_iter_batched__not_enough_data():
    with pytest.raises(ValueError):
        list(iter_batched([np.zeros(32)], batch_size=64))


def test_assert_consistent_shape():
    with pytest.raises(ValueError):
        get_number_of_samples()

    with pytest.raises(ValueError):
        get_number_of_samples(None, None)

    with pytest.raises(ValueError):
        get_number_of_samples(np.zeros([10]), np.zeros([20]))

    assert get_number_of_samples(np.zeros([10]), np.zeros([10, 20])) == 10

    assert get_number_of_samples(None, np.zeros([10]), np.zeros([10, 20])) == 10


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


def test_sized_generator__sized_content():
    i = sized_generator(lambda: iter([1, 2, 3]), length=3)

    assert len(i) == 3
    assert list(i) == [1, 2, 3]


def test_parallel_concat():
    assert parallel_concat([[[10, 20], [5]], [[30, 40], [6]]]) == (
        pytest.approx(np.asarray([10, 20, 30, 40])),
        pytest.approx(np.asarray([5, 6])),
    )


def test_parallel_concat_no_items():
    with pytest.raises(ValueError):
        parallel_concat([])


def test_parallel_concat_different_subitems():
    with pytest.raises(ValueError):
        # NOTE: the first entry has 2 items, the second only one
        parallel_concat([[[10, 20], [5]], [[30, 40]]])


def test_iter_batch_indices():
    actual = np.sort(
        np.concatenate(list(iter_batch_indices(100, batch_size=10, shuffle=True)))
    )
    assert actual == pytest.approx(np.arange(100))


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
