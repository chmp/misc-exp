import json

import numpy as np
import pytest

from chmp.ds import (
    Object,
    timed,
    singledispatch_on,
    piecewise_linear,
    piecewise_logarithmic,
    szip,
    copy_structure,
    json_numpy_default,
)


def test_object():
    a = Object(a=2, b=3)

    assert a == Object(a=2, b=3)
    assert a != Object(a=2, b=4)
    assert Object(a, b=4) == Object(a=2, b=4)

    assert a.a == 2
    assert a.b == 3

    assert vars(a) == dict(a=2, b=3)


def test_timed():
    with timed():
        assert True is True

    with timed("label"):
        assert True is True


def test_singledispatch_on():
    @singledispatch_on(1)
    def foo(a, b):
        return 1

    @foo.register(int)
    def bar(a, b):
        return 2

    assert foo(0, 2) == 2
    assert foo(1, None) == 1
    assert foo(None, 2) == 2
    assert foo(None, None) == 1


def test_piecewise_linear():
    assert piecewise_linear([0, 1, 2], [1, 2, 3], -0.1) == pytest.approx(1.0)
    assert piecewise_linear([0, 1, 2], [1, 2, 3], +0.0) == pytest.approx(1.0)
    assert piecewise_linear([0, 1, 2], [1, 2, 3], +0.2) == pytest.approx(1.2)
    assert piecewise_linear([0, 1, 2], [1, 2, 3], +0.6) == pytest.approx(1.6)
    assert piecewise_linear([0, 1, 2], [1, 2, 3], +1.0) == pytest.approx(2.0)
    assert piecewise_linear([0, 1, 2], [1, 2, 3], +1.5) == pytest.approx(2.5)
    assert piecewise_linear([0, 1, 2], [1, 2, 3], +2.0) == pytest.approx(3.0)
    assert piecewise_linear([0, 1, 2], [1, 2, 3], +2.2) == pytest.approx(3.0)


def test_piecewise_logarithmic():
    assert piecewise_logarithmic([0, 1, 2], [10, 100, 1000], -0.1) == pytest.approx(
        10 ** 1.0
    )
    assert piecewise_logarithmic([0, 1, 2], [10, 100, 1000], +0.0) == pytest.approx(
        10 ** 1.0
    )
    assert piecewise_logarithmic([0, 1, 2], [10, 100, 1000], +0.2) == pytest.approx(
        10 ** 1.2
    )
    assert piecewise_logarithmic([0, 1, 2], [10, 100, 1000], +0.6) == pytest.approx(
        10 ** 1.6
    )
    assert piecewise_logarithmic([0, 1, 2], [10, 100, 1000], +1.0) == pytest.approx(
        10 ** 2.0
    )
    assert piecewise_logarithmic([0, 1, 2], [10, 100, 1000], +1.5) == pytest.approx(
        10 ** 2.5
    )
    assert piecewise_logarithmic([0, 1, 2], [10, 100, 1000], +2.0) == pytest.approx(
        10 ** 3.0
    )
    assert piecewise_logarithmic([0, 1, 2], [10, 100, 1000], +2.2) == pytest.approx(
        10 ** 3.0
    )


def test_szip():
    actual = szip([{"a": 1, "b": (2, 3)}, {"a": 4, "b": (5, 6)}])

    assert actual == {"a": [1, 4], "b": ([2, 5], [3, 6])}


def test_szip_with_schema():
    actual, schema = szip(
        [{"a": 1, "b": (2, 3)}, {"a": 4, "b": (5, 6)}], return_schema=True
    )

    assert actual == {"a": [1, 4], "b": ([2, 5], [3, 6])}

    assert schema == {"a": None, "b": (None, None)}


@pytest.mark.parametrize(
    "value",
    [
        np.int0(1),
        np.int8(-2),
        np.int16(-5),
        np.int32(+32),
        np.int64(120),
        np.uint0(1),
        np.uint8(2),
        np.uint16(5),
        np.uint32(+32),
        np.uint64(120),
        np.float16(-2.0),
        np.float32(-2.0),
        np.float64(-2.0),
        np.float128(-2.0),
        np.array(2.0),
        np.array([1, 2, 3]),
        np.array([[1], [2], [3]]),
    ],
)
def test_json_nump_default__roundtrip(value):
    actual = json.loads(json.dumps(value, default=json_numpy_default))

    assert np.shape(actual) == np.shape(value)
    np.testing.assert_allclose(actual, value)


@pytest.mark.parametrize(
    "input, expected",
    [
        ((1, 2, 3), ("", "", "")),
        (((1, 2), 3), (("", ""), "")),
        ({"k": (1, 2)}, {"k": ("", "")}),
    ],
)
def test_copy_structure_examples(input, expected):
    assert copy_structure(input, "") == expected


@pytest.mark.parametrize("input", [(1, 2, 3), ((1, 2), 3), {"k": (1, 2)}])
def test_copy_structure__keep_structure(input):
    assert copy_structure(input, input) == input


@pytest.mark.parametrize(
    "template, obj",
    [((1, 2, 3), ("",)), (((1, 2), 3), ("", "")), ({"k": (1, 2)}, {"": ""})],
)
def test_copy_structure__mismatch(template, obj):
    with pytest.raises(ValueError):
        copy_structure(template, obj)
