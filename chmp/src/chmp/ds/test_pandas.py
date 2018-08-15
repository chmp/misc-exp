import collections

import pandas as pd
import pandas.util.testing as pdt
import pytest

from chmp.ds import fix_categories, find_high_frequency_categories


def callargs(*args, **kwargs):
    return dict(args=args, kwargs=kwargs)


def test_example_int():
    s = pd.Series([0, 1, 2] * 3)
    actual = fix_categories(s, [-1, 1, 2], ordered=True, other_category=-1)
    expected = pd.Series([-1, 1, 2] * 3, dtype="category").cat.as_ordered()

    pdt.assert_almost_equal(actual, expected)


def test_example_string__other():
    s = pd.Series(["a", "b", "c", "e", "c", "b", "a"], dtype="category")
    actual = fix_categories(s, ["b", "c"], "other")
    expected = pd.Series(
        ["other", "b", "c", "other", "c", "b", "other"], dtype="category"
    )

    pdt.assert_almost_equal(actual, expected)


def test_example_string__grouping():
    groups = collections.OrderedDict([("0", ["a", "b"]), ("1", ["c", "e"])])

    s = pd.Series(["a", "b", "c", "e", "c", "b", "a"], dtype="category")
    actual = fix_categories(s, groups=groups)
    expected = pd.Series(list("0011100"), dtype="category").cat.reorder_categories(
        ["0", "1"]
    )

    pdt.assert_almost_equal(actual, expected)


@pytest.mark.parametrize(
    "spec",
    [
        dict(
            label="example1",
            expected=["a"],
            **callargs(pd.Series(["a", "a", "a", "a", "b"]), 0.2),
        ),
        dict(
            label="example2",
            expected=["a", "c"],
            **callargs(pd.Series(["a", "a", "a", "a", "b", "c", "c"]), 0.2),
        ),
        dict(
            label="example3",
            expected=["a"],
            **callargs(pd.Series(["a", "a", "a", "a", "b", "c", "c"]), 0.2, n_max=1),
        ),
    ],
)
def test_find_high_frequency_categories_examples(spec):
    actual = find_high_frequency_categories(*spec["args"], **spec["kwargs"])
    assert actual == spec["expected"]


def test_fix_categories_example():
    actual = fix_categories(
        pd.Series(["a", "b", "c", "d"], dtype="category"),
        categories=["a", "b", "c"],
        other_category="other",
    )

    expected = pd.Series(["a", "b", "c", "other"], dtype="category")
    pdt.assert_series_equal(actual, expected)


def test_fix_categories_missing_other():
    # d would be removed, but no other category is given ...
    with pytest.raises(ValueError):
        fix_categories(
            pd.Series(["a", "b", "c", "d"], dtype="category"),
            categories=["a", "b", "c"],
        )


def test_find_high_frequency_categories():
    actual = find_high_frequency_categories(
        pd.Series(["a"] * 5 + ["b"] * 4 + ["c"], dtype="category"), min_frequency=0.20
    )

    expected = ["a", "b"]

    assert actual == expected


def test_find_high_frequency_categories_n_max():
    actual = find_high_frequency_categories(
        pd.Series(["a"] * 5 + ["b"] * 4 + ["c"], dtype="category"),
        min_frequency=0.20,
        n_max=1,
    )

    expected = ["a"]

    assert actual == expected
