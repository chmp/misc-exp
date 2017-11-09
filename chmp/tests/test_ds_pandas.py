import pandas as pd
import pandas.util.testing as pdt
import pytest

import chmp.ds


def test_fix_categories_example():
    actual = chmp.ds.fix_categories(
        pd.Series(['a', 'b', 'c', 'd'], dtype='category'),
        categories=['a', 'b', 'c'],
        other_category='other',
    )

    expected = pd.Series(['a', 'b', 'c', 'other'], dtype='category')
    pdt.assert_series_equal(actual, expected)


def test_fix_categories_missing_other():
    # d would be removed, but no other category is given ...
    with pytest.raises(ValueError):
        chmp.ds.fix_categories(
            pd.Series(['a', 'b', 'c', 'd'], dtype='category'),
            categories=['a', 'b', 'c'],
        )


def test_find_high_frequency_categories():
    actual = chmp.ds.find_high_frequency_categories(
        pd.Series(['a'] * 5 + ['b'] * 4 + ['c'], dtype='category'),
        min_frequency=0.20,
    )

    expected = ['a', 'b']

    assert actual == expected


def test_find_high_frequency_categories_n_max():
    actual = chmp.ds.find_high_frequency_categories(
        pd.Series(['a'] * 5 + ['b'] * 4 + ['c'], dtype='category'),
        min_frequency=0.20,
        n_max=1,
    )

    expected = ['a']

    assert actual == expected
