import collections

import pandas as pd
import pandas.util.testing as pdt

from chmp.ds import fix_categories, find_high_frequency_categories


def test_example_int():
    s = pd.Series([0, 1, 2] * 3)
    actual = fix_categories(s, [-1, 1, 2], ordered=True, other_category=-1)
    expected = pd.Series([-1, 1, 2] * 3, dtype='category').cat.as_ordered()

    pdt.assert_almost_equal(actual, expected)


def test_example_string__other():
    s = pd.Series(['a', 'b', 'c', 'e', 'c', 'b', 'a'], dtype='category')
    actual = fix_categories(s, ['b', 'c'], 'other')
    expected = pd.Series(['other', 'b', 'c', 'other', 'c', 'b', 'other'], dtype='category')

    pdt.assert_almost_equal(actual, expected)


def test_example_string__grouping():
    groups = collections.OrderedDict([('0', ['a', 'b']), ('1', ['c', 'e'])])

    s = pd.Series(['a', 'b', 'c', 'e', 'c', 'b', 'a'], dtype='category')
    actual = fix_categories(s, groups=groups)
    expected = pd.Series(list('0011100'), dtype='category').cat.reorder_categories(['0', '1'])

    pdt.assert_almost_equal(actual, expected)


def test_find_high_frequency_categories_examples():
    assert find_high_frequency_categories(pd.Series(['a', 'a', 'a', 'a', 'b']), 0.2) == ['a']
    assert find_high_frequency_categories(pd.Series(['a', 'a', 'a', 'a', 'b', 'c', 'c']), 0.2) == ['a', 'c']
    assert find_high_frequency_categories(pd.Series(['a', 'a', 'a', 'a', 'b', 'c', 'c']), 0.2, n_max=1) == ['a']
