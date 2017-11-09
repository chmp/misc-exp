import operator as op

import numpy as np
import pandas as pd
import pandas.util.testing as pdt

from chmp.ds import (
    FuncClassifier,
    FuncTransformer,

    as_frame,
    build_pipeline,
    column_transform,
    filter_low_frequency_categories,
    transform,
)


def test_func_transformer():
    est = FuncTransformer(lambda x: x + 2)
    est.fit(np.ones((5, 2)))

    np.testing.assert_almost_equal(
        est.transform(np.ones((5, 2))),
        3 * np.ones((5, 2)),
    )


def test_transform():
    est = transform(lambda x, a, b: a * x + b, a=2, b=3)
    est.fit(np.ones((5, 2)))

    np.testing.assert_almost_equal(
        est.transform(np.ones((5, 2))),
        2 * np.ones((5, 2)) + 3,
    )


def test_column_transform_single():
    x = as_frame(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9])

    est = column_transform('a', lambda s: s + 2)
    est.fit(x)

    pdt.assert_almost_equal(
        est.transform(x),
        as_frame(a=[3, 4, 5], b=[4, 5, 6], c=[7, 8, 9]),
    )


def test_column_transform_multiple():
    x = as_frame(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9])

    est = column_transform(['a', 'b'], lambda s: s + 2)
    est.fit(x)

    pdt.assert_almost_equal(
        est.transform(x),
        as_frame(a=[3, 4, 5], b=[6, 7, 8], c=[7, 8, 9]),
    )


def test_column_transform_multiple_kwargs():
    x = as_frame(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9])

    actual = column_transform(a=lambda s: s + 2, b=lambda s: s + 1).fit_transform(x)
    expected = as_frame(a=[3, 4, 5], b=[5, 6, 7], c=[7, 8, 9])

    pdt.assert_almost_equal(actual, expected)


def test_filter_low_frequency_columns():
    actual = pd.DataFrame({
        'a': pd.Series(['a'] * 5 + ['b'] * 4 + ['c'], dtype='category'),
        'b': pd.Series([1, 2, 3, 4, 5] * 2),
    })

    actual = (
        filter_low_frequency_categories('a', min_frequency=0.2, other_category='other')
       .fit_transform(actual)
    )

    expected = pd.DataFrame({
        'a': pd.Series(['a'] * 5 + ['b'] * 4 + ['other'], dtype='category'),
        'b': pd.Series([1, 2, 3, 4, 5] * 2),
    })

    pdt.assert_frame_equal(actual, expected)


def test_simple_pipeline():
    est = build_pipeline(
        transform=column_transform(
            a=op.abs,
            b=op.neg,
        ),
        predict=FuncClassifier(lambda df: (
            np.stack([df['b'] <= 0, df['b'] > 0], axis=1)
            .astype(float)
        )),
    )

    est.fit(as_frame(a=[1, 2, 3], b=[-1, +1, -1]))
    actual = est.predict(as_frame(a=[1, 2, 3], b=[-1, +1, -1]))

    expected = np.asarray([1, 0, 1])

    np.testing.assert_almost_equal(actual, expected)
