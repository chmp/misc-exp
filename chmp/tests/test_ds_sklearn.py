from chmp.ds import FuncTransformer, transform, column_transform, as_frame

import numpy as np
import pandas.util.testing as pdt


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
