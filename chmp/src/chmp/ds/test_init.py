from chmp.ds import Object, compute_moi, orient, timed

import numpy as np


def test_object():
    a = Object(a=2, b=3)

    assert a == Object(a=2, b=3)
    assert a != Object(a=2, b=4)
    assert Object(a, b=4) == Object(a=2, b=4)

    assert a.a == 2
    assert a.b == 3

    assert vars(a) == dict(a=2, b=3)


def test_orient_random():
    np.random.seed(424_242)
    pos = np.random.uniform(-1, +1, size=(100, 3))
    pos = np.transpose(
        np.stack(
            [pos[:, 0] - pos[:, 1], pos[:, 0] * pos[:, 1] ** 2, pos[:, 1] * pos[:, 2]]
        )
    )

    pos = orient(pos)
    moi = compute_moi(pos)

    # all non digaonal elements are zero
    np.testing.assert_almost_equal(moi - np.diag(np.diag(moi)), 0)

    # the diagonal is increasing
    np.testing.assert_array_equal(np.diff(np.diag(moi)) > 0, True)


def test_orient_straight_line():
    u = np.linspace(-10, +10, 10)
    pos = np.stack([0.5 ** 0.5 * u, 0.5 ** 0.5 * u]).T

    opos = orient(pos)

    np.testing.assert_almost_equal(opos[:, 0], u)
    np.testing.assert_almost_equal(opos[:, 1], 0)


def test_timed():
    with timed():
        assert True is True

    with timed("label"):
        assert True is True
