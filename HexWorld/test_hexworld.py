import hexworld
import numpy as np


def test_cube_point__basics():
    p = hexworld.CubePoint()
    assert p == hexworld.CubePoint(0, 0, 0)
    assert hexworld.CubePoint(1, 2, -3) == hexworld.CubePoint(1, 2, -3)

    assert +hexworld.CubePoint(1, 2, -3) == hexworld.CubePoint(1, 2, -3)
    assert -hexworld.CubePoint(1, 2, -3) == hexworld.CubePoint(-1, -2, +3)


def test_buffer():
    expected = hexworld.Int64Buffer(5, 10)
    expected[0, 2] =  20
    expected[4, 3] = 1

    actual = hexworld.to_buffer(np.asarray(expected))
    assert actual == expected


def test_rotate():
    p = hexworld.CubePoint(0, -1, 1)
    assert hexworld.rotate(p, +1) == hexworld.CubePoint(-1, +0, +1)
    assert hexworld.rotate(p, -1) == hexworld.CubePoint(+1, -1, +0)

    assert hexworld.rotate(p, +37) == hexworld.CubePoint(-1, +0, +1)
    assert hexworld.rotate(p, -37) == hexworld.CubePoint(+1, -1, +0)
