# coding=utf8
from chmp.ds import shuffle, shuffled


def test_shuffle():
    l = list(range(3))
    shuffle(42, l)

    assert sorted(l) == [0, 1, 2]

    # test that seed is taken into account
    assert shuffled(13, [0, 1, 2]) != shuffled(42, [0, 1, 2])
