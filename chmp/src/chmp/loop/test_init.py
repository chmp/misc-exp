# coding=utf8
import contextlib
from chmp.loop import tdformat, bar, Loop

import pytest


@pytest.mark.parametrize(
    'time_delta, formatted', [
        (30, '30.00s'),
        (90, '1m 30s'),
        (2 * 60 * 60 + 14 * 60 + 20, '2h 14m'),
        (3 * 24 * 60 * 60 + 2 * 60 * 60 + 14 * 60 + 20, '3d 2h'),
        (2 * 7 * 24 * 60 * 60 + 3 * 24 * 60 * 60 + 2, '2w 3d'),
    ],
)
def test_tdformat(time_delta, formatted):
    assert tdformat(time_delta) == formatted


# NOTE: add 0.01 to ensure correct rounding behavior
@pytest.mark.parametrize(
    'progress,result', [
        (0.401, u'⣿⣿⣿⣿▫▫▫▫▫▫'),
        (0.411, u'⣿⣿⣿⣿ ▫▫▫▫▫'),
        (0.421, u'⣿⣿⣿⣿⡀▫▫▫▫▫'),
        (0.431, u'⣿⣿⣿⣿⣀▫▫▫▫▫'),
        (0.441, u'⣿⣿⣿⣿⣄▫▫▫▫▫'),
        (0.451, u'⣿⣿⣿⣿⣤▫▫▫▫▫'),
        (0.461, u'⣿⣿⣿⣿⣦▫▫▫▫▫'),
        (0.471, u'⣿⣿⣿⣿⣶▫▫▫▫▫'),
        (0.481, u'⣿⣿⣿⣿⣷▫▫▫▫▫'),
        (0.491, u'⣿⣿⣿⣿⣿▫▫▫▫▫'),
        (0.501, u'⣿⣿⣿⣿⣿▫▫▫▫▫'),
    ]
)
def test_bar(progress, result):
    assert bar(progress) == result


def test_loop():
    assert loop_test(range(10), lambda x: x) == [
        '[⣿▫▫▫▫▫▫▫▫▫ 1.00s / 10.00s]',
        '[⣿⣿▫▫▫▫▫▫▫▫ 2.00s / 10.00s]',
        '[⣿⣿⣿▫▫▫▫▫▫▫ 3.00s / 10.00s]',
        '[⣿⣿⣿⣿▫▫▫▫▫▫ 4.00s / 10.00s]',
        '[⣿⣿⣿⣿⣿▫▫▫▫▫ 5.00s / 10.00s]',
        '[⣿⣿⣿⣿⣿⣿▫▫▫▫ 6.00s / 10.00s]',
        '[⣿⣿⣿⣿⣿⣿⣿▫▫▫ 7.00s / 10.00s]',
        '[⣿⣿⣿⣿⣿⣿⣿⣿▫▫ 8.00s / 10.00s]',
        '[⣿⣿⣿⣿⣿⣿⣿⣿⣿▫ 9.00s / 10.00s]',
        '[⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿ 10.00s / 10.00s]',
        '[done. took 10.00s]',
    ]


def test_loop_no_length():
    assert loop_test(iter(range(5)), lambda x: x) == [
        r'[- 1.00s / ?]',
        r'[\ 2.00s / ?]',
        r'[| 3.00s / ?]',
        r'[/ 4.00s / ?]',
        r'[- 5.00s / ?]',
        r'[done. took 5.00s]',
    ]


def test_loop_exception():
    def raise_(i):
        raise ValueError

    assert loop_test(range(10), raise_) == [
        '[aborted. took 1.00s]',
    ]


def loop_test(iterable, action):
    t = 0

    def time():
        nonlocal t
        return t

    loop = Loop(time=time)
    result = []

    try:
        for i in loop(iterable):
            t = i + 1
            action(i)
            result += ['{loop}'.format(loop=loop)]

    except ValueError:
        pass

    result += ['{loop}'.format(loop=loop)]

    return result
