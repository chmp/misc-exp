import pytest

from chmp.ds import (
    Loop,
    LoopState,

    ascii_bar,
    tdformat,
)


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
    assert ascii_bar(progress) == result


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
    result = []

    try:
        for loop, i in Loop.over(iterable, time=MockTime()):
            loop.now.time = i + 1
            action(i)
            result += ['{loop}'.format(loop=loop)]

    except ValueError:
        pass

    result += ['{loop}'.format(loop=loop)]

    return result


expected_states = [
    dict(expected=2, fraction=0.125, state=LoopState.running, total=0.25),
    dict(expected=2, fraction=0.250, state=LoopState.running, total=0.50),
    dict(expected=2, fraction=0.375, state=LoopState.running, total=0.75),
    dict(expected=2, fraction=0.500, state=LoopState.running, total=1.00),
    dict(expected=2, fraction=0.625, state=LoopState.running, total=1.25),
    dict(expected=2, fraction=0.750, state=LoopState.running, total=1.50),
    dict(expected=2, fraction=0.875, state=LoopState.running, total=1.75),
    dict(expected=2, fraction=1.000, state=LoopState.running, total=2.00),
    dict(expected=2, fraction=1.000, state=LoopState.done, total=2.00),
]


def test_single():
    states = []

    for loop, _ in Loop.over(range(8), time=MockTime()):
        loop.now.time += 0.25
        states += [loop.get_info()]

    states += [loop.get_info()]

    for s in states:
        s.pop('idx')

    assert states == expected_states


def test_nested():
    states = []

    for loop, _ in Loop.over(range(2), time=MockTime()):
        for _ in loop.nest(range(4)):
            loop.now.time += 0.25
            states += [loop.get_info()]

    states += [loop.get_info()]

    for s in states:
        s.pop('idx')

    assert states == expected_states


def test_nested_nested():
    states = []

    for loop, _ in Loop.over(range(2), time=MockTime()):
        for _ in loop.nest(range(2)):
            for _ in loop.nest(range(2)):
                loop.now.time += 0.25
                states += [loop.get_info()]

    states += [loop.get_info()]

    for s in states:
        s.pop('idx')

    assert states == expected_states


class MockTime:
    def __init__(self):
        self.time = 0

    def __call__(self):
        return self.time
