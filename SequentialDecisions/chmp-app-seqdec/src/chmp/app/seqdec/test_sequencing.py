import asyncio
import random

import pytest

from chmp.app.seqdec.sequencing import run, find_blocks, build_sequence

# TODO: document the test
#
run_examples = [
    dict(
        label='simple',
        sequence=[dict(message='foo', duration=3)],
        events=[
            [('say', 'foo', 0)],
            [('wait', 2, 1), (None, 2)],
            [('end', None, 3)]
        ],
    ),
    dict(
        label='multiple',
        sequence=[dict(message='1', duration=3), dict(message='2', duration=2)],
        events=[
            [('say', '1', 0)],
            [('wait', 2, 1), (None, 2)],
            [('say', '2', 3)],
            [('wait', 1, 4), (None, 1)],
            [('end', None, 5)],
        ],
    ),
    dict(
        label='multiple-continue',
        sequence=[dict(message='1', duration=3), dict(message='2', duration=2)],
        events=[
            [('say', '1', 0)],
            [('wait', 2, 1), ('continue', 1)],
            [('command', 'continue', 2)],
            [('say', '2', 2)],
            [('wait', 1, 3), (None, 1)],
            [('end', None, 4)],
        ],
    ),
    dict(
        label='multiple-stop',
        sequence=[dict(message='1', duration=3), dict(message='2', duration=2)],
        events=[
            [('say', '1', 0)],
            [('wait', 2, 1), ('stop', 1)],
            [('command', 'stop', 2)],
            [('end', None, 2)],
        ],
    ),
    dict(
        label='multiple-wait',
        sequence=[dict(message='1', duration=3), dict(message='2', duration=2)],
        events=[
            [('say', '1', 0)],
            [('wait', 2, 1), (None, 2)],
            [('say', '2', 3)],
            [('wait', 1, 4), ('wait', 0)],
            [('command', 'wait', 4)],
            [('say', 'Okay', 4)],
            [('wait', 9, 5), (None, 9)],
            [('say', '2', 14)],
            [('wait', 1, 15), (None, 1)],
            [('end', None, 16)],
        ],
    ),
    dict(
        label='multiple-wait-wait',
        sequence=[dict(message='1', duration=3), dict(message='2', duration=2)],
        events=[
            [('say', '1', 0)],
            [('wait', 2, 1), (None, 2)],
            [('say', '2', 3)],
            [('wait', 1, 4), ('wait', 0)],
            [('command', 'wait', 4)],
            [('say', 'Okay', 4)],
            [('wait', 9, 5), ('wait', 5)],
            [('command', 'wait', 10)],
            [('say', 'Okay', 10)],
            [('wait', 9, 11), (None, 9)],
            [('say', '2', 20)],
            [('wait', 1, 21), (None, 1)],
            [('end', None, 22)],
        ],
    ),
    dict(
        label='multiple-continue-wait',
        sequence=[dict(message='1', duration=3), dict(message='2', duration=2)],
        events=[
            [('say', '1', 0)],
            [('wait', 2, 1), ('continue', 1)],
            [('command', 'continue', 2)],
            [('say', '2', 2)],
            [('wait', 1, 3), ('wait', 0)],
            [('command', 'wait', 3)],
            [('say', 'Okay', 3)],
            [('wait', 9, 4), (None, 9)],
            [('say', '2', 13)],
            [('wait', 1, 14), (None, 1)],
            [('end', None, 15)],
        ],
    ),
    dict(
        label='multiple-unknown-command',
        sequence=[dict(message='1', duration=3), dict(message='2', duration=2)],
        events=[
            [('say', '1', 0)],
            [('wait', 2, 1), ('noise', 1)],
            [('command', 'noise', 2)],
            [('wait', 1, 2), (None, 1)],
            [('say', '2', 3)],
            [('wait', 1, 4), (None, 1)],
            [('end', None, 5)],
        ],
    ),
]


@pytest.mark.parametrize('spec', run_examples, ids=lambda spec: spec['label'])
def test__run__examples(spec):
    sequence = list(spec['sequence'])
    command_time_pairs = [l[1] for l in spec['events'] if len(l) == 2]
    expected_events = [event for event, *_ in spec['events']]
    expected_actions = [event[1] for event in expected_events if event[0] in {'say', 'wait'}]

    actions = []

    # arbitrary start time
    current_time = random.randint(0, 10)

    async def next_command(timeout=0):
        nonlocal current_time, command_time_pairs

        (command, delta), *command_time_pairs = command_time_pairs
        current_time += delta
        actions.append(timeout)
        return command

    def time():
        return current_time

    # NOTE: speaking always takes one second
    async def say(message):
        nonlocal current_time
        current_time += 1
        actions.append(message)

    loop = asyncio.get_event_loop()
    events = loop.run_until_complete(run(
        sequence,
        next_command=next_command,
        time=time,
        say=say,
        wait_command_time=10,
    ))

    assert events == expected_events
    assert actions == expected_actions


def test_find_blocks():
    assert find_blocks([False, False, False]) == []
    assert find_blocks([False, True, False]) == [(1, 2)]
    assert find_blocks([True, True, False]) == [(0, 2)]
    assert find_blocks([True, True, True]) == [(0, 3)]
    assert find_blocks([False, False, True]) == [(2, 3)]
    assert find_blocks([True, False, True]) == [(0, 1), (2, 3)]


build_sequence_examples = [
    dict(
        label='all-symmetric',
        sequence=[
            dict(name='1'),
            dict(name='2'),
            dict(name='3'),
        ],
        expected=[
            dict(name='1'),
            dict(name='2'),
            dict(name='3'),
        ],
    ),
    dict(
        label='asymmetric-start',
        sequence=[
            dict(name='1', asymmetric=True),
            dict(name='2'),
            dict(name='3'),
        ],
        expected=[
            dict(name='1', asymmetric=True, side='left'),
            dict(name='1', asymmetric=True, side='right'),
            dict(name='2'),
            dict(name='3'),
        ],
    ),
    dict(
        label='asymmetric-middle',
        sequence=[
            dict(name='1'),
            dict(name='2', asymmetric=True),
            dict(name='3'),
        ],
        expected=[
            dict(name='1'),
            dict(name='2', asymmetric=True, side='left'),
            dict(name='1'),
            dict(name='2', asymmetric=True, side='right'),
            dict(name='3'),
        ],
    ),
    dict(
        label='asymmetric-end',
        sequence=[
            dict(name='1'),
            dict(name='2'),
            dict(name='3', asymmetric=True),
        ],
        expected=[
            dict(name='1'),
            dict(name='2'),
            dict(name='3', asymmetric=True, side='left'),
            dict(name='2'),
            dict(name='3', asymmetric=True, side='right'),
        ],
    ),
    dict(
        label='asymmetric-start-end',
        sequence=[
            dict(name='1', asymmetric=True),
            dict(name='2'),
            dict(name='3', asymmetric=True),
        ],
        expected=[
            dict(name='1', asymmetric=True, side='left'),
            dict(name='1', asymmetric=True, side='right'),
            dict(name='2'),
            dict(name='3', asymmetric=True, side='left'),
            dict(name='2'),
            dict(name='3', asymmetric=True, side='right'),
        ],
    ),
    dict(
        label='asymmetric-middle-end',
        sequence=[
            dict(name='1'),
            dict(name='2', asymmetric=True),
            dict(name='3', asymmetric=True),
        ],
        expected=[
            dict(name='1'),
            dict(name='2', asymmetric=True, side='left'),
            dict(name='3', asymmetric=True, side='left'),
            dict(name='1'),
            dict(name='2', asymmetric=True, side='right'),
            dict(name='3', asymmetric=True, side='right'),
        ],
    ),
    dict(
        label='asymmetric-full',
        sequence=[
            dict(name='1', asymmetric=True),
            dict(name='2', asymmetric=True),
            dict(name='3', asymmetric=True),
        ],
        expected=[
            dict(name='1', asymmetric=True, side='left'),
            dict(name='2', asymmetric=True, side='left'),
            dict(name='3', asymmetric=True, side='left'),
            dict(name='1', asymmetric=True, side='right'),
            dict(name='2', asymmetric=True, side='right'),
            dict(name='3', asymmetric=True, side='right'),
        ],
    ),
]


@pytest.mark.parametrize('spec', build_sequence_examples, ids=lambda spec: spec['label'])
def test_build_sequence(spec):
    expected = spec['expected']
    actual = build_sequence(spec['sequence'], lambda: ('left', 'right'))

    assert actual == expected
