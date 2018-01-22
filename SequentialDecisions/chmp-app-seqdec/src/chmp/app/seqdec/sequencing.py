"""Support in creating sequences.
"""
import itertools as it
import logging
import random
import time

_logger = logging.getLogger(__name__)


async def run(sequence, *, say, next_command, time=time.time, wait_command_time=10):
    """Run a sequence of steps while listening for user commands.

    The following commands are supported:

    - ``continue``: continue with the next step
    - ``wait``: return to the previous exercise and wait
    - ``stop`` abort the exercise

    .. todo::

        Add ``explain``, explain the current exercise.

    """
    events = []
    start = time()

    sequence = PushIterator(sequence)
    last_waitable_item = None
    for item in sequence:
        until = time() + item['duration']

        if item.get('reason') != 'wait':
            last_waitable_item = item

        if item.get('message') is not None:
            events.append(('say', item['message'], time() - start))
            await say(item['message'])

        wait_time = max(0, until - time())
        events.append(('wait', wait_time, time() - start))
        command = await next_command(timeout=wait_time)

        if command is None:
            continue

        events.append(('command', command, time() - start))

        if command == 'continue':
            pass

        elif command == 'stop':
            # TODO: wait for ... and allow to continue?
            break

        elif command == 'wait':
            # only wait once for each item
            if last_waitable_item is not None:
                sequence.push(last_waitable_item)
                last_waitable_item = None

            sequence.push(dict(message='Okay', duration=wait_command_time, reason='wait'))

        else:
            # ignore command and wait for the remaining time
            _logger.info('unknown command %s', command)
            sequence.push(dict(message=None, duration=max(0, until - time()), reason='unknown'))

    events.append(('end', None, time() - start))
    return events


class PushIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.priority = []

    def push(self, item):
        self.priority.append(item)

    def __iter__(self):
        for item in self.iterable:
            while self.priority:
                yield self.priority.pop()

            yield item

        while self.priority:
            yield self.priority.pop()


def _random_side_order():
    return random.choice([('left', 'right'), ('right', 'left')])


def build_sequence(sequence, side_order=_random_side_order):
    result = []

    current = 0

    for start, end in find_blocks(p.get('asymmetric', False) for p in sequence):
        if start != 0:
            result.extend(dict(p) for p in sequence[current:start - 1])

        for side in side_order():
            if start != 0:
                result.append(dict(sequence[start -1]))
            result.extend(dict(p, side=side) for p in sequence[start:end])

        current = end

    result.extend(dict(p) for p in sequence[current:])

    return result


def find_blocks(sequence):
    """Find all block of asymmetric steps and any directly preceding step.

    :param Iterable[bool] sequence:
        a sequence of flags

    :returns:
        a sequence of ``(first-step-before-block, first-step-after-block)``
        pairs.
    """
    blocks = []
    in_block = False

    for idx, current in enumerate(it.chain(sequence, [False])):
        if not in_block and current:
            in_block = True
            blocks.append((idx, idx))

        elif in_block and not current:
            in_block = False
            blocks[-1] = (blocks[-1][0], idx)

    return blocks
