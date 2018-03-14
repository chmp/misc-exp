"""Helper for running ML experiments.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import enum
import hashlib
import itertools as it
import json
import logging
import math
import time

maximum_15_digit_hex = float(0xFFF_FFFF_FFFF_FFFF)
max_32_bit_integer = 0xFFFF_FFFF

_logger = logging.getLogger()


# ###################################################################### #
# #                                                                    # #
# #                 Deterministic Random Number Generation             # #
# #                                                                    # #
# ###################################################################### #

def sha1(obj):
    """Create a hash for a json-encode-able object
    """
    return int(str_sha1(obj)[:15], 16)


def str_sha1(obj):
    s = json.dumps(obj, indent=None, sort_keys=True, separators=(',', ':'))
    s = s.encode('utf8')
    return hashlib.sha1(s).hexdigest()


def random(obj):
    """Return a random float in the range [0, 1)"""
    return min(sha1(obj) / maximum_15_digit_hex, 0.9999999999999999)


def uniform(obj, a, b):
    return a + (b - a) * random(obj)


def randrange(obj, *range_args):
    r = range(*range_args)
    # works up to a len of 9007199254749999, rounds down afterwards
    i = int(random(obj) * len(r))
    return r[i]


def randint(obj, a, b):
    return randrange(obj, a, b + 1)


def np_seed(obj):
    """Return a seed usable by numpy.
    """
    return [randrange((obj, i), max_32_bit_integer) for i in range(10)]


def tf_seed(obj):
    """Return a seed usable by tensorflow.
    """
    return randrange(obj, max_32_bit_integer)


def std_seed(obj):
    """Return a seed usable by python random module.
    """
    return str_sha1(obj)


def shuffled(obj, l):
    l = list(l)
    shuffle(obj, l)
    return l


def shuffle(obj, l):
    """Shuffle `l` in place using Fisher–Yates algorithm.

    See: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    """
    n = len(l)
    for i in range(n - 1):
        j = randrange((obj, i), i, n)
        l[i], l[j] = l[j], l[i]


# ########################################################################### #
# #                                                                         # #
# #                            Loops                                        # #
# #                                                                         # #
# ########################################################################### #

status_characters = it.accumulate([64, 128, 4, 32, 2, 16, 1, 8])
status_characters = [chr(ord('\u2800') + v) for v in status_characters]
status_characters = ['\u25AB', ' '] + status_characters

running_characters = ['-', '\\', '|', '/']


class LoopState(enum.Enum):
    pending = 'pending'
    running = 'running'
    done = 'done'
    aborted = 'aborted'


class Loop:
    """Helper to track the status of a long-running loops.

    Usage::

        loop = Loop()
        for item in loop(iterable):
            # computationally intensive work
            ...
            print(f'{loop}'.ljust(120), end='\r')

    Nested loops::

        outer_loop = Loop()
        loop = Loop()

        for i in outer_loop(iterable1):
            for j in loop(iterable2):
                print(f'{outer_loop} {loop}'.ljust(120), end='\r')

    The loop object can be formatted by using a format string. For example::

        print(f'{loop:fr}'.ljust(120), end='\r')

    prints the fraction of work completed and the remaining time expected to
    finish all work.

    The following options are understood:

    * ``b``: a progress bar
    * ``t``: total time taken so far
    * ``e``: expected time for all work
    * ``r``: remaining time
    * ``f``: fraction of work completed

    The default format is ``[bt/e`` for bar, total, /, expected sourrounded by
    brackets. To print the work done and the remaining time in brackets use
    ``[fr``.
    """
    def __init__(self, time=time.time):
        self._now = time
        self._state = LoopState.pending
        self._idx = 0
        self._start = 0
        self._length = None
        self._expected = None

    @classmethod
    def over(cls, iterable, length=None, time=time.time):
        loop = cls(time)

        for item in loop(iterable, length=length):
            yield loop, item

    def __call__(self, iterable, length=None):
        if length is None:
            try:
                self._length = len(iterable)

            except TypeError:
                self._length = None

        else:
            self._length = int(length)

        self._state = LoopState.running
        self._start = self._now()
        for self._idx, item in enumerate(iterable):
            try:
                yield item

            except GeneratorExit:
                # NOTE: this is reached, when the generator is not fully consumed
                self._state = LoopState.aborted
                raise

            self._expected = self._compute_expected()

        self._state = LoopState.done

    def __format__(self, format_spec):
        status = self.get_status()

        if status['state'] is LoopState.pending:
            return '[pending]'

        elif status['state'] is LoopState.aborted:
            return f'[aborted. took {tdformat(status["total"])}]'

        elif status['state'] is LoopState.done:
            return f'[done. took {tdformat(status["total"])}]'

        elif status['state'] is not LoopState.running:
            raise RuntimeError('unknown state')

        if not format_spec:
            format_spec = '[bt/e'

        if format_spec[:1] == '[':
            outer = '[', ']'
            format_spec = format_spec[1:]

        else:
            outer = '', ''

        if format_spec[:1] == '-':
            join_char = ''
            format_spec = format_spec[1:]

        else:
            join_char = ' '

        result = [
            self._loop_formats.get(c, lambda _: c)(status)
            for c in format_spec
        ]
        return outer[0] + join_char.join(result) + outer[1]

    _loop_formats = {
        'b': lambda status: status['bar'],
        't': lambda status: tdformat(status["total"]),
        'e': lambda status: tdformat(status["expected"]),
        'r': lambda status: tdformat(status["remaining"]),
        'f': lambda status: f"{status['fraction']:.1%}",
    }

    def get_status(self):
        total = self._now() - self._start

        if self._length is None:
            return dict(
                state=self._state,
                total=total,
                expected=None,
                remaining=None,
                bar=running_characters[self._idx % len(running_characters)],
                fraction=math.nan,
            )

        fraction = (self._idx + 1) / self._length

        d = dict(
            state=self._state,
            total=total,
            expected=self._get_expected(total=total),
            bar=bar(fraction),
            fraction=fraction,
        )
        d['remaining'] = d['expected'] - d['total']
        return d

    def _get_expected(self, total=None):
        if self._expected is None:
            return self._compute_expected(total=total)

        return self._expected

    def _compute_expected(self, total=None):
        if self._length is None or self._start is None:
            return None

        if total is None:
            total = self._now() - self._start

        return total / (self._idx + 1) * self._length


def tdformat(time_delta):
    """Format a timedelta given in seconds.
    """
    if time_delta is None:
        return '?'

    # TODO: handle negative differences?
    time_delta = abs(time_delta)

    d = dict(
        weeks=int(time_delta // (7 * 24 * 60 * 60)),
        days=int(time_delta % (7 * 24 * 60 * 60) // (24 * 60 * 60)),
        hours=int(time_delta % (24 * 60 * 60) // (60 * 60)),
        minutes=int(time_delta % (60 * 60) // 60),
        seconds=time_delta % 60,
    )

    if d['weeks'] > 0:
        return '{weeks}w {days}d'.format(**d)

    elif d['days'] > 0:
        return '{days}d {hours}h'.format(**d)

    elif d['hours'] > 0:
        return '{hours}h {minutes}m'.format(**d)

    elif d['minutes'] > 0:
        return '{minutes}m {seconds:.0f}s'.format(**d)

    else:
        return '{seconds:.2f}s'.format(**d)


def bar(u, n=10):
    """Format a ASCII progressbar"""
    u = max(0.00, min(0.99, u))

    done = int((n * u) // 1)
    rest = max(0, n - done - 1)

    c = int(((n * u) % 1) * len(status_characters))
    return status_characters[-1] * done + status_characters[c] + status_characters[0] * rest
