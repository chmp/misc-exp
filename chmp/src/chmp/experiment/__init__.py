import enum
import hashlib
import itertools as it
import json
import math
import time

maximum_15_digit_hex = float(0xFFF_FFFF_FFFF_FFFF)
max_32_bit_integer = 0xFFFF_FFFF


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
    return max(sha1(obj) / maximum_15_digit_hex, 0.9999999999999999)


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


def loop(iterable, length=None):
    """Add a progressbar without an explicit print statement.

    Usage::

        for item in loop(values):
            ...

    """
    loop = Loop()

    for item in loop(iterable, length=length):
        yield item
        print('{}'.format(loop), end='\r')


class Loop(object):
    """Helper to track the status of a long-running loops.

    Usage::

        loop = Loop()
        for item in loop(iterable):
            ...
            print(f'{loop}'.ljust(120), end='\r')

    """
    def __init__(self, time=time.time):
        self.now = time
        self.state = LoopState.pending
        self.idx = 0
        self.start = 0
        self.length = None
        self.expected = None

    def __call__(self, iterable, length=None):
        if length is None:
            try:
                self.length = len(iterable)

            except TypeError:
                self.length = None

        else:
            self.length = int(length)

        self.state = LoopState.running
        self.start = self.now()
        for self.idx, item in enumerate(iterable):
            try:
                yield item

            except GeneratorExit:
                # NOTE: this is reached, when the generator is not fully consumed
                self.state = LoopState.aborted
                raise

            self.expected = self._compute_expected()

        self.state = LoopState.done

    def __format__(self, format_spec):
        status = self.get_status()

        if status['state'] is LoopState.pending:
            return '[pending]'

        elif status['state'] is LoopState.running:
            return f'[{status["bar"]} {tdformat(status["total"])} / {tdformat(status["expected"])}]'

        elif status['state'] is LoopState.aborted:
            return f'[aborted. took {tdformat(status["total"])}]'

        elif status['state'] is LoopState.done:
            return f'[done. took {tdformat(status["total"])}]'

    def get_status(self):
        total = self.now() - self.start

        if self.length is None:
            return dict(
                state=self.state,
                total=total,
                expected=None,
                bar=running_characters[self.idx % len(running_characters)],
                fraction=math.nan,
            )

        fraction = (self.idx + 1) / self.length

        return dict(
            state=self.state,
            total=total,
            expected=self._get_expected(total=total),
            bar=bar(fraction),
            fraction=fraction,
        )

    def _get_expected(self, total=None):
        if self.expected is None:
            return self._compute_expected(total=total)

        return self.expected

    def _compute_expected(self, total=None):
        if self.length is None or self.start is None:
            return None

        if total is None:
            total = self.now() - self.start

        return total / (self.idx + 1) * self.length


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
