"""Helper for running ML experiments.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
import argparse
import collections
import enum
import functools as ft
import hashlib
import inspect
import itertools as it
import json
import logging
import math
import os.path
import time
import typing
import uuid

maximum_15_digit_hex = float(0xFFF_FFFF_FFFF_FFFF)
max_32_bit_integer = 0xFFFF_FFFF

_logger = logging.getLogger()


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
# #                             Helpers                                     # #
# #                                                                         # #
# ########################################################################### #


def experiment(config_class):
    def main(func, args=None):
        parser = build_parser(config_class)
        args = parser.parse_args(args)

        kwargs = {
            k: v
            for k, v in inspect.getmembers(args)
            if not k.startswith('_')
        }
        return run(func, **kwargs)

    def run(func, *, base_path='.', experiment=None, config=None, **kwargs):
        if config is None:
            config = {}

        experiment, path = ensure_experiment(base_path, experiment)

        if isinstance(config, collections.Mapping):
            config = config_class.from_dict(config)

        return func(experiment=experiment, path=path, config=config, **kwargs)

    def decorator(func):
        func.main = ft.partial(main, func)
        func.run = ft.partial(run, func)
        return func

    return decorator


def ensure_experiment(base_path='.', name=None):
    """Return a valid experiment id and directory.

    :param str base_path:
    :param Optional[str] name:
    :rtype: Tuple[str,str]
    :return:

    Usage::

        experiment, path = ensure_experiment('./run')

    """
    base_path = os.path.abspath(base_path)

    if name is not None:

        path = os.path.join(base_path, name)
        if os.path.exists(path):
            _logger.warning('reusing existing result path %s', path)

        else:
            os.makedirs(path)
            logging.info('created experiment %s at %s', name, path)

        return name, os.path.join(base_path, name)

    else:
        while True:
            name = str(uuid.uuid4())

            try:
                os.mkdir(os.path.join(base_path, name))

            except FileExistsError:
                pass

            else:
                path = os.path.join(base_path, name)
                logging.info('created experiment %s at %s', name, path)
                return name, path


def build_namespace(d, result=None):
    if result is None:
        result = argparse.Namespace()

    for k, v in d.items():
        if isinstance(v, collections.Mapping):
            v = build_namespace(v)

        setattr(result, k, v)

    return result


def build_dict(ns):
    result = {}

    for k, v in inspect.getmembers(ns):
        if k.startswith('_'):
            continue

        if isinstance(v, argparse.Namespace):
            v = build_dict(v)

        result[k] = v

    return result


# ########################################################################### #
# #                                                                         # #
# #                       Commandline Parsers                               # #
# #                                                                         # #
# ########################################################################### #


def build_parser(cls):
    parser = ConfigParser(config=cls)
    arguments = {
        path: arg
        for path, arg in _get_arguments((), '--', cls)
    }
    # add config options
    for path, arg in arguments.items():
        parser.add_argument(arg, action=BuildNodeAction, path=path, default=argparse.SUPPRESS)

    # add default switches
    parser.add_argument('--config', dest='_config', help='if given, load config from this file')
    parser.add_argument('--experiment', default=argparse.SUPPRESS)
    parser.add_argument('base_path', default=argparse.SUPPRESS)

    return parser


def _get_arguments(path, prefix, cls):
    annotations = get_annotations(cls)

    for k, v in annotations.items():
        if k.startswith('_'):
            continue

        k = k.replace('_', '-')
        if isinstance(v, Case):
            for _, t in v.when_clauses:
                yield from _get_arguments(
                    path=path + (k,),
                    prefix=prefix + str(k) + '-',
                    cls=t,
                )

        else:
            yield path + (k,), prefix + str(k)


class BuildNodeAction(argparse.Action):
    def __init__(self, *, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def __call__(self, parser, namespace, value, option_string=None):
        if not hasattr(namespace, 'config'):
            namespace.config = argparse.Namespace()

        current = namespace.config
        path = self.path

        while True:
            if len(path) == 0:
                raise RuntimeError()

            elif len(path) == 1:
                break

            else:
                head, *path = path

                if not hasattr(current, head):
                    setattr(current, head, argparse.Namespace())

                current = getattr(current, head)

        setattr(current, path[0], value)


class ConfigParser(argparse.ArgumentParser):
    def __init__(self, *, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args, namespace)

        if not hasattr(args, 'config'):
            args.config = argparse.Namespace()

        if hasattr(args, '_config') and args._config:
            raise NotImplementedError('config loading not yet implemented')

        enforce(self.config, args.config)

        return args


def enforce(cls, ns):
    annotations = get_annotations(cls)

    for k, spec in annotations.items():

        if k.startswith('_'):
            continue

        elif isinstance(spec, Case):
            # handle later
            continue

        elif isinstance(spec, type) and issubclass(spec, Config):
            if not hasattr(ns, k):
                setattr(ns, k, argparse.Namespace())

            enforce(spec, getattr(ns, k))

        elif not hasattr(ns, k):
            setattr(ns, k, getattr(cls, k))

        # TODO: figure out what's the correct test here
        elif type(spec) is typing._Union:
            first_subtype, second_subtype = spec.__args__

            if first_subtype is type(None):
                pass

            elif second_subtype is type(None):
                first_subtype, second_subtype = second_subtype, first_subtype

            else:
                raise ValueError('can only handle Union with NoneType, i.e., Optional')

            value = getattr(ns, k)
            if value is not None:
                value = second_subtype(value)
                setattr(ns, k, value)

        elif isinstance(spec, type) and issubclass(spec, typing.Tuple):
            subtype, = spec.__args__
            value = getattr(ns, k)
            value = tuple(subtype(item) for item in value.split(':'))
            setattr(ns, k, value)

        elif isinstance(spec, type):
            value = getattr(ns, k)
            value = spec(value)
            setattr(ns, k, value)

        else:
            raise RuntimeError('unknown annotation {}: {}'.format(k, spec))

    for k, spec in annotations.items():
        if not isinstance(spec, Case):
            continue

        value = getattr(ns, spec.var_name)

        for needle, config in spec.when_clauses:
            if value == needle:
                if not hasattr(ns, k):
                    setattr(ns, k, argparse.Namespace())

                enforce(config, getattr(ns, k))
                break

        else:
            raise ValueError('Unknown value {!r} for {}'.format(value, k))

    cls.check(ns)


def get_annotations(cls):
    """Walk the MRO and pickup all annotations.
    """
    annotations = {}
    for base in reversed(inspect.getmro(cls)):
        base_annotations = getattr(base, '__annotations__', {})
        annotations.update(base_annotations)

    return annotations


# ########################################################################### #
# #                                                                         # #
# #                         Experiment Specs                                # #
# #                                                                         # #
# ########################################################################### #


class SelfNamedEnum(str, enum.Enum):
    # copied from the docs
    def _generate_next_value_(name, start, count, last_values):
        return name


class ClassGetItemMeta(type):
    def __getitem__(cls, k):
        return cls.getitem(k)


class ClassGetItem(metaclass=ClassGetItemMeta):
    @classmethod
    def getitem(cls, k):
        raise NotImplementedError()


class Case(ClassGetItem):
    def __init__(self, var_name, when_clauses):
        self.var_name = var_name
        self.when_clauses = when_clauses

    @classmethod
    def getitem(cls, keys):
        if not isinstance(keys, tuple):
            keys = keys,

        var_name, *when_clauses = keys

        return cls(var_name, when_clauses)

    def __repr__(self):
        parts = [str(self.var_name)]
        parts += [repr(clause) for clause in self.when_clauses]
        return 'Case[' + ', '.join(parts) + ']'


class When(ClassGetItem):
    def __init__(self, value, result):
        self.value = value
        self.result = result

    @classmethod
    def getitem(cls, keys):
        value, result = keys
        return cls(value, result)

    def __repr__(self):
        return 'When[{!r}, {!r}]'.format(self.value, self.result)

    def __iter__(self):
        return iter([self.value, self.result])


class Config:
    def __init__(self, __d=None, **kwargs):
        if __d is None:
            __d = {}

        __d.update(kwargs)

        build_namespace(__d, result=self)
        enforce(type(self), self)

    def to_dict(self):
        return build_dict(self)

    @classmethod
    def check(cls, d):
        pass


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
