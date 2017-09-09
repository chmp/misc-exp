"""Helpers for distributed data processing.

Distributed as part of ``https://github.com/chmp/misc-exp`` under the MIT
license, (c) 2017 Christopher Prohm.
"""
from __future__ import print_function, division, absolute_import

import collections
import itertools as it
import math

import toolz
from toolz import curry, curried

__all__ = [
    'DaskCounter',
    'DaskDict',
    'RuleSet',

    'apply',
    'chained',
    'extension_rules',
    'mean',
    'repartition',
    'rules',
    'reduction_rules',
    'std',
    'stdlib_rules',
    'toolz_rules',
    'var',
]


def apply(transformation, obj):
    """Apply the given transformation to a ``dask.bag.Bag``."""
    return rules(obj, transformation)


class RuleSet(object):
    def __init__(self, rules=()):
        self.rules = list(rules)

    def add(self, match, apply, help=None):
        self.rules.append(dict(match=match, apply=apply, help=help))

    def __or__(self, other):
        return RuleSet(self.rules + other.rules)

    def __call__(self, *args):
        for rule in self.rules:
            match = rule['match']
            does_match = match(self, *args)

            if does_match:
                apply = rule['apply']
                return apply(self, *args)

        raise RuntimeWarning('no matching rule')


def eq(obj):
    if isinstance(obj, curry):
        return lambda _, other, __: isinstance(other, curry) and other.func == obj.func

    return lambda _, other, __: other == obj


def a(t):
    return lambda _, other, __: isinstance(other, t)


def _raise(exc, *args, **kwargs):
    def impl(*_, **__):
        raise exc(*args, **kwargs)

    return impl


class DaskDict(object):
    def __init__(self, items):
        if isinstance(items, DaskDict):
            items = items._items

        self._items = items

    def items(self):
        return self._items

    def keys(self):
        return self._items.map(lambda t: t[0])

    def values(self):
        return self._items.map(lambda t: t[1])

    def copy(self):
        "dask bags are immutable"
        return self

    def compute(self, **kwargs):
        return dict(self._items.compute(**kwargs))


class DaskCounter(DaskDict):
    def _call(self, func):
        def impl(items):
            counter = collections.Counter()
            for k, v in items:
                counter[k] = v
            return func(counter)

        return self._items.repartition(1).map_partitions(impl)

    def elements(self):
        return self._call(lambda obj: list(obj.elements()))

    def most_common(self, n):
        return self._call(lambda obj: list(obj.most_common(n)))


# ########################################################################### #
# #                                                                         # #
# #                               Stdlib                                    # #
# #                                                                         # #
# ########################################################################### #


stdlib_rules = RuleSet()

stdlib_rules.add(eq(any), lambda _, __, obj: obj.any())
stdlib_rules.add(eq(all), lambda _, __, obj: obj.all())
stdlib_rules.add(eq(min), lambda _, __, obj: obj.min())
stdlib_rules.add(eq(max), lambda _, __, obj: obj.max())
stdlib_rules.add(eq(sum), lambda _, __, obj: obj.sum())
stdlib_rules.add(eq(len), lambda _, __, obj: obj.count())
stdlib_rules.add(eq(set), lambda _, __, obj: obj.distinct())
stdlib_rules.add(eq(list), lambda _, __, obj: obj)
stdlib_rules.add(eq(dict), lambda _, __, obj: DaskDict(obj))
stdlib_rules.add(eq(dict.items), lambda _, __, obj: DaskDict(obj).items())
stdlib_rules.add(eq(it.chain.from_iterable), lambda _, __, obj: obj.flatten())
stdlib_rules.add(eq(collections.Counter), lambda _, __, obj: DaskCounter(obj.frequencies()))

# TODO: add warning that this is inefficient for large collections
stdlib_rules.add(eq(sorted), lambda _, __, obj: obj.repartition(1).map_partitions(sorted))


# ########################################################################### #
# #                                                                         # #
# #                               Toolz                                     # #
# #                                                                         # #
# ########################################################################### #


def _map_items(impl):
    "impl: func(f, k, v)"
    return lambda _, f, o: DaskDict(DaskDict(o).items().map(lambda t: impl(f.args[0], t[0], t[1])))


def _filter_items(impl):
    "impl: func(f, k, v)"
    return lambda _, f, o: DaskDict(DaskDict(o).items().filter(lambda t: impl(f.args[0], t[0], t[1])))


def _toolz_item_filter(_, func, obj):
    return DaskDict(DaskDict(obj).items().filter(func.args[0]))


def _toolz_join(_, func, obj):
    # between toolz and dask the roles between left / right are reversed
    def impl(leftkey, leftseq, rightkey):
        return obj.join(leftseq, on_self=rightkey, on_other=leftkey)

    return impl(*func.args, **func.keywords)


def _call(impl, f):
    return impl(*f.args, **f.keywords)


toolz_rules = RuleSet()
toolz_rules.add(eq(curried.map), lambda _, f, o: _call(o.map, f))
toolz_rules.add(eq(curried.filter), lambda _, f, o: _call(o.filter, f))
toolz_rules.add(eq(curried.pluck), lambda _, f, o: _call(o.pluck, f))
toolz_rules.add(eq(curried.join), _toolz_join)
toolz_rules.add(eq(curried.countby), lambda _, f, o: DaskDict(_call(o.map, f).frequencies()))
toolz_rules.add(eq(curried.groupby), lambda _, f, o: DaskDict(_call(o.groupby, f)))
toolz_rules.add(eq(curried.keymap), _map_items(lambda f, k, v: (f(k), v)))
toolz_rules.add(eq(curried.valmap), _map_items(lambda f, k, v: (k, f(v))))
toolz_rules.add(eq(curried.keyfilter), _filter_items(lambda f, k, v: f(k)))
toolz_rules.add(eq(curried.valfilter), _filter_items(lambda f, k, v: f(v)))
toolz_rules.add(eq(curried.itemfilter), _toolz_item_filter)
toolz_rules.add(eq(curried.mapcat), lambda _, f, o: o.map(f.args[0]).flatten())
toolz_rules.add(eq(curried.random_sample), lambda _, f, o: _call(o.random_sample, f))
toolz_rules.add(eq(curried.reduce), lambda _, f, o: _call(o.fold, f))
toolz_rules.add(eq(curried.reduceby), lambda _, f, o: DaskDict(_call(o.foldby, f)))
toolz_rules.add(eq(curried.remove), lambda _, f, o: _call(o.remove, f))
toolz_rules.add(eq(curried.topk), lambda _, f, o: _call(o.topk, f))
toolz_rules.add(eq(curried.unique), lambda _, f, o: _call(o.distinct, f))
toolz_rules.add(eq(toolz.unique), lambda _, __, o: o.distinct())

toolz_rules.add(eq(curried.accumulate), _raise(NotImplementedError))
toolz_rules.add(eq(curried.assoc), _raise(NotImplementedError))
toolz_rules.add(eq(curried.assoc_in), _raise(NotImplementedError))
toolz_rules.add(eq(curried.cons), _raise(NotImplementedError))
toolz_rules.add(eq(curried.do), _raise(NotImplementedError))
toolz_rules.add(eq(curried.drop), _raise(NotImplementedError))
toolz_rules.add(eq(curried.excepts), _raise(NotImplementedError))
toolz_rules.add(eq(curried.get), _raise(NotImplementedError))
toolz_rules.add(eq(curried.get_in), _raise(NotImplementedError))
toolz_rules.add(eq(curried.interpose), _raise(NotImplementedError))
toolz_rules.add(eq(curried.iterate), _raise(NotImplementedError))
toolz_rules.add(eq(curried.nth), _raise(NotImplementedError))
toolz_rules.add(eq(curried.partial), _raise(NotImplementedError))
toolz_rules.add(eq(curried.partition), _raise(NotImplementedError))
toolz_rules.add(eq(curried.partition_all), _raise(NotImplementedError))
toolz_rules.add(eq(curried.partitionby), _raise(NotImplementedError))
toolz_rules.add(eq(curried.sliding_window), _raise(NotImplementedError))
toolz_rules.add(eq(curried.sorted), _raise(NotImplementedError))
toolz_rules.add(eq(curried.tail), _raise(NotImplementedError))
toolz_rules.add(eq(curried.take), _raise(NotImplementedError))
toolz_rules.add(eq(curried.take_nth), _raise(NotImplementedError))
toolz_rules.add(eq(curried.update_in), _raise(NotImplementedError))


# ########################################################################### #
# #                                                                         # #
# #                             Reductions                                  # #
# #                                                                         # #
# ########################################################################### #


class reduction(object):
    """Represent a reduction suited for parallelization with `dask.bag`.
    """
    pass


def mean(l):
    """Calculate the mean of a list of values."""
    l = iter(l)

    s0 = 1
    try:
        s1 = next(l)

    except StopIteration:
        raise RuntimeError('cannot compute the mean of an empty iterable')

    for x in l:
        s1 += x
        s0 += 1

    return s1 / s0


def var(*args, **kwargs):
    if not args:
        return _var(**kwargs)

    return _var(**kwargs)(*args)


def std(*args, **kwargs):
    if not args:
        return _std(**kwargs)

    return _std(**kwargs)(*args)


class _var(object):
    def __init__(self, ddof=0.0):
        self.ddof = ddof

    def __call__(self, seq):
        s2, s1, s0 = 0.0, 0.0, 0.0

        for x in seq:
            s2 += x ** 2.0
            s1 += x
            s0 += 1

        return ((s2 / s0) - (s1 / s0) ** 2.0) * s0 / (s0 - self.ddof)


class _std(object):
    def __init__(self, ddof=0.0):
        self.ddof = ddof

    def __call__(self, seq):
        return math.sqrt(_var(self.ddof)(seq))


reduction_rules = RuleSet()
reduction_rules.add(eq(mean), lambda _, __, o: o.mean())
reduction_rules.add(a(_var), lambda _, f, o: o.var(f.ddof))
reduction_rules.add(a(_std), lambda _, f, o: o.std(f.ddof))
reduction_rules.add(eq(var), lambda _, f, o: o.var())
reduction_rules.add(eq(std), lambda _, f, o: o.std())


# ########################################################################### #
# #                                                                         # #
# #                             Extensions                                  # #
# #                                                                         # #
# ########################################################################### #


class chained(object):
    """Represent the composition of functions.

    When the resulting object is called with a single argument, the passed
    object is transformed by passing it through all given functions.
    For example::

        a = chained(
            math.sqrt,
            math.log,
            math.cos,
        )(5.0)

    is equivalent to::

        a = 5.0
        a = math.sqrt(a)
        a = math.log(a)
        a = math.cos(a)

    """
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, obj):
        for func in self.funcs:
            obj = func(obj)

        return obj

    def __repr__(self):
        return 'tztz.chained({})'.format(', '.join(repr(func) for func in self.funcs))

    def __iter__(self):
        return iter(self.funcs)


def _db_chained(rules, chain, obj):
    for func in chain:
        obj = rules(func, obj)

    return obj


class repartition(object):
    """Express repartition of a ``dask.bag.Bag``, for non bags it is a nop-op.

    :param int n:
        the number of partitions
    """
    def __init__(self, n):
        self.n = n

    def __call__(self, obj):
        return obj


extension_rules = RuleSet()
extension_rules.add(a(chained), _db_chained)
extension_rules.add(a(repartition), lambda _, f, o: o.repartition(f.n))


# ########################################################################### #
# #                                                                         # #
# #                           Combined rules                                # #
# #                                                                         # #
# ########################################################################### #


rules = stdlib_rules | toolz_rules | reduction_rules | extension_rules
rules.add(lambda _, func, __: callable(func), lambda _, func, obj: func(obj))
