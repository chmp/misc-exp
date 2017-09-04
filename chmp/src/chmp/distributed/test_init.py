from __future__ import print_function, division, absolute_import

import collections
import itertools as it
import operator as op

import dask.bag as db
import pytest
import toolz
import toolz.curried as curried

from toolz import first, second

from chmp.distributed import apply, chained, mean, var, std


examples = {
    'mean': (mean, range(20)),
    'var': (var, range(20)),
    'std': (std, range(20)),
    'var (1)': (var(ddof=1.0), range(20)),
    'std (1)': (std(ddof=1.0), range(20)),

    'any 1': (any, [True, False, False]),
    'any 2': (any, [False, False, False]),
    'any 3': (any, [True, True, True]),
    'all 1': (all, [True, False, False]),
    'all 2': (all, [False, False, False]),
    'all 3': (all, [True, True, True]),
    'sum': (sum, list(range(20))),
    'min': (min, list(range(20))),
    'max': (max, list(range(20))),
    'len': (len, list(range(20))),
    'list': (list, list(range(20))),
    'dict': (dict, [(1, 2), (3, 4), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]),
    'dict keys': (
        chained(dict, op.methodcaller('keys'), list),
        [(1, 2), (3, 4), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
    ),
    'dict values': (
        chained(dict, op.methodcaller('values'), list),
        [(1, 2), (3, 4), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
    ),
    'dict items': (
        chained(dict, op.methodcaller('items'), list),
        [(1, 2), (3, 4), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
    ),
    'dict.items 2': (
        chained(dict, dict.items, list),
        [(1, 2), (3, 4), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
    ),
    'itertools.chain.from_iterable': (
        chained(it.chain.from_iterable, list),
        [(1, 2), (3, 4), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
    ),
    'collections.Counter': (chained(collections.Counter, dict), [1, 2, 2, 3, 3, 3, 4, 4, 4]),
    'collections.Counter.elements': (
        chained(collections.Counter, op.methodcaller('elements'), sorted),
        [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    ),
    'collections.Counter.most_common': (
        chained(collections.Counter, op.methodcaller('most_common', 2), sorted),
        [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    ),
    'set': (chained(set, sorted), [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]),

    'map': (chained(curried.map(lambda x: 2 * x), list), [1, 2, 3, 4, 5, 6]),
    'filter': (chained(curried.filter(lambda x: x % 2 == 0), list), range(10)),
    'remove': (chained(curried.remove(lambda x: x % 2 == 0), list), range(10)),

    # example from toolz docs
    'pluck-single': (
        chained(curried.pluck('name'), list),
        [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}],
    ),

    # example from toolz docs
    'pluck-mulitple': (
        chained(curried.pluck([0, 1]), list),
        [[1, 2, 3], [4, 5, 7]],
    ),

    # example from toolz docs
    'join': (
        chained(
            curried.join(
                second,
                [('Alice', 'Edith'), ('Alice', 'Zhao'), ('Edith', 'Alice'),
                 ('Zhao', 'Alice'), ('Zhao', 'Edith')],
                first
            ),
            curried.map(lambda t: (t[0][0], t[1][1])),
            sorted,
        ),
        [('Alice', 'NYC'), ('Alice', 'Chicago'), ('Dan', 'Syndey'),
         ('Edith', 'Paris'), ('Edith', 'Berlin'), ('Zhao', 'Shanghai')]
    ),

    'count_by': (curried.countby(lambda x: x % 2 == 0), range(20)),

    'groupby': (
        chained(
            curried.groupby(lambda x: x % 2 == 0),
            curried.valmap(sorted),
        ),
        range(20),
    ),
    'keymap': (
        chained(dict, curried.keymap(lambda x: 2 * x)),
        dict.items({1: 2, 3: 4, 5: 6, 7: 8, 9: 10}),
    ),
    'valmap': (
        chained(dict, curried.valmap(lambda x: 2 * x)),
        dict.items({1: 2, 3: 4, 5: 6, 7: 8, 9: 10}),
    ),
    'keyfilter': (
        chained(dict, curried.keyfilter(lambda x: x > 5)),
        dict.items({1: 2, 3: 4, 5: 6, 7: 8, 9: 10}),
    ),
    'valfilter': (
        chained(dict, curried.valfilter(lambda x: x > 5)),
        dict.items({1: 2, 3: 4, 5: 6, 7: 8, 9: 10}),
    ),
    'itemfilter': (
        chained(dict, curried.itemfilter(lambda i: i[0] % 2 == 0 and i[1] < 4)),
        dict.items({1: 2, 2: 3, 3: 4, 4: 5})
    ),
    # example taken from toolz docs
    'mapcat': (
        chained(
            curried.mapcat(lambda s: [c.upper() for c in s]),
            list,
        ),
        [["a", "b"], ["c", "d", "e"]],
    ),
    'reduce': (curried.reduce(op.add), range(20)),
    'reduceby': (curried.reduceby(lambda x: x % 2 == 0, op.add), range(20)),
    'topk': (chained(curried.topk(5), list), range(20)),
    'curried.unique': (chained(curried.unique, sorted, ), [1, 1, 2, 3, 4, 4]),
    'unique': (chained(toolz.unique, sorted, ), [1, 1, 2, 3, 4, 4]),
}


def params(spec, m):
    return pytest.mark.parametrize(
        spec, [v for (_, v) in sorted(m.items())], ids=sorted(m.keys()),
    )


@params('func, arg', examples)
def test_input_outputs(func, arg):
    actual_no_dask = func(arg)
    actual_dask = apply(db.from_sequence(arg, npartitions=3), func).compute()

    assert actual_dask == actual_no_dask
