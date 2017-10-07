from chmp.reactive import invert_graph, get_outstanding_updates, get_transitive

import pytest


@pytest.mark.parametrize('graph, roots, expected', [
    ({'b': ['a']}, ['a'], ['a', 'b']),
    ({'b': ['a'], 'c': ['b']}, ['a'], ['a', 'b', 'c']),
    ({'b': ['a'], 'c': ['b']}, ['a', 'b'], ['a', 'b', 'c']),
    ({'b': ['a'], 'c': ['b'], 'e': ['d']}, ['a'], ['a', 'b', 'c']),
    ({'b': ['a'], 'c': ['a'], 'd': ['b', 'c']}, ['a'], ['a', 'b', 'c', 'd']),
    (
        {'b': ['a'], 'c': ['a'], 'd': ['b', 'c'], 'f': ['e'], 'g': ['f']},
        ['a', 'e'],
        ['a', 'e', 'b', 'c', 'd', 'f', 'g'],
    ),
    (
        {'b': ['a'], 'c': ['a'], 'd': ['b', 'c'], 'f': ['e'], 'g': ['c', 'f']},
        ['a', 'e'],
        ['a', 'e', 'b', 'c', 'd', 'f', 'g'],
    ),
])
def test_get_outstanding_updates(graph, roots, expected):
    inverse_graph = invert_graph(graph)
    actual = get_outstanding_updates(roots, inverse_graph)

    assert actual == expected


@pytest.mark.parametrize('graph, expected', [
    ({'a': ['b']}, {'a': {'b'}, 'b': set()}),
    ({'b': ['a'], 'c': ['b']}, {'c': {'a', 'b'}, 'b': {'a'}, 'a': set()}),
({'b': ['a'], 'c': ['a'], 'd': ['b', 'c']}, {'b': {'a'}, 'c': {'a'}, 'd': {'a', 'b', 'c'}, 'a': set()}),
])
def test_transitive(graph, expected):
    actual = get_transitive(graph)

    assert actual == expected
