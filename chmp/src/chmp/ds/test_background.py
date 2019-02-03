import itertools as it

from chmp.ds import bgloop, wait, fast_product


def test_bgloop():
    result = []

    @bgloop("test", range(4), range(2))
    def _(_, a, b):
        result.append((a, b))

    wait("test")

    assert result == [*it.product(range(4), range(2))]


def test_fast_product():
    assert [*fast_product()] == [*it.product()]
    assert [*fast_product(range(5))] == [*it.product(range(5))]
    assert [*fast_product(range(5), range(10))] == [*it.product(range(5), range(10))]
    assert [*fast_product(range(5), range(10), range(3))] == [
        *it.product(range(5), range(10), range(3))
    ]
