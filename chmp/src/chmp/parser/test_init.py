import pytest
import chmp.parser as p


examples = [
    (p.eq("a"), "a", "", ["a"]),
    (p.eq("a"), "b", "b", None),
    (p.eq("a"), "ab", "b", ["a"]),
    (p.eq("a"), "ba", "ba", None),
    (p.ne("a"), "a", "a", None),
    (p.ne("a"), "b", "", ["b"]),
    (p.ne("a"), "ab", "ab", None),
    (p.ne("a"), "ba", "a", ["b"]),
    (p.any(), "a", "", ["a"]),
    (p.any(), "b", "", ["b"]),
    (p.any(), "ab", "b", ["a"]),
    (p.any(), "ba", "a", ["b"]),
    (p.sequential(p.eq("a"), p.eq("b")), "ab", "", ["a", "b"]),
    (p.sequential(p.eq("a"), p.eq("b")), "aab", "aab", None),
    (p.sequential(p.eq("a"), p.eq("b")), "abb", "b", ["a", "b"]),
    (p.repeat(p.eq("a")), "ab", "b", ["a"]),
    (p.repeat(p.eq("a")), "ba", "ba", []),
    (p.repeat(p.eq("a")), "aab", "b", ["a", "a"]),
    (p.repeat(p.eq("a")), "aaab", "b", ["a", "a", "a"]),
    (p.first(p.eq("a"), p.eq("b")), "abc", "bc", ["a"]),
    (p.first(p.eq("a"), p.eq("b")), "bca", "ca", ["b"]),
    (p.first(p.eq("a"), p.eq("b")), "cab", "cab", None),
    (p.ignore(p.eq("a")), "a", "", []),
    (p.ignore(p.eq("a")), "b", "b", None),
    (p.ignore(p.eq("a")), "ab", "b", []),
    (p.ignore(p.eq("a")), "ba", "ba", None),
]

ids = [
    "{}-{}-{}".format(parser.__name__, input, id(parser))
    for parser, input, _1, _2 in examples
]


@pytest.mark.parametrize(
    "parser, input, expected_rest, expected_result", examples, ids=ids
)
def test_eq(parser, input, expected_rest, expected_result):
    actual_rest, actual_result, _ = parser(input, 0)

    assert actual_result == expected_result
    assert actual_rest == expected_rest
