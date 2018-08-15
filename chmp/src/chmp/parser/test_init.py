import pytest
import chmp.parser as p


examples = {
    p.eq("a"): [
        ("a", "", ["a"]),
        ("b", "b", None),
        ("ab", "b", ["a"]),
        ("ba", "ba", None),
    ],
    p.ne("a"): [
        ("a", "a", None),
        ("b", "", ["b"]),
        ("ab", "ab", None),
        ("ba", "a", ["b"]),
    ],
    p.any(): [
        ("a", "", ["a"]),
        ("b", "", ["b"]),
        ("ab", "b", ["a"]),
        ("ba", "a", ["b"]),
    ],
    p.sequential(p.eq("a"), p.eq("b")): [
        ("ab", "", ["a", "b"]),
        ("aab", "aab", None),
        ("abb", "b", ["a", "b"]),
    ],
    p.repeat(p.eq("a")): [
        ("ab", "b", ["a"]),
        ("ba", "ba", []),
        ("aab", "b", ["a", "a"]),
        ("aaab", "b", ["a", "a", "a"]),
    ],
    p.first(p.eq("a"), p.eq("b")): [
        ("abc", "bc", ["a"]),
        ("bca", "ca", ["b"]),
        ("cab", "cab", None),
    ],
    p.ignore(p.eq("a")): [
        ("a", "", []),
        ("b", "b", None),
        ("ab", "b", []),
        ("ba", "ba", None),
    ],
}

# reformat the examples to fit the parametrize format
examples = [
    (parser, input, rest, result)
    for parser, l in examples.items()
    for input, rest, result in l
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
