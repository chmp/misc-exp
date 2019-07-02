from chmp.torch_utils.training import nested_format


def test_examples():
    assert nested_format(1.0, ".1f") == "1.0"
    assert nested_format((1.0, 2.0), ".1f") == "(1.0, 2.0)"
    assert (
        nested_format({"a": (1.0, 2.0), "b": 3.0}, ".1f")
        == "{'a': (1.0, 2.0), 'b': 3.0}"
    )
