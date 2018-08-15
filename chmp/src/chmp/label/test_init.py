import pytest

from chmp.label import BaseAnnotator


def test_base_annotator__example_session():
    annotator = BaseAnnotator()
    assert annotator.current_item is None

    annotator.annotate(["a", "b", "c"])
    assert annotator.current_item == ("order", 0, "a")

    annotator.annotate_current("foo")
    assert len(annotator.annotations) == 1
    assert annotator.annotations[-1] == {
        "item": "a",
        "index": 0,
        "reason": "order",
        "label": "foo",
    }
    assert annotator.current_item == ("order", 1, "b")

    annotator.repeat(0)
    assert len(annotator.annotations) == 1
    assert annotator.annotations[-1] == {
        "item": "a",
        "index": 0,
        "reason": "order",
        "label": "foo",
    }
    assert annotator.current_item == ("repeat", 0, "a")

    annotator.annotate_current("bar")
    assert len(annotator.annotations) == 2
    assert annotator.annotations[-1] == {
        "item": "a",
        "index": 0,
        "reason": "repeat",
        "label": "bar",
    }
    assert annotator.current_item == ("order", 1, "b")

    annotator.annotate_current("baz")
    assert len(annotator.annotations) == 3
    assert annotator.annotations[-1] == {
        "item": "b",
        "index": 1,
        "reason": "order",
        "label": "baz",
    }
    assert annotator.current_item == ("order", 2, "c")

    annotator.annotate_current("foo")
    assert len(annotator.annotations) == 4
    assert annotator.annotations[-1] == {
        "item": "c",
        "index": 2,
        "reason": "order",
        "label": "foo",
    }
    assert annotator.current_item is None

    annotator.repeat(2)
    assert len(annotator.annotations) == 4
    assert annotator.annotations[-1] == {
        "item": "c",
        "index": 2,
        "reason": "order",
        "label": "foo",
    }
    assert annotator.current_item == ("repeat", 1, "b")

    annotator.annotate_current("fxx")
    assert len(annotator.annotations) == 5
    assert annotator.annotations[-1] == {
        "item": "b",
        "index": 1,
        "reason": "repeat",
        "label": "fxx",
    }
    assert annotator.current_item is None

    assert annotator.get_latest() == [
        {"item": "b", "index": 1, "reason": "repeat", "label": "fxx"},
        {"item": "c", "index": 2, "reason": "order", "label": "foo"},
        {"item": "a", "index": 0, "reason": "repeat", "label": "bar"},
    ]


def test_base_annotator__clear():
    annotator = BaseAnnotator()
    annotator.annotate(["a", "b", "c"])

    with pytest.raises(RuntimeError):
        annotator.annotate(["a", "b", "c"])

    annotator.clear()
    annotator.annotate(["a", "b", "c"])
