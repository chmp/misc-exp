import pytest

from chmp.tools.mddocs import transform, transform_file, format_signature

doc = """# Example module

.. automodule:: chmp.tools.test_mddocs
    :members:


example_example_example
"""


class C:
    def positional_only(self, a, b, c):
        pass

    def varargs_only(*args):
        pass

    def varargs(self, a, b, *c):
        pass

    def defaults(self, a, b=2, c=3):
        pass

    def kwargs(self, a, **b):
        pass

    def pure_kwonly(self, a, *, b, c):
        pass

    def vargs_kwonly(self, a, *b, c):
        pass

    def kwonly_defaults(self, a, *, b, c=1):
        pass

    def kwonly_kwargs(self, a, *, b, c=1, **d):
        pass


@pytest.mark.parametrize(
    "func, expected",
    [
        (C.positional_only, "foo(self, a, b, c)"),
        (C.varargs_only, "foo(*args)"),
        (C.varargs, "foo(self, a, b, *c)"),
        (C.defaults, "foo(self, a, b=2, c=3)"),
        (C.pure_kwonly, "foo(self, a, *, b, c)"),
        (C.vargs_kwonly, "foo(self, a, *b, c)"),
        (C.kwonly_defaults, "foo(self, a, *, b, c=1)"),
        (C.kwargs, "foo(self, a, **b)"),
        (C.kwonly_kwargs, "foo(self, a, *, b, c=1, **d)"),
    ],
)
def test_format_signature(func, expected):
    assert format_signature("foo", func) == expected


def test_examples():
    """Test that certain keywords appear in the documentation"""
    result = transform(doc, __file__)

    # check that also members of classes are documented
    assert "foo_constructor" in result
    assert "foo_method" in result
    assert "example_example_example" in result
    assert "note_note_note" in result
    assert "warning_warning_warning" in result
    assert "seealso_seealso_seealso" in result
    assert "todo_todo_todo" in result


def test_multifile_exampele(tmpdir):
    tmpdir.join("root.md").write(root_source)
    tmpdir.join("example.md").write(doc)

    source = str(tmpdir.join("root.md"))
    target = str(tmpdir.join("target.md"))
    transform_file(source, target)


root_source = """# Root

.. literalinclude:: example.md

.. include:: example.md
"""


def example_numpy():
    """My numpydoc description of a kind of very exhautive numpydoc format docstring.

    Parameters
    ----------
    first : array_like
        the 1st param name `first`
    second :
        the 2nd param
    third : {'value', 'other'}, optional
        the 3rd param, by default 'value'

    Returns
    -------
    string
        a value in a string

    Raises
    ------
    KeyError
        when a key error
    OtherError
        when an other error
    """


def example_rest_style():
    """This is a reST style.

    :param int param1: this is a first param
    :param param2: this is a second param
    :returns: this is a description of what is returned
    :raises keyError: raises an exception
    """


def example_adminitions():
    """
    .. note::

        note_note_note

    .. warning::

        warning_warning_warning

    .. seealso::

        seealso_seealso_seealso

    .. todo::

        todo_todo_todo

    """


class Foo:
    """Bar"""

    def __init__(self):
        """foo_constructor"""
        pass

    def method(self, a, b):
        """foo_method"""
        pass
