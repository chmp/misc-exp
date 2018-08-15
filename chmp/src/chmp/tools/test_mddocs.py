from chmp.tools.mddocs import transform

doc = """# Example module

.. automodule:: chmp.tools.test_mddocs
    :members:


example_example_example
"""


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
