from chmp.ds import Object, timed, singledispatch_on


def test_object():
    a = Object(a=2, b=3)

    assert a == Object(a=2, b=3)
    assert a != Object(a=2, b=4)
    assert Object(a, b=4) == Object(a=2, b=4)

    assert a.a == 2
    assert a.b == 3

    assert vars(a) == dict(a=2, b=3)


def test_timed():
    with timed():
        assert True is True

    with timed("label"):
        assert True is True


def test_singledispatch_on():
    @singledispatch_on(1)
    def foo(a, b):
        return 1

    @foo.register(int)
    def bar(a, b):
        return 2

    assert foo(0, 2) == 2
    assert foo(1, None) == 1
    assert foo(None, 2) == 2
    assert foo(None, None) == 1
