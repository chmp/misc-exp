from chmp.widgets import JSExpr, JSObj

import pytest


@pytest.mark.parametrize(
    "expr, obj, expected",
    [
        ("datum['t']", {"t": 5}, 5),
        ("datum['t'] < 5", {"t": 1}, True),
        ("datum['t'] < 5", {"t": 6}, False),
        ("true", {}, True),
        ("false", {}, False),
        ("null", {}, None),
        ("datum.date == '2019-06-16'", JSObj(date="2011-02-15"), False),
        ("datum.date == '2019-06-16'", JSObj(date="2019-06-16"), True),
    ],
)
def test_js_expr(expr, obj, expected):
    actual = JSExpr(["datum"], expr)(obj)
    assert actual is expected
