from keep_odds import keep_odds


def test_keep_odds():
    assert keep_odds([1, 2, 3, 4, 5, 6]) == [1, 3, 5]
