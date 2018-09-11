import pandas as pd
import pandas.util.testing as pdt

from chmp.ds import (
    to_start_of_day,
    to_start_of_week,
    to_start_of_year,
    to_time_in_day,
    to_time_in_week,
    to_time_in_year,
)


def test_to_date():
    s = pd.Series(pd.to_datetime(["2011-01-08 11:23:51", "2018-09-11 13:20:05"]))
    expected = pd.Series(pd.to_datetime(["2011-01-08", "2018-09-11"]))
    actual = to_start_of_day(s)

    pdt.assert_series_equal(actual, expected)


def test_time_in_day():
    s = pd.Series(pd.to_datetime(["2011-01-08 11:23:51", "2018-09-11 13:20:05"]))
    expected = pd.Series(pd.to_timedelta(["11h 23m 51s", "13h 20m 5s"]))
    actual = to_time_in_day(s)

    pdt.assert_series_equal(actual, expected)


def test_to_start_of_week():
    s = pd.Series(pd.to_datetime(["2011-01-08 11:23:51", "2018-09-11 13:20:05"]))
    expected = pd.Series(pd.to_datetime(["2011-01-03", "2018-09-10"]))
    actual = to_start_of_week(s)

    pdt.assert_series_equal(actual, expected)


def test_to_time_in_week():
    s = pd.Series(pd.to_datetime(["2011-01-08 11:23:51", "2018-09-11 13:20:05"]))
    expected = pd.Series(pd.to_timedelta(["5d 11h 23m 51s", "1d 13h 20m 5s"]))
    actual = to_time_in_week(s)

    pdt.assert_series_equal(actual, expected)


def test_to_start_of_year():
    s = pd.Series(pd.to_datetime(["2011-01-08 11:23:51", "2018-09-11 13:20:05"]))
    expected = pd.Series(pd.to_datetime(["2011-01-01", "2018-01-01"]))
    actual = to_start_of_year(s)

    pdt.assert_series_equal(actual, expected)


def test_to_time_in_year():
    s = pd.Series(pd.to_datetime(["2011-01-08 11:23:51", "2018-09-11 13:20:05"]))
    expected = pd.Series(pd.to_timedelta(["7d 11h 23m 51s", "253d 13h 20m 5s"]))
    actual = to_time_in_year(s)

    pdt.assert_series_equal(actual, expected)
