"""Exchange-confirmed closures missing from or newer than the library calendar."""
import pandas as pd

from scripts.common import sessions


def test_2022_lunar_new_year_reopens_feb7_not_feb4():
    days = sessions("2022-01-26", "2022-02-07")
    assert list(days) == list(pd.to_datetime(["2022-01-26", "2022-02-07"]))


def test_verified_2026_typhoon_closures_preserve_adjacent_sessions():
    july = sessions("2026-07-09", "2026-07-13")
    october = sessions("2026-10-22", "2026-10-27")
    assert list(july) == list(pd.to_datetime(["2026-07-09", "2026-07-13"]))
    # Oct 26 is the observed national holiday, already in the XTAI calendar.
    assert list(october) == list(pd.to_datetime(["2026-10-22", "2026-10-27"]))
