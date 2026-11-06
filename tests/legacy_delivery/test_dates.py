"""The four 555 V2 dates stay in separate fields."""
from __future__ import annotations

from scripts.run_model import v2_date_record


def test_signal_cutoff_and_planned_fill_are_different_fields():
    dates = v2_date_record("2026-11-05", "2026-11-06", "2026-11-06")
    assert dates["signal_date"] == "2026-11-05"
    assert dates["data_asof"] == "2026-11-05"
    assert dates["run_date"] == "2026-11-06"
    assert dates["intended_execution_date"] == "2026-11-06"
    assert dates["signal_date"] != dates["intended_execution_date"]
    assert dates["data_asof"] != dates["intended_execution_date"]
