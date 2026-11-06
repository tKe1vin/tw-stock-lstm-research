"""V2.11 keeps the imported signal and does not treat yesterday as today."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "37591d21300977b9486212fc86fe257bb050db47177fa2f41bd7a8904919f659"
CREATED = "2026-11-05T23:23:09.114628+08:00"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_imported_signal_bytes_and_timestamp():
    path = ROOT / "experiments" / "results" / "v2" / "v211" / "daily" / "2026-11-05.csv"
    archived = ROOT / "research" / "v211" / "signal_registry" / "developer_original" / "2026-11-05.csv"
    assert _sha256(path) == EXPECTED
    assert path.read_bytes() == archived.read_bytes()
    import pandas as pd

    frame = pd.read_csv(path)
    assert set(frame.Created_At) == {CREATED}
    assert frame.Execution_Price.isna().all()
    assert frame.Actual_5d_Pct.isna().all()


def test_yesterday_signal_is_not_today_success():
    from experiments.v2.v211_daily import classify_exit

    now = datetime(2026, 11, 6, 0, 42, tzinfo=ZoneInfo("Asia/Taipei"))
    status = {
        "signal_published": False,
        "ok": True,
        "signal": {"reason": "would_backfill_an_earlier_session", "completed_session": "2026-11-05"},
    }
    report = {"completed_session": "2026-11-05", "ready_to_predict": True}
    assert classify_exit(status, report, lambda day: day == "2026-11-05", now) == 2


def test_check_run_does_not_rewrite_the_signal():
    from scripts.run_model import run

    path = ROOT / "experiments" / "results" / "v2" / "v211" / "daily" / "2026-11-05.csv"
    before = _sha256(path)
    first = run("v211", "check")
    second = run("v211", "check")
    assert before == _sha256(path) == EXPECTED
    assert first == 2
    assert second == 2
    assert sorted(item.name for item in path.parent.glob("*.csv")) == ["2026-11-05.csv"]
