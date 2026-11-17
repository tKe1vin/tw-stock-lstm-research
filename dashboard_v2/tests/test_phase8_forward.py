"""Phase 8 acceptance: maturity gate, conflict persistence, no writeback."""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dashboard_v2.data_access.load import historical_official, v211_status
from dashboard_v2.data_access.prospective import (
    EXPECTED_WEIGHT,
    compare_source_conflicts,
    ridge_status,
    source_status,
    v211_matured_local,
    v211_nav_local,
    v211_root,
)

TAIPEI = ZoneInfo("Asia/Taipei")
_EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "v2_integration"
    / "evidence"
    / "source_conflict_1110"
)
LOCAL_SIGNAL = _EVIDENCE / "LOCAL_2026-11-10.csv"
GITHUB_SIGNAL = _EVIDENCE / "GITHUB_2026-11-10.csv"


def test_1117_maturity_gate_respects_protocol_close():
    before = source_status(datetime(2026, 11, 17, 13, 25, tzinfo=TAIPEI))
    assert before["label_maturity_gate"]["state"] == "PENDING"
    assert before["label_maturity_gate"]["protocol_close_passed"] is False
    assert "2026-11-09" not in set(v211_matured_local()["signal_days"])

    after = source_status(datetime(2026, 11, 17, 15, 5, tzinfo=TAIPEI))
    # Even after clock close, without prices_through>=11/17 and matured file row, stay PENDING.
    assert after["label_maturity_gate"]["state"] == "PENDING"


def test_original_signal_sha_unchanged_for_conflict_pair():
    local = hashlib.sha256(LOCAL_SIGNAL.read_bytes()).hexdigest()
    github = hashlib.sha256(GITHUB_SIGNAL.read_bytes()).hexdigest()
    assert local != github
    assert "SYNTHETIC_DEMO" in LOCAL_SIGNAL.read_text(encoding="utf-8")
    assert "SYNTHETIC_DEMO" in GITHUB_SIGNAL.read_text(encoding="utf-8")
    conflicts = compare_source_conflicts("2026-11-10")
    assert conflicts[0]["local_sha256"] == local
    assert conflicts[0]["github_sha256"] == github
    assert conflicts[0]["demo_flag"] == "SYNTHETIC_DEMO"


def test_matured_scores_have_no_duplicate_signal_ticker():
    rows = v211_matured_local()["rows"]
    keys = [(row["Signal_Date"], row["Ticker"]) for row in rows]
    assert len(keys) == len(set(keys))
    assert set(v211_matured_local()["signal_days"]) <= {"2026-11-05", "2026-11-06"}


def test_ridge_and_v211_remain_separate_and_rmse_present():
    ridge = ridge_status()
    v211 = v211_status()
    assert ridge["not_merged_with_v211"] is True
    assert ridge["horizon"] == "1 trading day"
    assert ridge["rmse"] is not None
    assert v211["confirmation"] == "PENDING"
    assert ridge["status"] == "PENDING"


def test_immature_not_in_performance_claim():
    status = v211_status()
    assert status["performance"] == "PENDING"
    assert status["matured_signal_days"] < 60
    gate = source_status()["label_maturity_gate"]
    assert gate["state"] != "MATURED" or "2026-11-09" in v211_matured_local()["signal_days"]


def test_source_conflict_still_visible():
    status = source_status()
    assert status["source_conflict"] is True
    assert status["conflicts"][0]["status"] == "SOURCE_CONFLICT"


def test_nav_matches_snapshot_ledger():
    status = v211_status()
    nav_rows = v211_nav_local()["rows"]
    assert nav_rows
    latest = nav_rows[-1]
    assert abs(float(latest["NAV_Open"]) - float(status["open_nav"])) < 1e-9
    assert latest["Date"] >= "2026-11-16"


def test_protocols_not_mixed():
    hist = historical_official()
    assert hist["rows"]["444"]["ROI_Pct"] == "21.376684850112927"
    assert hist["rows"]["555"]["ROI_Pct"] == "11.953686872710325"
    ridge = ridge_status()
    assert "ROI" not in ridge
    assert ridge["protocol"].startswith("273-day")


def test_dashboard_does_not_write_formal_files():
    root = v211_root()
    targets = [
        root / "status.json",
        root / "matured_scores.csv",
        root / "live_nav.csv",
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "results"
        / "v2"
        / "v211"
        / "weights"
        / "weights.npz",
    ]
    before = {path: path.stat().st_mtime_ns for path in targets if path.exists()}
    source_status()
    ridge_status()
    v211_status()
    v211_matured_local()
    after = {path: path.stat().st_mtime_ns for path in targets if path.exists()}
    assert before == after
    weight = targets[-1]
    assert weight.exists()
    assert hashlib.sha256(weight.read_bytes()).hexdigest() == EXPECTED_WEIGHT
