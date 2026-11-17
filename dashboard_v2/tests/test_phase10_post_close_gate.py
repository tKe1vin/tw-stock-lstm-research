"""Phase 10: do not fabricate post-close scores when gate fails."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from dashboard_v2.data_access.load import historical_official
from dashboard_v2.data_access.prospective import ridge_status, source_status, v211_matured_local

TAIPEI = ZoneInfo("Asia/Taipei")


def test_phase10_keeps_pending_when_gate_incomplete():
    now = datetime.now(TAIPEI)
    status = source_status(now)
    gate = status["label_maturity_gate"]
    matured = set(v211_matured_local()["signal_days"])
    ridge = ridge_status(now)

    # Historical official tokens unchanged.
    hist = historical_official()
    assert hist["rows"]["444"]["ROI_Pct"] == "21.376684850112927"
    assert hist["rows"]["555"]["ROI_Pct"] == "11.953686872710325"

    # If live clock/data are still pre-gate, maturity must stay pending.
    if now.hour < 15 or status["local_v211_through"] < "2026-11-17":
        assert gate["state"] == "PENDING"
        assert "2026-11-09" not in matured
        assert ridge["scorable_days"] == 5
        assert "2026-11-17" in (ridge.get("pending_targets") or [])
        assert ridge["status"] == "PENDING"


def test_github_research_date_separated_from_v211_publish():
    status = source_status()
    # Paired research may be newer than frozen V2.11 publish cutoff.
    assert status.get("github_research_as_of_date") == "2026-11-16"
    assert status.get("github_latest_success") == "2026-11-10"
    assert status.get("github_delayed_run") is True
    assert status.get("github_orchestrator_status") in {
        "research_complete_tracker_deferred",
        "partial",
        "complete",
    }
    assert status.get("github_v211_maintenance_only") is True
    assert status.get("local_latest_signal") >= "2026-11-16"
    assert status["local_latest_signal"] != status["github_latest_success"]
