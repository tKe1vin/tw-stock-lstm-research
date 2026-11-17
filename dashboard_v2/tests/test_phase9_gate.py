"""Phase 9: maturity gate, ledger semantics, conflict integrity tags."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from dashboard_v2.data_access.prospective import (
    compare_source_conflicts,
    ledger_status,
    ridge_status,
    source_status,
    v211_matured_local,
)

TAIPEI = ZoneInfo("Asia/Taipei")


def test_phase9_gate_pending_without_forced_maturity():
    now = datetime.now(TAIPEI)
    status = source_status(now)
    gate = status["label_maturity_gate"]
    matured = set(v211_matured_local()["signal_days"])
    assert "2026-11-05" in matured and "2026-11-06" in matured
    if now.hour < 15 or status["local_v211_through"] < "2026-11-17":
        assert gate["state"] == "PENDING"
        assert "2026-11-09" not in matured


def test_ridge_keeps_prior_scores_when_1117_pending():
    ridge = ridge_status()
    assert ridge["scorable_days"] >= 5
    assert ridge["status"] == "PENDING"
    assert "2026-11-17" in (ridge.get("pending_targets") or []) or ridge.get("today_label_state") in {
        "PENDING_BEFORE_CLOSE",
        "NO_OR_PENDING_LABEL",
        "SCORED",
    }


def test_ledger_distinguishes_nav_open_from_unavailable_close():
    ledger = ledger_status()
    assert ledger["nav_open"] is not None
    assert ledger["nav_close_status"] == "DATA_UNAVAILABLE"
    assert ledger["nav_close"] is None
    assert ledger["execution_kind"] == "SIMULATED_EXECUTION"
    assert ledger["published_fill_count"] >= 2


def test_conflict_integrity_flags_present():
    conflicts = compare_source_conflicts("2026-11-10")
    assert conflicts
    integrity = conflicts[0]["integrity"]
    assert integrity["SOURCE_CONFLICT"] is True
    assert integrity["ORIGINAL_INPUT_UNAVAILABLE"] is True
    assert integrity["PARTIALLY_REPRODUCIBLE"] is True
    assert integrity["do_not_refetch_news_as_original"] is True
    assert integrity["do_not_overwrite_local_with_github_sentiment"] is True
