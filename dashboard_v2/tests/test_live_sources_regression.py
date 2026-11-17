"""Real overlays never borrow DEMO evidence or score unavailable future labels."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from dashboard_v2.data_access import prospective as data

TAIPEI = data.TAIPEI
NOW = datetime(2026, 11, 17, 10, tzinfo=TAIPEI)


def live_ridge(root, *, labels=True, future=False):
    """Test-only fake live schema; no supplied archive or real ledger is changed."""
    target = root / "ledger"
    target.mkdir()
    for name in ("signals.jsonl", "labels.jsonl"):
        if name == "labels.jsonl" and not labels:
            continue
        rows = [json.loads(line) for line in (data.SNAP / "ridge" / name).read_text().splitlines()]
        for row in rows:
            row.pop("demo_flag", None)
        if future:
            rows += [{key: value.replace("2026-11-10", "2026-11-18") if isinstance(value, str) else value
                      for key, value in row.items()} for row in rows[:5]]
        (target / name).write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    lines = (target / "signals.jsonl").read_text().splitlines()
    first_day = [line for line in lines if json.loads(line)["target_date"] == "2026-11-10"]
    return hashlib.sha256(("\n".join(first_day) + "\n").encode()).hexdigest()


def test_live_ridge_missing_labels_returns_unavailable_without_demo_fallback(tmp_path, monkeypatch):
    live_ridge(tmp_path, labels=False)
    monkeypatch.setenv("TW_RIDGE_ROOT", str(tmp_path))
    result = data.ridge_status(NOW)
    assert result["status"] == "DATA_UNAVAILABLE"
    assert result["source_labels"] is None and result["scorable_days"] == 0
    assert result["mae"] is None and result["official_claim_blocked"] is True
    assert result["first_day_prediction_sha256"] is None
    assert "fixtures" not in result["source_signals"]


def test_live_ridge_first_day_lock_failure_blocks_accepted_samples(tmp_path, monkeypatch):
    live_ridge(tmp_path)
    monkeypatch.setenv("TW_RIDGE_ROOT", str(tmp_path))
    result = data.ridge_status(NOW)
    assert result["data_kind"] == "LIVE_READONLY"
    assert result["first_day_ok"] is False and result["official_claim_blocked"] is True
    assert result["scorable_days"] == 0 and result["rows"] == 0
    assert result["mae"] is None and result["diagnostic_rows"] == 25
    assert "first_day_prediction_lock_mismatch" in result["integrity_reasons"]


def test_future_labels_are_excluded_even_with_matching_first_day_lock(tmp_path, monkeypatch):
    expected = live_ridge(tmp_path, future=True)
    monkeypatch.setenv("TW_RIDGE_ROOT", str(tmp_path))
    monkeypatch.setattr(data, "RIDGE_FIRST_DAY_LIVE", expected)
    result = data.ridge_status(NOW)
    assert result["first_day_ok"] is True
    assert result["scorable_days"] == 5 and result["rows"] == 25
    assert "2026-11-18" not in result["scored_days"]
    assert "2026-11-18" in result["pending_targets"]
    assert result["rejected_timing_rows"] == 5
    assert result["official_claim_blocked"] is True  # far below252days


def test_labels_observed_after_now_do_not_enter_earlier_scores(tmp_path, monkeypatch):
    expected = live_ridge(tmp_path)
    path = tmp_path / "ledger/labels.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows[:5]:
        row["label_filled_timestamp"] = "2026-11-17T15:05:00+08:00"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("TW_RIDGE_ROOT", str(tmp_path))
    monkeypatch.setattr(data, "RIDGE_FIRST_DAY_LIVE", expected)
    result = data.ridge_status(NOW)
    assert result["scorable_days"] == 4
    assert "2026-11-10" not in result["scored_days"]


def test_live_v211_without_published_signals_is_explicitly_unavailable(tmp_path, monkeypatch):
    payload = json.loads((data.SNAP / "v211_local/status.json").read_text())
    payload.pop("demo_flag", None)
    (tmp_path / "status.json").write_text(json.dumps(payload))
    (tmp_path / "signal_registry.json").write_text("{}")
    monkeypatch.setenv("TW_V211_LOCAL", str(tmp_path))
    result = data.v211_status_local()
    assert result["status"] == result["performance"] == "DATA_UNAVAILABLE"
    assert result["latest_signal_date"] is result["latest_signal_at"] is None
    assert result["metrics"] is None and result["official_claim_blocked"] is True


def test_live_v211_compares_real_checkout_and_never_synthetic_github(tmp_path, monkeypatch):
    monkeypatch.setenv("TW_V211_LOCAL", str(data.ROOT / "experiments/results/v2/v211"))
    monkeypatch.delenv("TW_GITHUB_REVIEW_ROOT", raising=False)
    monkeypatch.setattr(data, "GITHUB", None)
    assert data.compare_source_conflicts("2026-11-10") == []  # same real source
    status = data.github_research_status()
    assert "fixtures" not in (status["source_meta"] or "")
    monkeypatch.setenv("TW_GITHUB_REVIEW_ROOT", str(tmp_path))
    assert data.compare_source_conflicts("2026-11-10") == []  # missing, notDEMO
    status = data.github_research_status()
    assert status["source_meta"] is status["source_summary"] is None


def test_explicit_demo_records_keep_demo_identity_even_outside_fixture_path(tmp_path, monkeypatch):
    target = tmp_path / "ledger"
    target.mkdir()
    for name in ("signals.jsonl", "labels.jsonl"):
        (target / name).write_bytes((data.SNAP / "ridge" / name).read_bytes())
    monkeypatch.setenv("TW_RIDGE_ROOT", str(tmp_path))
    result = data.ridge_status(NOW)
    assert result["data_kind"] == "SYNTHETIC_DEMO" and result["official_claim_blocked"] is True


def test_live_sync_without_evidence_never_borrows_demo_manifest_or_time(tmp_path, monkeypatch):
    monkeypatch.setenv("TW_V211_LOCAL", str(tmp_path))
    sync = data.sync_manifest()
    assert sync["status"] == "MISSING" and sync["synced_at"] is None
    assert "fixtures" not in sync["source"]
    assert data._freshness(NOW, {"prices_through": "2026-11-16"}, sync)["evidence_status"] == "UNKNOWN"
    # Copying the public DEMO manifest into a live root does not authenticate it.
    (tmp_path / "SYNC_MANIFEST.json").write_bytes((data.SNAP / "matrix/SYNC_MANIFEST.json").read_bytes())
    copied_demo = data.sync_manifest()
    assert copied_demo["status"] == "MISSING" and copied_demo["synced_at"] is None
    assert copied_demo["reason"] == "synthetic_sync_is_not_live_evidence"


def test_live_github_summary_chooses_newest_valid_date_not_fixed_1117(tmp_path, monkeypatch):
    for day, status in (("2026-11-17", "older"), ("2026-11-18", "newest"),
                        ("2099-99-99", "invalid"), ("latest", "not-a-date")):
        path = tmp_path / "outputs/daily_summary" / day / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"output_day": day, "status": status}))
    monkeypatch.setenv("TW_GITHUB_REVIEW_ROOT", str(tmp_path))
    result = data.github_research_status()
    assert result["research_as_of_date"] == "2026-11-18"
    assert result["orchestrator_status"] == "newest"
    assert result["source_summary"].endswith("2026-11-18/summary.json")
