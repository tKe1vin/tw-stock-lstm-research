"""Actual-run journals remain durable when scheduling produces no forecasts."""
import json

import pytest

from scripts import daily_reports


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_reports, "ROOT", tmp_path)
    return tmp_path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def attempt(**updates):
    value = {"run_date": "2026-11-17", "output_day": "2026-11-16", "completed_session": "2026-11-16",
             "started": "2026-11-17T07:01:53.255335+08:00", "status": "skipped",
             "reason": "today_has_no_completed_session", "steps": {}}
    value.update(updates)
    return value


def test_empty_delayed_attempt_is_journaled_without_erasing_latest_success(root):
    latest = root / "reports/latest/four_systems.json"
    save(latest, {"as_of_date": "2026-11-10", "status": "complete",
                  "models": {model: {"current": True} for model in daily_reports.MODELS}})
    text = root / "reports/latest/summary.md"
    text.write_text("Original complete report", encoding="utf-8")
    before = latest.read_bytes()
    run = attempt()
    first = daily_reports.build_reports(run, "2026-11-16")
    second = daily_reports.build_reports(run, "2026-11-16")
    assert first == second
    assert first["publication_status"] == "attempt_only"
    assert latest.read_bytes() == before and text.read_text() == "Original complete report"
    paths = list((root / "reports/attempts/2026-11-17").glob("*.json"))
    assert len(paths) == 1
    recorded = json.loads(paths[0].read_text())
    assert recorded["run_date"] == "2026-11-17"
    assert recorded["as_of_date"] == "2026-11-16"
    assert recorded["delayed_run"] and not recorded["historical_signal_backfill_performed"]
    assert "07:01:53" in (paths[0].parent / "summary.md").read_text()


def test_older_attempt_cannot_move_latest_backwards(root):
    latest = root / "reports/latest/four_systems.json"
    save(latest, {"as_of_date": "2026-11-16", "status": "partial", "models": {}})
    before = latest.read_bytes()
    run = attempt(status="waiting", steps={"444": {"status": "waiting", "exit_code": 2}})
    report = daily_reports.build_reports(run, "2026-11-13")
    assert report["preserved_reason"] == "older_session_cannot_replace_latest_report"
    assert latest.read_bytes() == before
    assert not (root / "reports/daily/2026-11-13/status.json").exists()


def test_rejected_success_still_gets_an_immutable_failure_attempt(root):
    run = attempt(status="failed", steps={"444": {"status": "success", "exit_code": 0}})
    with pytest.raises(ValueError, match="missing its current"):
        daily_reports.build_reports(run, "2026-11-16")
    records = list((root / "reports/attempts/2026-11-17").glob("*.json"))
    assert len(records) == 1
    value = json.loads(records[0].read_text())
    assert value["report"]["publication_status"] == "rejected"
    assert not (root / "reports/latest/four_systems.json").exists()


def test_news_freshness_is_coverage_metadata_not_article_content(root):
    save(root / "experiments/results/v2/v211/market/news_scan_coverage.json", {
        "calendar_days": {"2026-11-16": {"articles": 43, "text": "do not publish provider content"}},
        "signal_sessions": {"2026-11-16": {"source_start": "2026-11-13", "source_end": "2026-11-16",
                                              "completed_at": "2026-11-17T07:05:00+08:00"}}})
    report = daily_reports.build_reports(attempt(), "2026-11-16")
    assert report["news"]["signal_session_scan_complete"]
    assert report["news"]["articles_on_signal_calendar_day"] == 43
    assert not report["news"]["historical_availability_retroactively_certified"]
    assert "do not publish provider content" not in json.dumps(report)


def test_attempt_record_cannot_be_replaced_in_place(root):
    run = attempt()
    path = daily_reports._attempt_path(run, "2026-11-16")
    daily_reports._write_attempt(run, "2026-11-16", {"status": "skipped"}, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="cannot be overwritten"):
        daily_reports._write_attempt(run, "2026-11-16", {"status": "complete"}, path)
    assert path.read_bytes() == before


def test_reused_444_555_does_not_block_tracker_status_upgrade():
    old = {"as_of_date": "2026-11-16", "status": "partial", "models": {
        model: {"current": model != "v211"} for model in daily_reports.MODELS}}
    run = {"steps": {model: {"reason": "already_completed_for_this_session"}
                     for model in ("444", "555")}}
    current = {model: {"current": True} for model in daily_reports.MODELS}
    assert daily_reports._preserve_reason(run, "2026-11-16", old, old, current) is None


def test_reused_444_555_keeps_good_report_when_no_other_system_improves():
    old = {"as_of_date": "2026-11-16", "status": "partial", "models": {
        model: {"current": model != "v211"} for model in daily_reports.MODELS}}
    run = {"steps": {model: {"reason": "already_completed_for_this_session"}
                     for model in ("444", "555")}}
    assert daily_reports._preserve_reason(run, "2026-11-16", old, old, old["models"]) == "already_published_444_555_for_this_session"
