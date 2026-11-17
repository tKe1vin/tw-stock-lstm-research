"""Retained journal browsing preserves dates, failures and original evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from dashboard import journals

ROOT = Path(__file__).resolve().parents[1]


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_existing_dates_and_original_documents_are_not_reconstructed():
    dates = journals.available_dates(root=ROOT)
    assert dates == sorted(dates, reverse=True)
    assert "2026-10-29" in dates and "2026-11-16" in dates
    documents = journals.daily_documents("2026-11-16", ROOT)
    assert "444 日誌" in documents and "555 日誌" in documents and "主研究配對比較" in documents
    document = journals.read_document(documents["444 日誌"], ROOT)
    assert document["text"] == (ROOT / document["path"]).read_text(encoding="utf-8")
    assert "2026-11-17T13:15:11" in document["text"]
    status = journals.daily_context("2026-11-16", ROOT)
    assert status["run_date"] == "2026-11-17"


def test_cumulative_filters_retain_original_delayed_entry_and_history():
    document = journals.read_document("reports/journals/444.md", ROOT)
    entries = journals.journal_entries(document["text"])
    delayed = journals.filtered_entries(entries, "2026-11-16", "延後")
    assert len(delayed) == 1
    assert "2026-11-17T13:15:11" in delayed[0]["text"]
    assert "未下單或成交" in delayed[0]["text"]
    assert all(entry["text"] in document["text"] for entry in entries)
    assert journals.filtered_entries(entries, search="文字未曾出現unmatched") == []


def test_actual_attempts_keep_failure_and_maintenance_not_current_success():
    records = journals.attempt_records("2026-11-17", ROOT)
    assert len(records) >= 3
    assert all(row["run_date"] == "2026-11-17" and row["as_of_date"] == "2026-11-16" for row in records)
    assert any(row["status"] == "failed" for row in records)
    assert any(row["steps"]["555"]["status"] == "failed" for row in records)
    assert all(row["steps"]["v211"]["status"] == "maintenance_only" for row in records)


def test_missing_dates_and_journals_stay_missing(tmp_path):
    assert journals.available_dates(root=tmp_path) == []
    assert journals.daily_context("2026-11-16", tmp_path) is None
    assert journals.daily_documents("2026-11-16", tmp_path) == {}
    with pytest.raises(journals.JournalError):
        journals.read_document("reports/journals/555_v2.md", tmp_path)


@pytest.mark.parametrize("relative", ["/etc/passwd", "reports/journals/../../.env",
                                      "reports/daily/2026-11-16/../../../data/news_scored_cache.csv"])
def test_journal_reader_rejects_arbitrary_paths(relative, tmp_path):
    with pytest.raises(journals.JournalError):
        journals.read_document(relative, tmp_path)


def test_symlink_cannot_leak_outside_or_other_repository_files(tmp_path):
    source = write(tmp_path, "private.txt", "private source")
    target = tmp_path / "reports/journals/444.md"
    target.parent.mkdir(parents=True)
    target.symlink_to(source)
    with pytest.raises(journals.JournalError, match="符號連結"):
        journals.read_document("reports/journals/444.md", tmp_path)
    assert journals.available_dates(root=tmp_path) == []


def test_contradictory_or_malformed_timing_is_not_displayed_as_true(tmp_path):
    status_path = "reports/daily/2026-11-16/status.json"
    write(tmp_path, status_path, json.dumps({"as_of_date": "2026-11-17", "run_date": "2026-11-17"}))
    with pytest.raises(journals.JournalError, match="保存目錄矛盾"):
        journals.daily_context("2026-11-16", tmp_path)
    write(tmp_path, "reports/attempts/2026-11-17/attempt.json", json.dumps({
        "run_date": "2026-11-17", "as_of_date": "2026-11-16", "started": "2026-11-16T13:00:00+08:00"}))
    with pytest.raises(journals.JournalError, match="啟動日期"):
        journals.attempt_records("2026-11-17", tmp_path)


def test_relative_original_report_links_point_to_github_not_streamlit_paths():
    text = "[主研究](paired/comparison.md) [新聞](https://example.org/article)"
    rendered = journals.display_markdown(text, "reports/daily/2026-11-16/summary.md")
    assert journals.GITHUB + "reports/daily/2026-11-16/paired/comparison.md" in rendered
    assert "https://example.org/article" in rendered
    assert text == "[主研究](paired/comparison.md) [新聞](https://example.org/article)"


def app(root=ROOT):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string(
        f"from dashboard.journals import render_journals\nrender_journals(root={str(root)!r})",
        default_timeout=30).run()


def snapshot():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "reports").rglob("*") if path.is_file()}


def test_journal_ui_switches_all_modes_and_search_without_writes_or_runtime(monkeypatch):
    before = snapshot()
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    at = app()
    assert not at.exception
    assert at.selectbox(key="journals_daily_date").value == "2026-11-16"
    assert any(metric.label == "真正執行日" and metric.value == "2026-11-17" for metric in at.metric)
    at.selectbox(key="journals_daily_document").set_value("555 日誌").run()
    assert not at.exception
    at.radio(key="journals_mode").set_value("累積日誌").run()
    at.selectbox(key="journals_cumulative_date").set_value("2026-11-16").run()
    at.text_input(key="journals_search").set_value("延後").run()
    assert not at.exception and len(at.expander) == 1
    at.radio(key="journals_mode").set_value("實際執行紀錄").run()
    assert not at.exception and len(at.dataframe) == 1
    assert any("maintenance_only" in str(item.value) for item in at.json)
    at.radio(key="journals_mode").set_value("最新摘要").run()
    assert not at.exception
    assert snapshot() == before


def test_journal_ui_empty_checkout_shows_missing_without_fake_report(tmp_path):
    at = app(tmp_path)
    assert not at.exception
    assert not at.metric and len(at.info) == 1
