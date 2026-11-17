"""Display metadata acquisition cannot rewrite historical scoring evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import capture_news_display as capture
from scripts.persist_daily_state import allowed_path, validate_state


def fixture_cache(root):
    path = root / capture.CACHES[0]
    path.parent.mkdir(parents=True)
    path.write_text("news_id,ticker,Date\n123,2330.TW,2026-10-01\n456,2317.TW,2026-10-01\n")
    return path


def test_capture_appends_only_display_fields_and_keeps_actual_observation_time(tmp_path):
    cache = fixture_cache(tmp_path)
    original = cache.read_bytes()
    clock = lambda: datetime.fromisoformat("2026-11-17T15:00:00+08:00")
    first = capture.capture(tmp_path, limit=1, fetcher=lambda identity: "當前標題 " + identity, clock=clock)
    path = tmp_path / capture.METADATA
    prior = path.read_bytes()
    second = capture.capture(tmp_path, fetcher=lambda identity: "補充標題 " + identity, clock=clock)
    assert first["captured"] == second["captured"] == 1
    assert path.read_bytes().startswith(prior) and cache.read_bytes() == original
    rows = capture.validate_metadata(path, prior)
    assert all(row["purpose"] == capture.PURPOSE and row["observed_at"] == clock().isoformat() for row in rows)
    assert not first["historical_availability_proof"]
    assert all("content" not in row and "score" not in row for row in rows)
    assert capture.capture(tmp_path, fetcher=lambda identity: pytest.fail("already captured"))["captured"] == 0


def test_failed_sources_are_missing_and_invalid_batch_does_not_change_existing_file(tmp_path):
    fixture_cache(tmp_path)
    def fail(_identity):
        raise TimeoutError("provider unavailable")
    result = capture.capture(tmp_path, fetcher=fail)
    assert result["captured"] == 0 and len(result["failures"]) == 2
    assert not (tmp_path / capture.METADATA).exists()
    with pytest.raises(ValueError):
        capture.capture(tmp_path, fetcher=lambda _: "bad\ntitle")
    assert not (tmp_path / capture.METADATA).exists()


def test_parser_extracts_only_identifiable_title_and_rejects_other_article():
    assert capture.parse_title('<meta property="og:title" content="台積電 &amp; 新聞">', "123") == "台積電 & 新聞"
    with pytest.raises(ValueError):
        capture.parse_title('<meta property="og:title" content="新聞"><meta property="og:url" content="https://news.cnyes.com/news/id/456">', "123")
    with pytest.raises(ValueError):
        capture.article_url("../../.env")


def test_default_command_is_read_only_and_cannot_fetch(monkeypatch, tmp_path, capsys):
    fixture_cache(tmp_path)
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    # saved_ids defaults bind the project root; explicit patch keeps the fixture small.
    monkeypatch.setattr(capture, "saved_ids", lambda: ["123", "456"])
    monkeypatch.setattr(capture, "capture", lambda **kwargs: pytest.fail("default must not acquire"))
    assert capture.main([]) == 0
    assert json.loads(capsys.readouterr().out)["read_only"] is True
    assert not (tmp_path / capture.METADATA).exists()


def test_persistence_rejects_changed_committed_titles_and_full_article_payload(tmp_path):
    fixture_cache(tmp_path)
    capture.capture(tmp_path, fetcher=lambda _: "展示標題")
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("add", ".")
    git("commit", "-m", "display fixture")
    validate_state(tmp_path, immutable={})
    path = tmp_path / capture.METADATA
    original = path.read_bytes()
    path.write_bytes(original.replace("展示標題".encode(), "修改標題".encode()))
    with pytest.raises(ValueError, match="unchanged prefix"):
        validate_state(tmp_path, immutable={})
    row = json.loads(original.splitlines()[0])
    row["content"] = "full article is not display-title metadata"
    path.write_bytes(original + (json.dumps(row) + "\n").encode())
    with pytest.raises(ValueError, match="compact"):
        validate_state(tmp_path, immutable={})
    assert allowed_path(capture.METADATA) and not allowed_path("data/news_full_article_dump.jsonl")
