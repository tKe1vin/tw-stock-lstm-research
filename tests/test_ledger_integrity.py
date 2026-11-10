"""Committed post-seed inputs and realized rows are append-only, including ZIP use."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from experiments.v2 import v211
from scripts import ledger_integrity as ledger
from scripts import persist_daily_state as persistence


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, check=True)
    git("init", "-b", "main")
    git("config", "user.name", "Ledger test")
    git("config", "user.email", "ledger@example.invalid")
    (tmp_path / "README.md").write_text("fixture\n")
    git("add", ".")
    git("commit", "-m", "fixture")
    book = tmp_path / persistence.BOOK_ROOTS[0]
    (book / "market").mkdir(parents=True)
    (book / "market/prices.csv").write_text("Date,ticker,Close\n2026-11-05,2330.TW,100\n2026-11-06,2330.TW,101\n")
    (book / "market/sentiment.csv").write_text("Date,ticker,Daily_Sentiment\n2026-11-05,2330.TW,0.1\n2026-11-06,2330.TW,0.2\n")
    git("add", ".")
    git("commit", "-m", "committed post-seed source")
    return tmp_path, book, git


@pytest.mark.parametrize("name,old,new", [("prices.csv", b",101", b",102"),
                                          ("sentiment.csv", b",0.2", b",0.3")])
def test_postseed_committed_source_revision_is_rejected(repo, name, old, new):
    root, book, _ = repo
    path = book / "market" / name
    path.write_bytes(path.read_bytes().replace(old, new))
    with pytest.raises(ValueError, match="Committed ledger history changed"):
        ledger.validate_ledger_sources(root, book)
    with pytest.raises(ValueError, match="Committed ledger history changed"):
        persistence.validate_state(root, {})


def test_only_newer_source_dates_can_be_appended(repo):
    root, book, _ = repo
    path = book / "market/prices.csv"
    before = path.read_bytes()
    path.write_bytes(before + b"2026-11-09,2330.TW,103\n")
    assert ledger.validate_ledger_sources(root, book)["checked_baselines"] == 2
    assert path.read_bytes().startswith(before)
    path.write_bytes(before + b"2026-11-06,2317.TW,200\n")
    with pytest.raises(ValueError, match="only append later dates"):
        ledger.validate_ledger_sources(root, book)


def test_deleted_committed_output_is_rejected(repo):
    root, book, git = repo
    path = book / "matured_scores.csv"
    path.write_text("Signal_Date,Ticker,Predicted,Actual_5d_Pct\n2026-11-05,2330.TW,0.3,0.4\n")
    git("add", ".")
    git("commit", "-m", "realized score")
    path.unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        ledger.validate_ledger_sources(root, book)


def test_score_append_preserves_original_bytes_and_is_idempotent(tmp_path):
    path = tmp_path / "matured_scores.csv"
    old = b"Signal_Date,Ticker,Predicted,Actual_5d_Pct\r\n2026-11-05,2330.TW,0.300000,0.400000\r\n"
    path.write_bytes(old)
    candidate = pd.DataFrame([{"Signal_Date": "2026-11-05", "Ticker": "2330.TW", "Predicted": .3, "Actual_5d_Pct": .4},
                              {"Signal_Date": "2026-11-06", "Ticker": "2330.TW", "Predicted": .5, "Actual_5d_Pct": .6}])
    keys = ("Signal_Date", "Ticker")
    assert ledger.apply_append(path, ledger.append_plan(path, candidate, keys)) == 1
    saved = path.read_bytes()
    assert saved.startswith(old)
    assert ledger.apply_append(path, ledger.append_plan(path, candidate, keys)) == 0
    assert path.read_bytes() == saved
    candidate.loc[0, "Actual_5d_Pct"] = 2.
    with pytest.raises(ValueError, match="Previously realized ledger row changed"):
        ledger.append_plan(path, candidate, keys)
    assert path.read_bytes() == saved


def test_no_git_first_observation_stays_unverified_and_read_only(tmp_path, monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("git unavailable")
    monkeypatch.setattr(ledger.subprocess, "run", absent)
    book = tmp_path / "book"
    book.mkdir()
    before = list(book.iterdir())
    result = ledger.validate_ledger_sources(tmp_path, book)
    assert result == {"checked_baselines": 0, "baseline": "legacy_first_observation_unverified"}
    assert list(book.iterdir()) == before


def test_portable_checkpoint_protects_appends_without_git(tmp_path, monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("git unavailable")
    monkeypatch.setattr(ledger.subprocess, "run", absent)
    book = tmp_path / "book"
    (book / "market").mkdir(parents=True)
    source = book / "market/prices.csv"
    source.write_text("Date,ticker,Close\n2026-11-05,2330.TW,100\n")
    panel = pd.DataFrame({"2330.TW": [100.]}, index=pd.to_datetime(["2026-11-05"]))
    ledger.record_ledger_sources(tmp_path, book, panel, "fixture", {"matured_scores.csv": 1})
    checkpoint = json.loads((book / ledger.CHECKPOINT).read_text())
    audit = json.loads((book / ledger.AUDIT).read_text())
    assert not checkpoint["historical_availability_certified"]
    assert checkpoint["initial_history_status"] == "legacy_first_observation_unverified"
    assert audit["panel_sha256"] and audit["file_prefixes"]["market/prices.csv"]["sha256"]
    assert ledger.validate_ledger_sources(tmp_path, book)["checked_baselines"] == 1
    source.write_text(source.read_text().replace(",100", ",101"))
    with pytest.raises(ValueError, match="Committed ledger history changed"):
        ledger.validate_ledger_sources(tmp_path, book)


def _tracker(tmp_path, monkeypatch):
    book = tmp_path / "book"
    live = book / "daily"
    live.mkdir(parents=True)
    monkeypatch.setattr(v211, "ROOT", tmp_path)
    monkeypatch.setattr(v211, "V211", book)
    monkeypatch.setattr(v211, "LIVE", live)
    return book, live


def _signal(day, entry):
    return pd.DataFrame([{"Signal_Date": day, "Planned_Execution_Date": entry,
                          "Ticker": ticker, "Predicted_Return_Pct": .1,
                          "Target_Weight": .1, "Risk_Off": True, "Close": 100.}
                         for ticker in v211.TICKERS])


def _context(index):
    panel = pd.DataFrame(100., index=index, columns=v211.TICKERS)
    actual = pd.DataFrame(.2, index=index, columns=v211.TICKERS)
    return panel, None, None, None, actual, None


def test_matured_tracker_rerun_cannot_replace_realized_score(tmp_path, monkeypatch):
    book, live = _tracker(tmp_path, monkeypatch)
    _signal("2026-11-05", "2026-11-06").to_csv(live / "2026-11-05.csv", index=False)
    context = _context(pd.bdate_range("2026-11-05", periods=7))
    assert v211.score_matured(context)["matured_signal_days"] == 1
    saved = {path.name: path.read_bytes() for path in book.glob("*") if path.is_file()}
    assert v211.score_matured(context)["matured_signal_days"] == 1
    assert {path.name: path.read_bytes() for path in book.glob("*") if path.is_file()} == saved
    altered = list(context)
    altered[4] = context[4] * 2
    with pytest.raises(ValueError, match="Previously realized ledger row changed"):
        v211.score_matured(tuple(altered))
    assert {path.name: path.read_bytes() for path in book.glob("*") if path.is_file()} == saved


def test_invalid_new_matured_score_is_rejected_before_write(tmp_path, monkeypatch):
    book, live = _tracker(tmp_path, monkeypatch)
    _signal("2026-11-05", "2026-11-06").to_csv(live / "2026-11-05.csv", index=False)
    context = list(_context(pd.bdate_range("2026-11-05", periods=7)))
    context[4].iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        v211.score_matured(tuple(context))
    assert not (book / "matured_scores.csv").exists()
    assert not (book / ledger.CHECKPOINT).exists()


def test_live_book_append_preserves_fills_nav_and_snapshot_bytes(tmp_path, monkeypatch):
    book, live = _tracker(tmp_path, monkeypatch)
    _signal("2026-11-05", "2026-11-06").to_csv(live / "2026-11-05.csv", index=False)
    early = _context(pd.bdate_range("2026-11-06", periods=2))
    assert v211.update_book(early)["status"] == "updated"
    names = ["live_fills.csv", "live_nav.csv", "equal_nav.csv"]
    before = {name: (book / name).read_bytes() for name in names}
    later = _context(pd.bdate_range("2026-11-06", periods=3))
    assert v211.update_book(later)["status"] == "updated"
    assert all((book / name).read_bytes().startswith(before[name]) for name in names)
    saved = {path.name: path.read_bytes() for path in book.glob("*") if path.is_file()}
    assert v211.update_book(later)["status"] == "updated"
    assert {path.name: path.read_bytes() for path in book.glob("*") if path.is_file()} == saved
    changed = list(later)
    changed[0] = later[0].copy()
    changed[0].iloc[0] = 105.
    with pytest.raises(ValueError, match="Previously realized ledger row changed"):
        v211.update_book(tuple(changed))
    assert {path.name: path.read_bytes() for path in book.glob("*") if path.is_file()} == saved


def test_allowlist_includes_new_checkpoint_and_source_evidence():
    for book in persistence.BOOK_ROOTS:
        assert persistence.allowed_path(book + "/" + ledger.CHECKPOINT)
        assert persistence.allowed_path(book + "/" + ledger.AUDIT)
