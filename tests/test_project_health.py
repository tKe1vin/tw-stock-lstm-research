"""Health diagnostics remain read-only and separate waiting from broken evidence."""
import builtins
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts import project_health as health
from scripts.common import TICKERS, TZ


def market(root, day="2026-11-18", *, known_news=True):
    folder = root / health.MARKET
    folder.mkdir(parents=True)
    pd.DataFrame([{"Date": day, "ticker": symbol, "Close": 100.0} for symbol in [*TICKERS, "^TWII"]]).to_csv(folder / "prices.csv", index=False)
    pd.DataFrame([{"Date": day, "ticker": symbol, "Daily_Sentiment": 0.0, "News_Count": 0,
                   "Latest_News_Timestamp": None, "Source_Status": "verified_no_matching_article" if known_news else None,
                   "Scoring_Model": "neutral_no_news"} for symbol in TICKERS]).to_csv(folder / "sentiment.csv", index=False)


def test_market_waiting_and_unknown_news_cannot_be_certified_neutral(tmp_path):
    now = datetime(2026, 11, 19, 2, tzinfo=TZ)
    market(tmp_path, known_news=False)
    report = health.market_clock(tmp_path, now)
    assert report["waiting"] and not report["sentiment.csv"]["source_evidence_ready"]
    path = tmp_path / health.MARKET / "sentiment.csv"
    frame = pd.read_csv(path)
    frame.Source_Status = "verified_no_matching_article"
    frame.to_csv(path, index=False)
    assert not health.market_clock(tmp_path, now)["waiting"]
    # After the next completed session the same retained inputs are waiting,
    # while their correctness remains distinct from current-source availability.
    assert health.market_clock(tmp_path, datetime(2026, 11, 19, 16, tzinfo=TZ))["waiting"]


def test_future_canonical_source_fails_even_when_rows_and_values_look_valid(tmp_path):
    market(tmp_path, day="2026-11-19")
    with pytest.raises(health.HealthFailure, match="Future"):
        health.market_clock(tmp_path, datetime(2026, 11, 19, 2, tzinfo=TZ))


def test_future_sentiment_and_partial_prices_are_not_silently_accepted(tmp_path):
    market(tmp_path)
    sentiment = tmp_path / health.MARKET / "sentiment.csv"
    frame = pd.read_csv(sentiment)
    frame.Date = "2026-11-19"
    frame.to_csv(sentiment, index=False)
    with pytest.raises(health.HealthFailure, match="Future"):
        health.market_clock(tmp_path, datetime(2026, 11, 19, 2, tzinfo=TZ))
    prices = tmp_path / health.MARKET / "prices.csv"
    pd.read_csv(prices).iloc[:-1].to_csv(prices, index=False)
    with pytest.raises(health.HealthFailure, match="complete cross-section"):
        health.market_clock(tmp_path, datetime(2026, 11, 19, 2, tzinfo=TZ))


def test_failed_check_does_not_hide_remaining_checks_and_waiting_does_not_block_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(health, "environment", lambda _profile: {"ready": True})
    monkeypatch.setattr(health, "syntax", lambda _root: {"python_files": 1})
    monkeypatch.setattr(health, "frozen_bundle", lambda _root: {"verified": True})
    monkeypatch.setattr(health, "retained_state", lambda _root: {"verified": True})
    monkeypatch.setattr(health, "latest_research", lambda *_args: {"waiting": True})
    monkeypatch.setattr(health, "historical_research", lambda _root: {"verified": True})
    monkeypatch.setattr(health, "market_clock", lambda *_args: {"waiting": True})
    now = datetime(2026, 11, 19, 2, tzinfo=TZ)
    result = health.evaluate(tmp_path, now=now)
    assert result["status"] == "waiting" and result["exit_code"] == 0
    assert set(result["waiting_checks"]) == {"market_clock", "latest_research"}
    def broken(_root):
        raise health.HealthFailure("changed source checksum")
    monkeypatch.setattr(health, "retained_state", broken)
    result = health.evaluate(tmp_path, now=now)
    assert result["status"] == "fail" and result["exit_code"] == 1
    assert result["failed_checks"] == ["immutable_daily_state"]
    assert result["checks"][-1]["name"] == "retained_research"


def test_saved_evidence_checks_do_not_import_training_fetch_or_write_files(monkeypatch):
    root = health.ROOT
    folders = ("data", "reports", "models", "research", "experiments/results/v2/v211")
    def snapshot():
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for folder in folders for path in (root / folder).rglob("*")
                if path.is_file() and "__pycache__" not in path.parts}
    before = snapshot()
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"tensorflow", "torch", "transformers", "yfinance", "requests"}:
            pytest.fail(f"A retained health check imported a training/network runtime: {name}")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = health.evaluate(profile="dashboard")
    # A production-only test environment need not install the optional UI. All
    # retained checks must still run; environment availability is separately visible.
    assert all(item["status"] == "pass" for item in result["checks"] if item["name"] != "environment")
    assert snapshot() == before


def test_timezone_is_required_for_daily_health_clock(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        health.evaluate(tmp_path, now=datetime(2026, 11, 19))


def test_missing_partial_report_is_waiting_but_claimed_complete_or_corrupt_report_is_failure(tmp_path):
    path = tmp_path / "reports/latest/four_systems.json"
    path.parent.mkdir(parents=True)
    state = {"as_of_date": "2026-11-18", "run_date": "2026-11-19", "status": "partial",
             "research_status": "waiting", "primary_research": {"status": "waiting"},
             "models": {name: {"status": "waiting", "current": False} for name in ("444", "555", "555_v2", "v211")}}
    path.write_text(json.dumps(state))
    now = datetime(2026, 11, 19, 2, tzinfo=TZ)
    result = health.latest_research(tmp_path, now, "core")
    assert result["waiting"] and result["paired_outputs"] == 0
    state["research_status"] = "complete"
    path.write_text(json.dumps(state))
    with pytest.raises(health.HealthFailure, match="Missing"):
        health.latest_research(tmp_path, now, "core")
    state["research_status"] = "waiting"
    path.write_text(json.dumps(state))
    manifest = tmp_path / "reports/daily/2026-11-18/paired/reproduction_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("broken partial JSON")
    with pytest.raises(ValueError):
        health.latest_research(tmp_path, now, "core")
