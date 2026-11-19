"""Bounded source workers preserve saved inputs and do not mask integrity failures."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
import pytest

from experiments.v2 import v211_daily
from experiments.v2 import v211_market as market
from scripts import update_market_data as snapshots


NOW = datetime.fromisoformat("2026-11-09T17:37:00+08:00")


@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("error", ["KeyError: 'price_seed_sha256'", "JSONDecodeError: malformed manifest"])
def test_integrity_gate_is_a_failure_even_when_an_existing_signal_is_present(published, error):
    status = {"ok": published, "signal_published": published,
              "signal": {"completed_session": "2026-11-09", "reason": "already_written" if published else None}}
    report = {"ready_to_predict": False, "gate": "integrity_error", "error": error,
              "completed_session": "2026-11-09"}
    assert v211_daily.classify_exit(status, report, lambda day: published, NOW) == 1


@pytest.fixture
def saved_market(tmp_path, monkeypatch):
    folder = tmp_path / "canonical"
    folder.mkdir()
    for field in ("PRICES", "SENTIMENT", "PRICE_SEED", "SENTIMENT_SEED", "MANIFEST",
                  "STATE", "SCAN_COVERAGE", "CACHE", "OFFICIAL_PRICES", "OFFICIAL_SENTIMENT"):
        monkeypatch.setattr(market, field, folder / (field.lower() + (".json" if field in
                           {"MANIFEST", "STATE", "SCAN_COVERAGE"} else ".csv")))
    monkeypatch.setattr(snapshots, "ROOT", tmp_path)
    for module in (snapshots, market):
        monkeypatch.setattr(module, "current_completed_session", lambda now=None: pd.Timestamp("2026-11-09"))
    price_rows = [{"Date": day, "ticker": ticker, "Close": 100.}
                  for day in ("2026-11-06", "2026-11-09") for ticker in market.NAMES]
    prices = pd.DataFrame(price_rows)
    prices[prices.Date == "2026-11-06"].to_csv(market.PRICES, index=False)
    sentiment = pd.DataFrame([
        {"Date": "2026-11-06", "ticker": ticker, "Daily_Sentiment": 0., "News_Count": 0,
         "Latest_News_Timestamp": "", "Source_Status": "verified_no_matching_article",
         "Scoring_Model": "neutral_no_news"} for ticker in market.TICKERS
    ])
    sentiment.to_csv(market.SENTIMENT, index=False)
    for source, copies in ((market.PRICES, (market.PRICE_SEED, market.OFFICIAL_PRICES)),
                           (market.SENTIMENT, (market.SENTIMENT_SEED, market.OFFICIAL_SENTIMENT))):
        for path in copies:
            path.write_bytes(source.read_bytes())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    market.MANIFEST.write_text(json.dumps({
        "price_seed_sha256": digest(market.PRICE_SEED),
        "sentiment_seed_sha256": digest(market.SENTIMENT_SEED),
        "price_seed_through": "2026-11-06", "sentiment_seed_through": "2026-11-06",
        "official_price_sha256_at_seed": digest(market.OFFICIAL_PRICES),
        "official_sentiment_sha256_at_seed": digest(market.OFFICIAL_SENTIMENT),
    }))
    return prices, sentiment


def _worker_script(monkeypatch, tmp_path, body):
    script = tmp_path / "worker_fixture.py"
    script.write_text("import sys\nfrom pathlib import Path\n" + body)
    monkeypatch.setattr(snapshots, "_worker_command",
                        lambda now, report: [sys.executable, str(script), str(report)])


def _complete_saved_market(saved_market):
    prices, sentiment = saved_market
    prices.to_csv(market.PRICES, index=False)
    current = sentiment.assign(Date="2026-11-09")
    pd.concat([sentiment, current]).to_csv(market.SENTIMENT, index=False)


@pytest.mark.skipif(os.name != "posix", reason="The production Linux runner uses POSIX process groups")
def test_timeout_kills_descendants_and_materializes_a_completed_price_append(saved_market, tmp_path, monkeypatch):
    prices, _sentiment = saved_market
    original_price = market.PRICES.read_bytes()
    original_sentiment = market.SENTIMENT.read_bytes()
    marker = tmp_path / "late-write"
    child_pid = tmp_path / "child-pid"
    child = ("import time\nfrom pathlib import Path\n"
             f"time.sleep(1)\nPath({str(marker)!r}).write_text('orphan wrote after timeout')\ntime.sleep(60)")
    _worker_script(monkeypatch, tmp_path,
                   "import subprocess, time\n"
                   f"Path({str(market.PRICES)!r}).write_bytes({prices.to_csv(index=False).encode()!r})\n"
                   f"child = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
                   f"Path({str(child_pid)!r}).write_text(str(child.pid))\n"
                   "time.sleep(60)\n")
    report = snapshots.build_supervised_snapshot(NOW, timeout_seconds=.3, clock=lambda: NOW)
    assert report["update_timed_out"] and "exceeded" in report["update_error"]
    assert report["prices_ready"] and report["snapshot_prices_ready"]
    assert not report["sentiment_ready"] and not report["snapshot_sentiment_ready"]
    assert snapshots._exit_code(report) == 2
    assert market.PRICES.read_bytes().startswith(original_price)
    assert market.SENTIMENT.read_bytes() == original_sentiment
    folder = snapshots.snapshot_dir("2026-11-09")
    assert (folder / "prices.csv").read_bytes() == market.PRICES.read_bytes()
    assert not (folder / "sentiment.csv").exists()
    assert json.loads((folder / "manifest.json").read_text())["update_timed_out"]
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split()[2] == "Z":
            break
        time.sleep(.01)
    else:
        pytest.fail("Source worker descendant survived the deadline")
    assert not marker.exists()


def test_timeout_reuses_complete_current_inputs_with_explicit_failure_evidence(saved_market, tmp_path, monkeypatch):
    _complete_saved_market(saved_market)
    original = {path: path.read_bytes() for path in (market.PRICES, market.SENTIMENT, market.MANIFEST)}
    _worker_script(monkeypatch, tmp_path, "import time\ntime.sleep(60)\n")
    report = snapshots.build_supervised_snapshot(NOW, timeout_seconds=.1, clock=lambda: NOW)
    assert report["update_timed_out"] and report["used_existing_current_data"]
    assert report["ready_to_predict"] and snapshots._exit_code(report) == 0
    assert "exceeded" in report["error"]
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("kind", ["failed", "invalid_json", "invalid_fields"])
@pytest.mark.parametrize("existing_complete", [False, True])
def test_worker_failures_keep_market_report_and_only_reuse_verified_current_inputs(
        saved_market, tmp_path, monkeypatch, kind, existing_complete):
    if existing_complete:
        _complete_saved_market(saved_market)
    body = {"failed": "sys.exit(7)\n",
            "invalid_json": "Path(sys.argv[1]).write_text('not JSON')\n",
            "invalid_fields": "Path(sys.argv[1]).write_text('{}')\n"}[kind]
    _worker_script(monkeypatch, tmp_path, body)
    report = snapshots.build_supervised_snapshot(NOW, timeout_seconds=2, clock=lambda: NOW)
    assert report["update_timed_out"] is False
    assert report["completed_session"] == "2026-11-09"
    assert report["update_error"] and report["error"]
    assert report["used_existing_current_data"] is existing_complete
    assert snapshots._exit_code(report) == (0 if existing_complete else 1)
    assert report["prices_ready"] is existing_complete


def test_successful_worker_retains_source_failure_details_and_partial_readiness(saved_market, tmp_path, monkeypatch):
    prices, _sentiment = saved_market
    prices.to_csv(market.PRICES, index=False)
    worker_report = {"completed_session": "2026-11-09", "prices_ready": True,
                     "sentiment_ready": False, "ready_to_predict": False,
                     "appended_price_dates": ["2026-11-09"], "appended_sentiment_dates": [],
                     "sentiment_fetch_error": "HTTPError: 503", "error": "sentiment: HTTPError: 503"}
    _worker_script(monkeypatch, tmp_path, f"Path(sys.argv[1]).write_text({json.dumps(worker_report)!r})\n")
    report = snapshots.build_supervised_snapshot(NOW, timeout_seconds=2, clock=lambda: NOW)
    assert report["snapshot_prices_ready"] and not report["snapshot_sentiment_ready"]
    assert report["appended_price_dates"] == ["2026-11-09"]
    assert report["sentiment_fetch_error"] == "HTTPError: 503"
    assert snapshots._exit_code(report) == 2


def test_worker_integrity_failure_cannot_be_overridden_by_existing_ready_files(saved_market, tmp_path, monkeypatch):
    _complete_saved_market(saved_market)
    worker_report = {"completed_session": "2026-11-09", "prices_ready": False,
                     "sentiment_ready": False, "ready_to_predict": False,
                     "gate": "integrity_error", "error": "JSONDecodeError: malformed manifest"}
    _worker_script(monkeypatch, tmp_path, f"Path(sys.argv[1]).write_text({json.dumps(worker_report)!r})\n")
    report = snapshots.build_supervised_snapshot(NOW, timeout_seconds=2, clock=lambda: NOW)
    assert not report["ready_to_predict"]
    assert snapshots._exit_code(report) == 1


def test_corrupt_canonical_manifest_is_failure_even_with_preexisting_complete_snapshots(saved_market, tmp_path, monkeypatch):
    _complete_saved_market(saved_market)
    snapshots._snapshot_from_report(NOW, market.inspect_market(NOW), materialize=True)
    market.MANIFEST.write_text("malformed JSON")
    _worker_script(monkeypatch, tmp_path, "sys.exit(7)\n")
    report = snapshots.build_supervised_snapshot(NOW, timeout_seconds=2, clock=lambda: NOW)
    assert report["gate"] == "integrity_error"
    assert report["snapshot_prices_ready"] and report["snapshot_sentiment_ready"]
    assert not report["ready_to_predict"] and not report["used_existing_current_data"]
    assert snapshots._exit_code(report) == 1


@pytest.mark.parametrize("later_rows", [False, True])
def test_worker_crossing_close_keeps_the_parent_session_and_original_prefix_bytes(
        saved_market, tmp_path, monkeypatch, later_rows):
    started = NOW.replace(hour=14, minute=59)
    decision = NOW.replace(hour=15, minute=1)
    monkeypatch.setenv("TW_LSTM_TARGET_SESSION", "2026-11-06")
    monkeypatch.setenv("TW_LSTM_OUTPUT_DAY", "2026-11-06")
    monkeypatch.setenv("TW_LSTM_RUN_STARTED_AT", started.isoformat())
    monkeypatch.setattr(market, "current_completed_session", snapshots.current_completed_session)
    monkeypatch.setattr(snapshots, "current_completed_session", lambda now=None: pd.Timestamp(
        "2026-11-09" if now.hour >= 15 and now.date() == NOW.date() else "2026-11-06"))
    monkeypatch.setattr(market, "current_completed_session", snapshots.current_completed_session)
    original = {path: path.read_bytes() for path in (market.PRICES, market.SENTIMENT)}
    if later_rows:
        _complete_saved_market(saved_market)
    worker_report = {"completed_session": "2026-11-06", "prices_ready": True,
                     "sentiment_ready": True, "ready_to_predict": True}
    _worker_script(monkeypatch, tmp_path, f"Path(sys.argv[1]).write_text({json.dumps(worker_report)!r})\n")
    report = snapshots.build_supervised_snapshot(started, timeout_seconds=2, clock=lambda: decision)
    assert report["completed_session"] == "2026-11-06"
    assert report["run_started_at"] == started.isoformat()
    assert report["latest_completed_session"] == "2026-11-09"
    assert report["snapshot_dir"] == "data/snapshots/2026-11-06"
    assert snapshots._exit_code(report) == 0
    for source in original:
        destination = snapshots.snapshot_dir("2026-11-06") / source.name
        assert destination.read_bytes() == original[source]
        assert pd.to_datetime(pd.read_csv(destination).Date).max() == pd.Timestamp("2026-11-06")
    assert not snapshots.snapshot_dir("2026-11-09").exists()


def test_worker_entry_only_updates_canonical_sources_and_transports_its_report(tmp_path, monkeypatch):
    report_file = tmp_path / "report.json"
    calls = []
    report = {"completed_session": "2026-11-09", "ready_to_predict": False}
    monkeypatch.setattr(market, "update_market", lambda now: calls.append(now) or report)
    monkeypatch.setattr(snapshots, "build_snapshot", lambda *a, **k: pytest.fail("worker cannot materialize snapshots"))
    assert snapshots.main(["--worker", "--started-at", NOW.isoformat(), "--report-file", str(report_file)]) == 0
    assert calls == [NOW]
    assert json.loads(report_file.read_text()) == report
    assert list(tmp_path.iterdir()) == [report_file]


@pytest.mark.parametrize("value", ["0", "3601", "nan", "inf", "bad"])
def test_invalid_timeout_configuration_is_error_before_any_worker_starts(value, monkeypatch, capsys):
    monkeypatch.setenv(snapshots.TIMEOUT_ENV, value)
    monkeypatch.setattr(snapshots, "build_supervised_snapshot", lambda **kwargs: pytest.fail("invalid deadline launched worker"))
    assert snapshots.main([]) == 1
    assert json.loads(capsys.readouterr().out)["exit_code"] == 1


def test_timeout_cli_overrides_environment_and_default_is_thirty_minutes(monkeypatch, capsys):
    calls = []
    report = {"ready_to_predict": True, "snapshot_prices_ready": True, "snapshot_sentiment_ready": True}
    monkeypatch.setattr(snapshots, "build_supervised_snapshot", lambda **kwargs: calls.append(kwargs) or report.copy())
    monkeypatch.delenv(snapshots.TIMEOUT_ENV, raising=False)
    assert snapshots.main([]) == 0
    monkeypatch.setenv(snapshots.TIMEOUT_ENV, "23")
    assert snapshots.main([]) == 0
    assert snapshots.main(["--timeout-seconds", "17"]) == 0
    assert [call["timeout_seconds"] for call in calls] == [1800, 23, 17]
    assert all(json.loads(line)["exit_code"] == 0 for line in capsys.readouterr().out.splitlines())
