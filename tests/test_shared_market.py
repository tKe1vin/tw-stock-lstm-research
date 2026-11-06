"""The four model paths share complete, append-only market inputs."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.v2 import v211_market as market
from scripts import news_update, update_market_data as snapshots
from scripts.common import TZ


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sentiment(day):
    return pd.DataFrame([{
        "Date": day, "ticker": ticker, "Daily_Sentiment": 0.0,
        "News_Count": 0, "Latest_News_Timestamp": "",
        "Source_Status": "verified_no_matching_article", "Scoring_Model": "neutral_no_news",
    } for ticker in market.TICKERS])


def _prices(day):
    return pd.DataFrame([{"Date": day, "ticker": name, "Close": 100.0} for name in market.NAMES])


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    folder = tmp_path / "research" / "v211" / "market"
    folder.mkdir(parents=True)
    paths = {
        "MARKET_DIR": folder, "PRICES": folder / "prices.csv", "SENTIMENT": folder / "sentiment.csv",
        "PRICE_SEED": folder / "prices_seed.csv", "SENTIMENT_SEED": folder / "sentiment_seed.csv",
        "CACHE": folder / "news_cache.csv", "MANIFEST": folder / "manifest.json",
        "STATE": folder / "state.json", "AUDIT": folder / "fetch_audit.jsonl",
        "SCAN_COVERAGE": folder / "news_scan_coverage.json", "STAGING": folder / "staging.csv",
        "OFFICIAL_PRICES": tmp_path / "official_prices.csv",
        "OFFICIAL_SENTIMENT": tmp_path / "official_sentiment.csv", "LIVE": folder.parent / "daily",
    }
    for name, path in paths.items():
        monkeypatch.setattr(market, name, path)
    monkeypatch.setattr(market, "ROOT", tmp_path)
    monkeypatch.setattr(snapshots, "ROOT", tmp_path)

    def guarded(path):
        path = Path(path)
        assert folder == path or folder in path.parents
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(market, "assert_v211", guarded)
    _prices("2026-11-06").to_csv(paths["PRICE_SEED"], index=False)
    _sentiment("2026-11-06").to_csv(paths["SENTIMENT_SEED"], index=False)
    for source, destination in (
        ("PRICE_SEED", "PRICES"), ("PRICE_SEED", "OFFICIAL_PRICES"),
        ("SENTIMENT_SEED", "SENTIMENT"), ("SENTIMENT_SEED", "OFFICIAL_SENTIMENT"),
    ):
        paths[destination].write_bytes(paths[source].read_bytes())
    paths["MANIFEST"].write_text(json.dumps({
        "price_seed_sha256": _digest(paths["PRICE_SEED"]),
        "sentiment_seed_sha256": _digest(paths["SENTIMENT_SEED"]),
        "price_seed_through": "2026-11-06", "sentiment_seed_through": "2026-11-06",
        "official_price_sha256_at_seed": _digest(paths["OFFICIAL_PRICES"]),
        "official_sentiment_sha256_at_seed": _digest(paths["OFFICIAL_SENTIMENT"]),
    }))
    paths["STATE"].write_text('{"sessions": {}}')
    return paths


def _calendar(monkeypatch, completed, holidays=()):
    def sessions(start, end):
        days = pd.bdate_range(start, end).normalize()
        return days.difference(pd.DatetimeIndex(holidays))

    monkeypatch.setattr(market, "sessions", sessions)
    monkeypatch.setattr(news_update, "sessions", sessions)
    monkeypatch.setattr(market, "current_completed_session", lambda now=None: pd.Timestamp(completed))
    monkeypatch.setattr(snapshots, "current_completed_session", lambda now=None: pd.Timestamp(completed))


@pytest.mark.parametrize("completed,holidays", [
    ("2026-11-09", []),
    ("2026-11-10", ["2026-11-09"]),
])
def test_news_covers_previous_close_weekend_and_holiday(ledger, monkeypatch, completed, holidays):
    _calendar(monkeypatch, completed, holidays)
    fetched = []
    before = market.SENTIMENT.read_bytes()

    def fetch(day):
        stamp = str(day.date())
        fetched.append(stamp)
        if stamp in {"2026-11-06", "2026-11-07"}:
            return [{"news_id": stamp, "publish_time": f"{stamp}T14:00:00+08:00",
                     "title": "台積電", "summary": "獲利提升"}]
        return []

    class Scorer:
        def score(self, text):
            return 0.6

    monkeypatch.setattr(news_update, "fetch_day", fetch)
    monkeypatch.setattr(news_update, "SentimentScorer", Scorer)
    report = {}
    market.extend_sentiment(pd.Timestamp(completed), datetime.fromisoformat(f"{completed}T17:37:00+08:00"), report)
    assert fetched == [str(day.date()) for day in pd.date_range("2026-11-06", completed)]
    assert market.SENTIMENT.read_bytes().startswith(before)
    result = pd.read_csv(market.SENTIMENT)
    part = result[(result.Date == completed) & (result.ticker == "2330.TW")].iloc[0]
    assert part.News_Count == 2
    assert part.Daily_Sentiment == pytest.approx(0.6)
    assert market.sentiment_day_ready(result, completed)
    coverage = json.loads(market.SCAN_COVERAGE.read_text())
    assert set(coverage["calendar_days"]) == set(fetched)
    assert coverage["signal_sessions"][completed]["source_start"] == "2026-11-06"
    assert report["sentiment_ready"] is True


def test_failed_news_fetch_preserves_ledger_and_no_news_is_not_certified(ledger, monkeypatch):
    _calendar(monkeypatch, "2026-11-09")
    before = market.SENTIMENT.read_bytes()

    def fetch(day):
        if str(day.date()) == "2026-11-07":
            raise RuntimeError("upstream unavailable")
        return []

    monkeypatch.setattr(news_update, "fetch_day", fetch)
    with pytest.raises(RuntimeError, match="upstream unavailable"):
        market.extend_sentiment(pd.Timestamp("2026-11-09"), datetime(2026, 11, 9, 17, 37, tzinfo=TZ), {})
    assert market.SENTIMENT.read_bytes() == before
    assert not market.CACHE.exists()
    assert not market.SCAN_COVERAGE.exists()


def test_shared_snapshot_updates_once_and_keeps_official_fixtures(ledger, monkeypatch):
    _calendar(monkeypatch, "2026-11-06")
    official = {name: ledger[name].read_bytes() for name in ("OFFICIAL_PRICES", "OFFICIAL_SENTIMENT")}
    calls = []

    def update(now):
        calls.append(now)
        return market.inspect_market(now)

    monkeypatch.setattr(market, "update_market", update)
    report = snapshots.build_snapshot(datetime(2026, 11, 6, 17, 37, tzinfo=TZ))
    folder = snapshots.snapshot_dir("2026-11-06")
    assert len(calls) == 1
    assert report["prices_ready"] and report["sentiment_ready"]
    assert report["snapshot_prices_ready"] and report["snapshot_sentiment_ready"]
    assert (folder / "prices.csv").read_bytes() == market.PRICES.read_bytes()
    assert (folder / "sentiment.csv").read_bytes() == market.SENTIMENT.read_bytes()
    assert (folder / "manifest.json").exists()
    assert all(ledger[name].read_bytes() == body for name, body in official.items())


def test_read_only_snapshot_does_not_create_any_files(ledger, monkeypatch):
    _calendar(monkeypatch, "2026-11-06")
    before = {path.relative_to(ledger["MARKET_DIR"].parents[2]): path.read_bytes()
              for path in ledger["MARKET_DIR"].parents[2].rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only inspection attempted an update")

    monkeypatch.setattr(market, "update_market", forbidden)
    report = snapshots.build_snapshot(datetime(2026, 11, 6, 17, 37, tzinfo=TZ), allow_download=False)
    after = {path.relative_to(ledger["MARKET_DIR"].parents[2]): path.read_bytes()
             for path in ledger["MARKET_DIR"].parents[2].rglob("*") if path.is_file()}
    assert before == after
    assert not snapshots.snapshot_dir("2026-11-06").exists()
    assert report["prices_ready"] and report["sentiment_ready"]
    assert not report["snapshot_prices_ready"] and not report["snapshot_sentiment_ready"]


def test_price_success_is_available_when_news_fetch_fails(ledger, monkeypatch):
    _calendar(monkeypatch, "2026-11-09")
    pd.concat([_prices("2026-11-06"), _prices("2026-11-09")]).to_csv(market.PRICES, index=False)

    def fetch(day):
        raise RuntimeError("news unavailable")

    monkeypatch.setattr(news_update, "fetch_day", fetch)
    report = snapshots.build_snapshot(datetime(2026, 11, 9, 17, 37, tzinfo=TZ))
    folder = snapshots.snapshot_dir("2026-11-09")
    assert report["prices_ready"] is True
    assert report["sentiment_ready"] is False
    assert report["snapshot_prices_ready"] is True
    assert not (folder / "sentiment.csv").exists()
    assert report["ready_to_predict"] is False
    assert "news unavailable" in report["sentiment_fetch_error"]


def test_existing_verified_current_day_survives_a_source_update_error(ledger, monkeypatch):
    _calendar(monkeypatch, "2026-11-06")

    def failed_update(*args):
        raise RuntimeError("source temporarily unavailable")

    monkeypatch.setattr(market, "extend_sentiment", failed_update)
    report = market.update_market(datetime(2026, 11, 6, 17, 37, tzinfo=TZ))
    assert report["prices_ready"] and report["sentiment_ready"]
    assert report["ready_to_predict"] is True
    assert report["used_existing_current_data"] is True
    assert "source temporarily unavailable" in report["sentiment_fetch_error"]


def test_new_news_never_rewrites_locked_signal_day(ledger, monkeypatch):
    _calendar(monkeypatch, "2026-11-10")
    pd.concat([_sentiment("2026-11-06"), _sentiment("2026-11-09")]).to_csv(market.SENTIMENT, index=False)
    market.LIVE.mkdir(parents=True)
    (market.LIVE / "2026-11-09.csv").write_text("Signal_Date\n2026-11-09\n")
    before = market.SENTIMENT.read_bytes()

    def fetch(day):
        return [{"news_id": "late-monday", "publish_time": "2026-11-09T14:00:00+08:00",
                 "title": "台積電", "summary": "展望改善"}] if str(day.date()) == "2026-11-09" else []

    class Scorer:
        def score(self, text):
            return 0.5

    monkeypatch.setattr(news_update, "fetch_day", fetch)
    monkeypatch.setattr(news_update, "SentimentScorer", Scorer)
    market.extend_sentiment(pd.Timestamp("2026-11-10"), datetime(2026, 11, 10, 17, 37, tzinfo=TZ), {})
    assert market.SENTIMENT.read_bytes().startswith(before)
    result = pd.read_csv(market.SENTIMENT)
    assert market.sentiment_day_ready(result, "2026-11-10")
    assert len(result[result.Date == "2026-11-09"]) == 5
    assert result.loc[(result.Date == "2026-11-09") & (result.ticker == "2330.TW"), "Daily_Sentiment"].iloc[0] == 0


def test_snapshot_gates_reject_duplicates_and_missing_provenance():
    prices = _prices("2026-11-06")
    assert not market.price_day_ready(pd.concat([prices, prices.iloc[:1]]), "2026-11-06")
    prices.loc[0, "Close"] = float("inf")
    assert not market.price_day_ready(prices, "2026-11-06")
    sentiment = _sentiment("2026-11-06")
    assert not market.sentiment_day_ready(sentiment.drop(columns=["News_Count"]), "2026-11-06")
    sentiment.loc[0, "Daily_Sentiment"] = 0.5
    assert not market.sentiment_day_ready(sentiment, "2026-11-06")
