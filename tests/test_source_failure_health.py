"""Provider failures remain recoverable without changing saved source evidence."""
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
import pytest
import requests
from urllib3.connectionpool import HTTPSConnectionPool
from urllib3.exceptions import ReadTimeoutError
from urllib3.response import HTTPResponse

from experiments.v2 import v211_market as market
from scripts import news_update as news
from scripts import prices


NOW = datetime.fromisoformat("2026-11-09T17:37:00+08:00")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def source_ledger(tmp_path, monkeypatch):
    folder = tmp_path / "market"
    folder.mkdir()
    for field, name in {
        "PRICES": "prices.csv", "SENTIMENT": "sentiment.csv",
        "PRICE_SEED": "prices_seed.csv", "SENTIMENT_SEED": "sentiment_seed.csv",
        "CACHE": "news_cache.csv", "MANIFEST": "manifest.json", "STATE": "state.json",
        "AUDIT": "audit.jsonl", "SCAN_COVERAGE": "news_scan_coverage.json",
        "STAGING": "yahoo_staging.csv", "OFFICIAL_PRICES": "official_prices.csv",
        "OFFICIAL_SENTIMENT": "official_sentiment.csv", "LIVE": "daily",
    }.items():
        monkeypatch.setattr(market, field, folder / name)
    monkeypatch.setattr(market, "ROOT", tmp_path)
    monkeypatch.setattr(market, "assert_v211", lambda path: Path(path))
    calendar = lambda start, end: pd.bdate_range(start, end)
    monkeypatch.setattr(market, "sessions", calendar)
    monkeypatch.setattr(news, "sessions", calendar)
    monkeypatch.setattr(market, "current_completed_session", lambda now=None: pd.Timestamp("2026-11-09"))
    saved_prices = pd.DataFrame([
        {"Date": day, "ticker": ticker, "Close": 100.}
        for day in ("2026-11-06", "2026-11-09") for ticker in market.NAMES
    ])
    saved_prices.to_csv(market.PRICES, index=False)
    saved_prices[saved_prices.Date == "2026-11-06"].to_csv(market.PRICE_SEED, index=False)
    pd.DataFrame([
        {"Date": "2026-11-06", "ticker": ticker, "Daily_Sentiment": 0., "News_Count": 0,
         "Latest_News_Timestamp": "", "Source_Status": "verified_no_matching_article",
         "Scoring_Model": "neutral_no_news"} for ticker in market.TICKERS
    ]).to_csv(market.SENTIMENT, index=False)
    market.SENTIMENT_SEED.write_bytes(market.SENTIMENT.read_bytes())
    market.OFFICIAL_PRICES.write_bytes(market.PRICE_SEED.read_bytes())
    market.OFFICIAL_SENTIMENT.write_bytes(market.SENTIMENT_SEED.read_bytes())
    market.MANIFEST.write_text(json.dumps({
        "price_seed_sha256": _digest(market.PRICE_SEED),
        "sentiment_seed_sha256": _digest(market.SENTIMENT_SEED),
        "price_seed_through": "2026-11-06", "sentiment_seed_through": "2026-11-06",
        "official_price_sha256_at_seed": _digest(market.OFFICIAL_PRICES),
        "official_sentiment_sha256_at_seed": _digest(market.OFFICIAL_SENTIMENT),
    }))
    market.STATE.write_text('{"sessions": {}}')
    article = {"news_id": "saved-version", "publish_time": "2026-11-06T12:00:00+08:00",
               "title": "台積電", "summary": "已保存版本"}
    row = {**article, "ticker": "2330.TW", "Date": "2026-11-06", "Daily_Sentiment": .3,
           "Model_ID": news.MODEL_ID,
           **news.article_provenance(article, "2026-11-06T17:37:00+08:00"),
           "available_at": "2026-11-06T17:38:00+08:00"}
    row["cache_row_hash"] = news.cache_row_hash(row)
    news.append_provenance(market.CACHE, [{"record_type": "scored_cache_row", **row}])
    news.append_news_cache(market.CACHE, pd.DataFrame([row]))

    class Scorer:
        def score(self, text):
            return .6

    monkeypatch.setattr(news, "SentimentScorer", Scorer)
    return folder


class SourceSession:
    """Exercise requests response validation while keeping all traffic offline."""

    def __init__(self, failure=None, article=True):
        self.failure = failure
        self.article = article
        self.calls = []

    def get(self, url, *, params, timeout, allow_redirects, **kwargs):
        assert url == news.API
        assert timeout == (10, 40)
        assert allow_redirects is False
        day = datetime.fromtimestamp(params["startAt"], NOW.tzinfo).date().isoformat()
        self.calls.append((day, params["page"]))
        response = requests.Response()
        response.url = news.API
        response.status_code = 200
        data = []
        if day == "2026-11-09":
            if self.failure == "timeout":
                raise requests.ReadTimeout("injected provider timeout")
            if self.failure in (429, 503):
                response.status_code = self.failure
            if self.article:
                data = [{"newsId": 100, "publishAt": int(NOW.replace(hour=12).timestamp()),
                         "title": "台積電", "summary": "獲利成長", "revision": "observed-v1"}]
        payload = {"items": {"data": data}}
        if day == "2026-11-09" and self.failure == "invalid_json":
            response._content = b"not valid JSON"
        else:
            if day == "2026-11-09" and self.failure == "invalid_shape":
                payload = {"items": {"data": None}}
            response._content = json.dumps(payload).encode()
        return response


@pytest.mark.parametrize("failure", ["read_timeout", 429, 503])
@pytest.mark.parametrize("recover", [True, False])
def test_real_http_adapter_retries_transient_failures_and_stops_at_limit(monkeypatch, failure, recover):
    # Keep requests' actual adapter/pool/retry code; replace only the wire operation.
    session = requests.Session()
    session.trust_env = False
    monkeypatch.setattr(news.requests, "Session", lambda: session)
    calls = []
    waits = []

    def response(pool, connection, method, url, **kwargs):
        calls.append(method)
        failed = not recover or len(calls) == 1
        if failed and failure == "read_timeout":
            raise ReadTimeoutError(pool, url, "injected provider timeout")
        return HTTPResponse(
            body=BytesIO(b'{"items":{"data":[]}}'),
            status=int(failure) if failed else 200,
            headers={"Content-Type": "application/json"}, preload_content=False,
        )

    monkeypatch.setattr(HTTPSConnectionPool, "_make_request", response)
    monkeypatch.setattr("urllib3.util.retry.time.sleep", waits.append)
    try:
        if recover:
            assert news.fetch_day("2026-11-09", clock=lambda: NOW) == []
            assert calls == ["GET", "GET"]
        else:
            with pytest.raises((requests.ConnectionError, requests.exceptions.RetryError)):
                news.fetch_day("2026-11-09", clock=lambda: NOW)
            assert calls == ["GET"] * 5
            assert sum(waits) == 28
    finally:
        session.close()


@pytest.mark.parametrize("failure,error_type", [
    ("timeout", "ReadTimeout"), (429, "HTTPError"), (503, "HTTPError"),
    ("invalid_json", "JSONDecodeError"), ("invalid_shape", "ValueError"),
])
def test_failed_source_then_same_session_recovery_preserves_evidence(
        source_ledger, monkeypatch, failure, error_type):
    session = SourceSession(failure)
    fetch = news.fetch_day
    monkeypatch.setattr(news, "fetch_day", lambda day: fetch(day, session, clock=lambda: NOW))
    protected = [market.PRICES, market.SENTIMENT, market.PRICE_SEED, market.SENTIMENT_SEED,
                 market.MANIFEST, market.STATE, market.CACHE, news.provenance_path(market.CACHE)]
    original = {path: path.read_bytes() for path in protected}
    failed = market.update_market(NOW, clock=lambda: NOW)
    assert failed["prices_ready"] and not failed["sentiment_ready"]
    assert not failed["ready_to_predict"] and failed["gate"] == "sentiment_waiting"
    assert failed["sentiment_fetch_error"].startswith(error_type + ":")
    assert {path: path.read_bytes() for path in protected} == original
    assert not market.SCAN_COVERAGE.exists()
    assert "2026-11-09" not in set(pd.read_csv(market.SENTIMENT).Date)

    session.failure = None
    recovered = market.update_market(NOW, clock=lambda: NOW)
    assert recovered["ready_to_predict"] and recovered["error"] is None
    assert recovered["appended_sentiment_dates"] == ["2026-11-09"]
    for path in (market.PRICES, market.PRICE_SEED, market.SENTIMENT_SEED, market.MANIFEST):
        assert path.read_bytes() == original[path]
    for path in (market.SENTIMENT, market.CACHE, news.provenance_path(market.CACHE)):
        assert path.read_bytes().startswith(original[path])
    current = pd.read_csv(market.SENTIMENT).query("Date == '2026-11-09' and ticker == '2330.TW'").iloc[0]
    assert current.News_Count == 1 and current.Daily_Sentiment == .6
    scored = news.load_news_cache(market.CACHE)
    assert scored.iloc[0].cache_row_hash == news.cache_row_hash(scored.iloc[0])
    assert scored.iloc[1].source_revision == "observed-v1"
    coverage = json.loads(market.SCAN_COVERAGE.read_text())
    assert set(coverage["calendar_days"]) == {"2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09"}

    market.LIVE.mkdir()
    signal = market.LIVE / "2026-11-09.csv"
    signal.write_text("Signal_Date\n2026-11-09\n")
    stable = {path: path.read_bytes() for path in [*protected, market.SCAN_COVERAGE, signal]}
    monkeypatch.setattr(news, "fetch_day", lambda day: pytest.fail("completed session must be reused"))
    repeated = market.update_market(NOW, clock=lambda: NOW)
    assert repeated["ready_to_predict"]
    assert repeated["appended_sentiment_dates"] == []
    assert {path: path.read_bytes() for path in stable} == stable


def test_successful_empty_source_certifies_no_articles_only_after_complete_interval(source_ledger, monkeypatch):
    session = SourceSession(article=False)
    fetch = news.fetch_day
    monkeypatch.setattr(news, "fetch_day", lambda day: fetch(day, session, clock=lambda: NOW))
    monkeypatch.setattr(news, "SentimentScorer", lambda: pytest.fail("empty source needs no scorer"))
    cache_before = market.CACHE.read_bytes()
    result = market.update_market(NOW, clock=lambda: NOW)
    assert result["ready_to_predict"]
    assert session.calls == [(day, 1) for day in ("2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09")]
    current = pd.read_csv(market.SENTIMENT).query("Date == '2026-11-09'")
    assert len(current) == len(market.TICKERS)
    assert current.News_Count.eq(0).all() and current.Daily_Sentiment.eq(0).all()
    assert current.Source_Status.eq("verified_no_matching_article").all()
    assert market.CACHE.read_bytes() == cache_before
    coverage = json.loads(market.SCAN_COVERAGE.read_text())["signal_sessions"]["2026-11-09"]
    assert coverage["source_start"] == "2026-11-06"
    assert coverage["scoring_provenance_status"] == "not_applicable_no_articles"


def test_future_bars_cannot_enter_the_current_price_append_or_mature_labels():
    dates = pd.bdate_range("2026-11-05", periods=7)
    incoming = pd.DataFrame([
        {"Date": day, "ticker": ticker, "Close": 100. + i}
        for i, day in enumerate(dates) for ticker in market.NAMES
    ])
    saved = incoming[incoming.Date == dates[0]]
    selected, missing = market.select_new_price_rows(saved, incoming, dates[1], dates, market.NAMES)
    assert not missing
    assert set(selected.Date) == {dates[1]}
    from experiments.v2.v211 import label_maturity
    bounded_index = pd.DatetimeIndex(pd.concat([saved, selected]).Date.unique()).sort_values()
    assert not label_maturity(dates[0], bounded_index)["mature"]
    # The provider parser independently removes future rows before validating the current close.
    raw = pd.DataFrame({"Close": [100. + i for i in range(len(dates))]}, index=dates)
    raw.index.name = "Date"
    parsed, _ = prices._prepare_frame(raw, market.TICKERS[0], dates[1], dates[:2])
    assert set(parsed.Date) == set(dates[:2])
