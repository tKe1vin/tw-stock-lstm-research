"""Availability evidence describes observations, without inventing historical access."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.v2 import v211_market as market
from scripts import news_update as news
from scripts.common import TZ


def at(value):
    return datetime.fromisoformat(value)


def test_legacy_cache_stays_unverified_and_new_rows_keep_its_bytes(tmp_path):
    path = tmp_path / "news_cache.csv"
    legacy = {"news_id": "old", "ticker": "2330.TW", "publish_time": "2026-11-06T12:00:00+08:00",
              "Date": "2026-11-06", "Daily_Sentiment": .3, "Model_ID": news.MODEL_ID}
    pd.DataFrame([legacy]).to_csv(path, index=False, encoding="utf-8-sig")
    before = path.read_bytes()
    frame = news.load_news_cache(path)
    assert frame.iloc[0].provenance_status == "retrospective_unverified"
    assert pd.isna(frame.iloc[0].first_seen_at)
    assert pd.isna(frame.iloc[0].available_at)
    article = {"news_id": "new", "publish_time": "2026-11-09T12:00:00+08:00",
               "title": "台積電", "summary": "獲利成長"}
    row = {**legacy, "news_id": "new", "Date": "2026-11-09", "publish_time": article["publish_time"],
           **news.article_provenance(article, "2026-11-09T17:40:00+08:00"),
           "available_at": "2026-11-09T17:41:00+08:00"}
    row["cache_row_hash"] = news.cache_row_hash(row)
    news.append_provenance(path, [{"record_type": "scored_cache_row", **row}])
    news.append_news_cache(path, pd.DataFrame([row]))
    assert path.read_bytes().startswith(before)
    frame = news.load_news_cache(path)
    assert frame.iloc[0].provenance_status == "retrospective_unverified"
    assert pd.isna(frame.iloc[0].first_seen_at)
    assert frame.iloc[1].first_seen_at == "2026-11-09T17:40:00+08:00"
    assert frame.iloc[1].available_at == "2026-11-09T17:41:00+08:00"
    assert frame.iloc[1].provenance_status == "observed_at_recorded_time"
    # A different score cannot inherit evidence for an earlier scored cache row.
    changed = frame.iloc[1].to_dict()
    changed["Daily_Sentiment"] = -.9
    assert news.cache_row_hash(changed) != frame.iloc[1].cache_row_hash


def test_api_payload_revision_and_content_scope_are_recorded():
    payload = {"newsId": 42, "publishAt": int(at("2026-11-09T12:00:00+08:00").timestamp()),
               "updatedAt": "revision-2", "title": "台積電", "summary": "營收 &amp; 獲利"}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"items": {"data": [payload]}}

    class Session:
        def get(self, *args, **kwargs):
            return Response()

    article = news.fetch_day("2026-11-09", Session(),
                             clock=lambda: at("2026-11-09T17:37:00+08:00"))[0]
    evidence = news.article_provenance(article, "2026-11-09T17:38:00+08:00")
    assert evidence["first_seen_at"] == "2026-11-09T17:37:00+08:00"
    assert evidence["fetched_at"] == evidence["first_seen_at"]
    assert evidence["source_revision"] == "revision-2"
    assert evidence["source_revision_kind"] == "provider_revision"
    assert evidence["content_scope"] == "api_title_summary"
    assert len(evidence["raw_payload_sha256"]) == len(evidence["content_sha256"]) == 64
    changed = dict(article, summary="另一個版本")
    assert news.article_provenance(changed, evidence["fetched_at"])["content_sha256"] != evidence["content_sha256"]


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    folder = tmp_path / "market"
    folder.mkdir()
    for key, name in {"PRICES": "prices.csv", "SENTIMENT": "sentiment.csv", "PRICE_SEED": "prices_seed.csv",
                      "SENTIMENT_SEED": "sentiment_seed.csv", "CACHE": "news_cache.csv", "MANIFEST": "manifest.json",
                      "STATE": "state.json", "AUDIT": "audit.jsonl", "SCAN_COVERAGE": "news_scan_coverage.json",
                      "OFFICIAL_PRICES": "official_prices.csv", "OFFICIAL_SENTIMENT": "official_sentiment.csv",
                      "LIVE": "daily"}.items():
        monkeypatch.setattr(market, key, folder / name)
    monkeypatch.setattr(market, "assert_v211", lambda path: Path(path))
    prices = pd.DataFrame([{"Date": date, "ticker": ticker, "Close": 100.}
                           for date in ("2026-11-06", "2026-11-09") for ticker in market.NAMES])
    sentiment = pd.DataFrame([{"Date": "2026-11-06", "ticker": ticker, "Daily_Sentiment": 0.,
                              "News_Count": 0, "Latest_News_Timestamp": "", "Source_Status": "verified_no_matching_article",
                              "Scoring_Model": "neutral_no_news"} for ticker in market.TICKERS])
    prices.to_csv(market.PRICES, index=False)
    prices[prices.Date == "2026-11-06"].to_csv(market.PRICE_SEED, index=False)
    sentiment.to_csv(market.SENTIMENT, index=False)
    sentiment.to_csv(market.SENTIMENT_SEED, index=False)
    market.OFFICIAL_PRICES.write_bytes(market.PRICE_SEED.read_bytes())
    market.OFFICIAL_SENTIMENT.write_bytes(market.SENTIMENT_SEED.read_bytes())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    market.MANIFEST.write_text(json.dumps({
        "price_seed_sha256": digest(market.PRICE_SEED), "sentiment_seed_sha256": digest(market.SENTIMENT_SEED),
        "price_seed_through": "2026-11-06", "sentiment_seed_through": "2026-11-06",
        "official_price_sha256_at_seed": digest(market.OFFICIAL_PRICES),
        "official_sentiment_sha256_at_seed": digest(market.OFFICIAL_SENTIMENT)}))
    market.STATE.write_text('{"sessions": {}}')
    monkeypatch.setattr(market, "sessions", lambda start, end: pd.bdate_range(start, end))
    monkeypatch.setattr(news, "sessions", lambda start, end: pd.bdate_range(start, end))
    monkeypatch.setattr(market, "current_completed_session", lambda now=None: pd.Timestamp("2026-11-09"))

    class Scorer:
        def score(self, text):
            return .6

    monkeypatch.setattr(news, "SentimentScorer", Scorer)
    return folder


def test_after_close_acquisition_is_valid_but_not_available_before_completion(ledger, monkeypatch):
    before = market.SENTIMENT.read_bytes()

    def fetch(day):
        return [{"news_id": "current", "publish_time": "2026-11-09T13:20:00+08:00",
                 "title": "台積電", "summary": "獲利提升"}] if str(day.date()) == "2026-11-09" else []

    monkeypatch.setattr(news, "fetch_day", fetch)
    report = market.update_market(at("2026-11-09T17:37:00+08:00"),
                                  clock=lambda: at("2026-11-09T18:00:00+08:00"))
    assert report["ready_to_predict"]
    assert report["run_started_at"] == "2026-11-09T17:37:00+08:00"
    assert report["fetched_at"] == "2026-11-09T18:00:00+08:00"
    assert report["fetched_at_kind"] == "observed_completion"
    assert report["news_provenance_status"] == "observed_at_recorded_time"
    assert market.SENTIMENT.read_bytes().startswith(before)
    cache = news.load_news_cache(market.CACHE)
    assert set(cache.first_seen_at) == {"2026-11-09T18:00:00+08:00"}
    assert set(cache.available_at) == {"2026-11-09T18:00:00+08:00"}
    early = market.inspect_market(at("2026-11-09T17:59:00+08:00"))
    assert not early["ready_to_predict"]
    assert early["gate"] == "news_not_yet_available_at_decision"
    completed = market.inspect_market(at("2026-11-09T18:01:00+08:00"))
    assert completed["ready_to_predict"]
    assert completed["price_sha256"] and completed["sentiment_sha256"] and completed["fetched_at"]


def test_retrospective_fetch_does_not_claim_historical_availability(ledger, monkeypatch):
    monkeypatch.setattr(news, "fetch_day", lambda day: [])
    market.extend_sentiment(pd.Timestamp("2026-11-09"), at("2026-11-10T17:37:00+08:00"), {},
                            clock=lambda: at("2026-11-10T18:00:00+08:00"))
    coverage = json.loads(market.SCAN_COVERAGE.read_text())
    evidence = coverage["signal_sessions"]["2026-11-09"]
    assert evidence["available_at"] == "2026-11-10T18:00:00+08:00"
    assert evidence["provenance_status"] == "retrospective_unverified"


def test_skip_download_uses_source_hashes_and_rechecks_decision_clock(ledger, monkeypatch):
    from experiments.v2 import v211, v211_daily as daily

    monkeypatch.setattr(news, "fetch_day", lambda day: [])
    market.update_market(at("2026-11-09T17:37:00+08:00"),
                         clock=lambda: at("2026-11-09T18:00:00+08:00"))
    monkeypatch.setenv("TW_LSTM_SKIP_MARKET_UPDATE", "1")
    monkeypatch.setattr(daily, "seal_signals", lambda: None)
    monkeypatch.setattr(daily, "snapshot", lambda: {})
    monkeypatch.setattr(daily, "assert_unchanged", lambda before: None)
    monkeypatch.setattr(daily, "update_recorded_results", lambda now: {})
    weights = ledger / "weights"
    weights.mkdir()
    (weights / "weights.npz").write_bytes(b"test-weights")
    digest = hashlib.sha256(b"test-weights").hexdigest()
    (weights / "model_card.json").write_text(json.dumps({"weight_sha256": digest}))
    monkeypatch.setattr(daily, "WEIGHT_DIR", weights)
    captured = {}

    def run(now, fetch=None, publication_clock=None):
        captured.update({"now": now, "fetch": fetch, "publication_clock": publication_clock})
        return {"signal_published": True, "signal": {"completed_session": "2026-11-09"}}

    monkeypatch.setattr(v211, "run", run)
    monkeypatch.setattr(daily, "update_market", lambda *a, **kw: pytest.fail("skip downloaded data"))
    assert daily.main(clock=lambda: at("2026-11-09T18:02:00+08:00")) == 0
    assert captured["fetch"]["price_sha256"] and captured["fetch"]["sentiment_sha256"]
    assert captured["fetch"]["fetched_at"] == "2026-11-09T18:00:00+08:00"
    assert captured["fetch"]["skipped_download"]
    assert captured["publication_clock"]() == captured["now"]
