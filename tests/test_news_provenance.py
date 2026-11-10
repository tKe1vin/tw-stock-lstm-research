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


def scored_record(schema=2):
    article = {"news_id": "verified", "publish_time": "2026-11-09T12:00:00+08:00",
               "title": "台積電", "summary": "營收成長"}
    row = {"schema_version": schema, "source": news.API, "record_type": "scored_cache_row",
           "news_id": "verified", "ticker": "2330.TW", "publish_time": article["publish_time"],
           "Date": "2026-11-09", "Daily_Sentiment": .25, "Model_ID": news.MODEL_ID,
           **news.article_provenance(article, "2026-11-09T17:37:00+08:00"),
           "available_at": "2026-11-09T17:38:00+08:00"}
    if schema == 2:
        row["scoring_provenance_status"] = "checkpoint_unverified_custom_scorer"
    row["cache_row_hash"] = news.cache_row_hash(row)
    return row


def test_schema_one_cache_preserves_weaker_evidence_without_rewriting(tmp_path):
    path = tmp_path / "news.csv"
    row = scored_record(schema=1)
    pd.DataFrame([{name: row[name] for name in news.CACHE_COLUMNS}]).to_csv(path, index=False)
    sidecar = news.provenance_path(path)
    sidecar.write_text(json.dumps(row) + "\n")
    original = (path.read_bytes(), sidecar.read_bytes())
    frame = news.load_news_cache(path)
    assert frame.iloc[0].provenance_status == "observed_at_recorded_time"
    assert frame.iloc[0].scoring_provenance_status == "checkpoint_unverified_legacy"
    assert pd.isna(frame.iloc[0].model_revision)
    assert original == (path.read_bytes(), sidecar.read_bytes())


@pytest.mark.parametrize("field,value", [
    ("Daily_Sentiment", float("nan")), ("Daily_Sentiment", float("inf")),
    ("Daily_Sentiment", 1.01), ("content_sha256", "bad"),
    ("raw_payload_sha256", "bad"), ("cache_row_hash", "0" * 64),
    ("available_at", "2026-11-09T17:36:00+08:00"),
    ("available_at", "2026-11-09T17:38:00"),
    ("first_seen_at", "2026-11-09T11:59:00+08:00"),
    ("publish_time", None), ("Date", "2026-11-10"),
    ("source_revision", "sha256:" + "0" * 64),
    ("Model_ID", "unrecorded-model"), ("ticker", "0050.TW"),
    ("scoring_provenance_status", "verified"),
])
def test_malformed_scored_evidence_fails_closed(field, value):
    row = scored_record()
    row[field] = value
    if field in news.CACHE_COLUMNS:
        row["cache_row_hash"] = news.cache_row_hash(row)
    with pytest.raises(ValueError):
        news.validate_scored_provenance(row)


def test_invalid_batch_cannot_partially_append_evidence(tmp_path):
    cache = tmp_path / "news.csv"
    good = scored_record()
    bad = {**good, "news_id": "other", "available_at": "2026-11-09T17:38:00"}
    bad["cache_row_hash"] = news.cache_row_hash(bad)
    with pytest.raises(ValueError):
        news.append_provenance(cache, [good, bad])
    assert not news.provenance_path(cache).exists()


def test_checkpoint_claim_requires_matching_commit_config_and_runtime():
    commit = "a" * 40
    evidence = {"model_revision": commit, "tokenizer_revision": commit,
                "scoring_config": dict(news.SCORING_CONFIG),
                "runtime_versions": {"torch": "2.8.0", "transformers": "4.57.0"}}

    class Scorer:
        def provenance(self):
            return evidence

    assert news.scorer_provenance(Scorer())["scoring_provenance_status"] == "resolved_checkpoint_recorded"
    row = {**scored_record(), **evidence, "scoring_provenance_status": "resolved_checkpoint_recorded"}
    news.validate_scored_provenance(row)
    for changed in ({"tokenizer_revision": "b" * 40}, {"model_revision": "main"},
                    {"scoring_config": {"formula": "different"}}, {"runtime_versions": {"torch": "2.8.0"}}):
        with pytest.raises(ValueError, match="checkpoint"):
            news.validate_scored_provenance({**row, **changed})
    assert news.scorer_provenance(object())["scoring_provenance_status"] == "checkpoint_unverified_custom_scorer"


def test_sentiment_constructor_pins_model_and_tokenizer_to_same_resolved_commit(monkeypatch):
    import sys
    from types import SimpleNamespace

    commit = "c" * 40
    calls = []
    config = SimpleNamespace(_commit_hash=commit, id2label={0: "negative", 1: "positive"})

    class ConfigLoader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("config", kwargs))
            return config

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("tokenizer", kwargs))
            return object()

    class Model:
        def __init__(self):
            self.config = config

        def eval(self):
            return self

        def cpu(self):
            return self

    class ModelLoader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("model", kwargs))
            return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.8.0"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        __version__="4.57.0", AutoConfig=ConfigLoader, AutoTokenizer=TokenizerLoader,
        AutoModelForSequenceClassification=ModelLoader))
    scorer = news.SentimentScorer()
    assert calls[0][1]["revision"] == "main"
    assert calls[1][1]["revision"] == calls[2][1]["revision"] == commit
    assert all(call[1]["trust_remote_code"] is False for call in calls)
    assert news.scorer_provenance(scorer)["scoring_provenance_status"] == "resolved_checkpoint_recorded"
    copied = scorer.provenance()
    copied["scoring_config"]["max_tokens"] = 999
    assert scorer.provenance()["scoring_config"]["max_tokens"] == 256
    with pytest.raises(ValueError, match="checkpoint"):
        news.SentimentScorer(revision="d" * 40)


@pytest.mark.parametrize("status,url", [(302, news.API), (200, "https://example.com/news")])
def test_news_fetch_rejects_unreviewed_redirect_or_origin(status, url):
    class Session:
        def get(self, *args, **kwargs):
            assert kwargs["allow_redirects"] is False
            return type("Response", (), {"status_code": status, "url": url})()

    with pytest.raises(ValueError):
        news.fetch_day("2026-11-09", Session())


def test_shared_market_scoring_path_retains_exact_checkpoint(ledger, monkeypatch):
    commit = "e" * 40

    class Scorer:
        def score(self, text):
            return .6

        def provenance(self):
            return {"model_revision": commit, "tokenizer_revision": commit,
                    "scoring_config": dict(news.SCORING_CONFIG),
                    "runtime_versions": {"torch": "2.8.0", "transformers": "4.57.0"}}

    def fetch(day):
        return [{"news_id": "checkpoint", "publish_time": "2026-11-09T12:00:00+08:00",
                 "title": "台積電", "summary": "獲利成長"}] if str(day.date()) == "2026-11-09" else []

    monkeypatch.setattr(news, "SentimentScorer", Scorer)
    monkeypatch.setattr(news, "fetch_day", fetch)
    report = market.update_market(at("2026-11-09T17:37:00+08:00"),
                                  clock=lambda: at("2026-11-09T18:00:00+08:00"))
    assert report["ready_to_predict"]
    assert report["news_scoring_provenance_status"] == "resolved_checkpoint_recorded"
    frame = news.load_news_cache(market.CACHE)
    assert set(frame.model_revision) == {commit}
    assert set(frame.tokenizer_revision) == {commit}
    assert set(frame.scoring_provenance_status) == {"resolved_checkpoint_recorded"}
    assert frame.iloc[0].scoring_config == news.SCORING_CONFIG
    assert market.inspect_market(at("2026-11-09T18:01:00+08:00"))[
        "news_scoring_provenance_status"] == "resolved_checkpoint_recorded"


def test_subsecond_publication_after_cutoff_moves_to_next_session():
    assert news.to_signal_date("2026-11-09T13:30:00+08:00") == pd.Timestamp("2026-11-09")
    assert news.to_signal_date("2026-11-09T13:30:00.000001+08:00") == pd.Timestamp("2026-11-10")
