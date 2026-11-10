"""Append-only market snapshot for the V2.11 tracker.

The official price file, official sentiment file, and experimental sentiment
copy stay untouched. New sessions are fetched with the same public sources the
daily notebooks already use, then stored under the V2.11 result directory.
A missing sentiment day is left missing. It is never filled from the day before.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.horizon_experiment import SENTIMENT_PATH, executable_labels
from experiments.risk_experiment import MARKET, RUN, TICKERS, build_frames, load_panel, load_sentiment
from experiments.v2.guard import ROOT, sha256
from experiments.v2.v211 import LIVE, V211, assert_v211
from scripts.common import TZ, current_completed_session, sessions

MARKET_DIR = V211 / "market"
PRICES = MARKET_DIR / "prices.csv"
SENTIMENT = MARKET_DIR / "sentiment.csv"
PRICE_SEED = MARKET_DIR / "prices_seed.csv"
SENTIMENT_SEED = MARKET_DIR / "sentiment_seed.csv"
CACHE = MARKET_DIR / "news_cache.csv"
MANIFEST = MARKET_DIR / "manifest.json"
STATE = MARKET_DIR / "state.json"
AUDIT = MARKET_DIR / "fetch_audit.jsonl"
SCAN_COVERAGE = MARKET_DIR / "news_scan_coverage.json"
STAGING = MARKET_DIR / "yahoo_staging.csv"
PRICE_SOURCE = "yfinance unadjusted Close"
SENTIMENT_SOURCE = "cnyes tw_stock + IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
CALENDAR = "exchange_calendars XTAI, excluding scripts.common.SPECIAL_MARKET_CLOSURES"
ALLOWED_SENTIMENT = {"observed_article", "verified_no_matching_article"}
PRICE_ORIGIN = RUN / "historical_prices.csv"
OFFICIAL_PRICES = ROOT / "data" / "prices" / "historical_prices.csv"
OFFICIAL_SENTIMENT = ROOT / "data" / "sentiment_features_2026.csv"
NAMES = TICKERS + [MARKET]


def market_files_ready():
    return PRICES.exists() and SENTIMENT.exists() and MANIFEST.exists() and PRICE_SEED.exists() and SENTIMENT_SEED.exists()


def _write_json(path, payload):
    path = assert_v211(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _append_audit(record):
    path = assert_v211(AUDIT)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def same_prefix(base, current, through, value):
    """True when every saved row on or before `through` still has the same value."""
    left = base.copy()
    right = current.copy()
    left["Date"] = pd.to_datetime(left["Date"]).dt.normalize()
    right["Date"] = pd.to_datetime(right["Date"]).dt.normalize()
    through = pd.Timestamp(through).normalize()
    left = left[left.Date <= through].sort_values(["Date", "ticker"])
    right = right[right.Date <= through].sort_values(["Date", "ticker"])
    if len(left) != len(right):
        return False
    if not left.ticker.astype(str).reset_index(drop=True).equals(right.ticker.astype(str).reset_index(drop=True)):
        return False
    if not left.Date.reset_index(drop=True).equals(right.Date.reset_index(drop=True)):
        return False
    a = pd.to_numeric(left[value], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(right[value], errors="coerce").to_numpy(dtype=float)
    return bool(np.allclose(a, b, rtol=0, atol=1e-8, equal_nan=True))


def sentiment_day_ready(frame, day, tickers=None):
    tickers = list(tickers or TICKERS)
    required = {"Date", "ticker", "Daily_Sentiment", "News_Count", "Latest_News_Timestamp", "Source_Status", "Scoring_Model"}
    if frame is None or len(frame) == 0 or not required.issubset(frame.columns):
        return False
    work = frame.copy()
    work["Date"] = pd.to_datetime(work["Date"]).dt.normalize()
    part = work[work.Date == pd.Timestamp(day).normalize()]
    if len(part) != len(tickers) or set(part.ticker.astype(str)) != set(tickers):
        return False
    values = pd.to_numeric(part.Daily_Sentiment, errors="coerce")
    if values.isna().any() or not bool(values.between(-1, 1).all()):
        return False
    if not set(part.Source_Status.astype(str)).issubset(ALLOWED_SENTIMENT):
        return False
    counts = pd.to_numeric(part.News_Count, errors="coerce")
    if counts.isna().any() or not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        return False
    cutoff = pd.Timestamp(day).normalize().tz_localize(TZ) + pd.Timedelta(hours=13, minutes=30)
    for _, row in part.iterrows():
        if row.Source_Status == "verified_no_matching_article":
            if float(row.News_Count) != 0 or float(row.Daily_Sentiment) != 0 or row.Scoring_Model != "neutral_no_news":
                return False
        else:
            if float(row.News_Count) < 1 or row.Scoring_Model != "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment":
                return False
            try:
                published = pd.Timestamp(row.Latest_News_Timestamp)
                if pd.isna(published) or published.tzinfo is None or published.tz_convert(TZ) > cutoff:
                    return False
            except (ValueError, TypeError):
                return False
    return True


def price_day_ready(frame, day, names=None):
    """Require one finite positive close for every stock and the market index."""
    names = list(names or NAMES)
    if frame is None or not {"Date", "ticker", "Close"}.issubset(frame.columns):
        return False
    work = frame.copy()
    work["Date"] = pd.to_datetime(work["Date"]).dt.normalize()
    part = work[work.Date == pd.Timestamp(day).normalize()]
    if len(part) != len(names) or set(part.ticker.astype(str)) != set(names):
        return False
    closes = pd.to_numeric(part.Close, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(closes).all() and (closes > 0).all())


def prediction_gate(prices, sentiment, completed):
    """Publish only when the completed session has its own prices and sentiment."""
    completed = pd.Timestamp(completed).normalize()
    if not price_day_ready(prices, completed):
        return False, "prices_missing"
    if not sentiment_day_ready(sentiment, completed):
        return False, "sentiment_waiting"
    return True, "ready"


def select_new_price_rows(ledger, incoming, as_of, required_sessions, names):
    """Rows strictly after the saved ledger. Existing dates are never returned."""
    saved = ledger.copy()
    fresh = incoming.copy()
    saved["Date"] = pd.to_datetime(saved["Date"]).dt.normalize()
    fresh["Date"] = pd.to_datetime(fresh["Date"]).dt.normalize()
    as_of = pd.Timestamp(as_of).normalize()
    required = pd.DatetimeIndex(required_sessions).normalize()
    required = required[(required > saved.Date.max()) & (required <= as_of)]
    if len(required) == 0:
        return fresh.iloc[0:0].copy(), []
    panel = fresh.pivot(index="Date", columns="ticker", values="Close")
    missing = []
    for day in required:
        if day not in panel.index or any(name not in panel.columns for name in names):
            missing.append(str(pd.Timestamp(day).date()))
            continue
        values = pd.to_numeric(panel.loc[day, list(names)], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            missing.append(str(pd.Timestamp(day).date()))
    if missing:
        return None, missing
    new = fresh[fresh.Date.isin(required) & fresh.ticker.astype(str).isin(names)].copy()
    saved_keys = set(zip(saved.Date, saved.ticker.astype(str)))
    overlap = [day for day, ticker in zip(new.Date, new.ticker.astype(str)) if (day, ticker) in saved_keys]
    if overlap:
        raise RuntimeError("新價格碰到已保存的歷史列")
    return new.sort_values(["Date", "ticker"]).reset_index(drop=True), []


def price_basis_ok(ledger, incoming):
    saved = ledger.copy()
    fresh = incoming.copy()
    saved["Date"] = pd.to_datetime(saved["Date"]).dt.normalize()
    fresh["Date"] = pd.to_datetime(fresh["Date"]).dt.normalize()
    left = saved.pivot(index="Date", columns="ticker", values="Close")
    right = fresh.pivot(index="Date", columns="ticker", values="Close")
    common = left.index.intersection(right.index)
    if len(common) == 0 or left.index.max() not in common:
        return False, {}
    day = left.index.max()
    diffs = {}
    for name in NAMES:
        base = float(left.loc[day, name])
        other = float(right.loc[day, name])
        diffs[name] = abs(base - other) / abs(base)
        if diffs[name] > 0.002:
            return False, diffs
    return True, diffs


def rows_for_sessions(cache, session_days):
    """Score each requested session from that session's articles only."""
    old = cache if cache is not None else pd.DataFrame()
    rows = []
    for day in session_days:
        stamp = str(pd.Timestamp(day).date())
        for ticker in TICKERS:
            if len(old) == 0:
                part = old
            else:
                part = old.loc[(old.Date.astype(str) == stamp) & (old.ticker.astype(str) == ticker)]
            latest = part["publish_time"].max() if len(part) else None
            rows.append({
                "Date": pd.Timestamp(day).normalize(),
                "ticker": ticker,
                "Daily_Sentiment": float(part.Daily_Sentiment.astype(float).mean()) if len(part) else 0.0,
                "News_Count": int(len(part)),
                "Latest_News_Timestamp": latest,
                "Source_Status": "observed_article" if len(part) else "verified_no_matching_article",
                "Scoring_Model": part.Model_ID.iloc[0] if len(part) else "neutral_no_news",
            })
    built = pd.DataFrame(rows)
    if not built.Daily_Sentiment.between(-1, 1).all():
        raise RuntimeError("情緒分數超出 -1 到 1")
    return built


def locked_signal_dates():
    found = set()
    if not LIVE.exists():
        return found
    for path in LIVE.glob("*.csv"):
        frame = pd.read_csv(path, usecols=["Signal_Date"])
        found.add(pd.Timestamp(frame.Signal_Date.iloc[0]).normalize())
    return found


def _copy_once(source, destination):
    destination = assert_v211(destination)
    if destination.exists():
        raise RuntimeError(f"快照已存在，拒絕重抄：{destination.name}")
    shutil.copyfile(source, destination)
    return sha256(destination)


def _clear_incomplete_seed():
    if MANIFEST.exists():
        return
    for path in (PRICE_SEED, SENTIMENT_SEED, PRICES, SENTIMENT):
        if path.exists():
            path.unlink()


def seed_market():
    if MANIFEST.exists():
        manifest = _read_json(MANIFEST)
        if sha256(PRICE_SEED) != manifest["price_seed_sha256"] or sha256(SENTIMENT_SEED) != manifest["sentiment_seed_sha256"]:
            raise RuntimeError("V2.11 種子快照被改過，停止更新")
        if not PRICES.exists() or not SENTIMENT.exists():
            raise RuntimeError("V2.11 市場帳本不完整，拒絕重抄種子")
        return manifest
    _clear_incomplete_seed()
    for path in (PRICE_ORIGIN, SENTIMENT_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    assert_v211(MARKET_DIR / "manifest.json")
    price_dates = pd.to_datetime(pd.read_csv(PRICE_ORIGIN, usecols=["Date"]).Date)
    sentiment_dates = pd.to_datetime(pd.read_csv(SENTIMENT_PATH, usecols=["Date"]).Date)
    price_hash = _copy_once(PRICE_ORIGIN, PRICE_SEED)
    sentiment_hash = _copy_once(SENTIMENT_PATH, SENTIMENT_SEED)
    _copy_once(PRICE_SEED, PRICES)
    _copy_once(SENTIMENT_SEED, SENTIMENT)
    manifest = {
        "price_origin": str(PRICE_ORIGIN.relative_to(ROOT)),
        "sentiment_origin": str(SENTIMENT_PATH.relative_to(ROOT)),
        "price_origin_sha256_at_seed": sha256(PRICE_ORIGIN),
        "sentiment_origin_sha256_at_seed": sha256(SENTIMENT_PATH),
        "official_price_sha256_at_seed": sha256(OFFICIAL_PRICES),
        "official_sentiment_sha256_at_seed": sha256(OFFICIAL_SENTIMENT),
        "price_seed_sha256": price_hash,
        "sentiment_seed_sha256": sentiment_hash,
        "price_seed_through": str(price_dates.max().date()),
        "sentiment_seed_through": str(sentiment_dates.max().date()),
        "seeded_at": datetime.now(TZ).isoformat(),
        "note": "Rows on or before the seed dates are never replaced.",
    }
    _write_json(MANIFEST, manifest)
    return manifest


def assert_history_intact(manifest):
    if sha256(PRICE_SEED) != manifest["price_seed_sha256"] or sha256(SENTIMENT_SEED) != manifest["sentiment_seed_sha256"]:
        raise RuntimeError("V2.11 種子快照被改過，停止更新")
    prices = pd.read_csv(PRICES)
    sentiment = pd.read_csv(SENTIMENT)
    if not same_prefix(pd.read_csv(PRICE_SEED), prices, manifest["price_seed_through"], "Close"):
        raise RuntimeError("已保存的歷史股價被改寫，停止發布")
    if not same_prefix(pd.read_csv(SENTIMENT_SEED), sentiment, manifest["sentiment_seed_through"], "Daily_Sentiment"):
        raise RuntimeError("已保存的歷史情緒被改寫，停止發布")


def _append_csv(path, new_rows):
    """Add rows without rewriting the bytes already saved."""
    current = pd.read_csv(path)
    header = list(current.columns)
    current["Date"] = pd.to_datetime(current["Date"]).dt.normalize()
    fresh = new_rows.copy()
    fresh["Date"] = pd.to_datetime(fresh["Date"]).dt.normalize()
    keys = set(zip(current.Date, current.ticker.astype(str)))
    clash = [str(day.date()) for day, ticker in zip(fresh.Date, fresh.ticker.astype(str)) if (day, ticker) in keys]
    if clash:
        raise RuntimeError(f"拒絕覆蓋既有列：{sorted(set(clash))}")
    fresh["Date"] = fresh["Date"].dt.strftime("%Y-%m-%d")
    body = fresh[header].to_csv(index=False, header=False, lineterminator="\r\n", na_rep="")
    original = path.read_bytes()
    if not original.endswith(b"\n"):
        original += b"\r\n"
    temporary = assert_v211(path.with_suffix(path.suffix + ".tmp"))
    temporary.write_bytes(original + body.encode("utf-8"))
    temporary.replace(path)


def _remember(day, report):
    state = _read_json(STATE)
    sessions_state = state.setdefault("sessions", {})
    key = str(pd.Timestamp(day).date())
    if key not in sessions_state and report.get("prices_ready") and report.get("sentiment_ready"):
        sessions_state[key] = {
            "ready_at": report.get("fetched_at"),
            "price_sha256": report.get("price_sha256"),
            "sentiment_sha256": report.get("sentiment_sha256"),
            "fetched_at_kind": report.get("fetched_at_kind", "unverified_legacy_record"),
            "news_provenance_status": report.get("news_provenance_status", "retrospective_unverified"),
            "news_provenance_path": report.get("news_provenance_path"),
            "news_available_at": report.get("news_available_at"),
        }
        _write_json(STATE, state)
        return sessions_state[key]
    return sessions_state.get(key)


def source_for(day):
    info = _read_json(STATE).get("sessions", {}).get(str(pd.Timestamp(day).date()))
    if not info:
        return None
    return {
        "price_source": PRICE_SOURCE,
        "sentiment_source": SENTIMENT_SOURCE,
        "calendar": CALENDAR,
        "price_sha256": sha256(PRICES),
        "sentiment_sha256": sha256(SENTIMENT),
        "fetched_at": info.get("ready_at"),
        "fetched_at_kind": info.get("fetched_at_kind", "unverified_legacy_record"),
        "news_provenance_status": info.get("news_provenance_status", "retrospective_unverified"),
        "news_provenance_path": info.get("news_provenance_path"),
        "news_available_at": info.get("news_available_at"),
        "used_previous_day_sentiment": False,
    }


def extend_prices(as_of, now, report, *, clock=None):
    ledger = pd.read_csv(PRICES)
    ledger["Date"] = pd.to_datetime(ledger["Date"]).dt.normalize()
    as_of = pd.Timestamp(as_of).normalize()
    have = ledger.Date.max()
    report["prices_through"] = str(have.date())
    report["price_sha256"] = sha256(PRICES)
    if have >= as_of:
        if not price_day_ready(ledger, as_of):
            raise RuntimeError(f"{as_of.date()} 的股價欄位不完整")
        report["prices_ready"] = True
        report["prices_through"] = str(as_of.date())
        report["appended_price_dates"] = []
        report["price_sha256"] = sha256(PRICES)
        return
    from scripts.prices import refresh_prices

    refresh_prices(as_of, assert_v211(STAGING))
    incoming = pd.read_csv(STAGING)
    ok, diffs = price_basis_ok(ledger, incoming)
    report["price_overlap_relative_diff"] = diffs
    if not ok:
        raise RuntimeError("Yahoo 與已保存收盤不一致，拒絕接上新日期")
    required = sessions(have + pd.Timedelta(days=1), as_of)
    new, missing = select_new_price_rows(ledger, incoming, as_of, required, NAMES)
    if missing:
        raise RuntimeError(f"價格缺少交易日：{missing}")
    report["appended_price_dates"] = []
    if len(new):
        _append_csv(PRICES, new[list(ledger.columns)])
        report["appended_price_dates"] = sorted({str(pd.Timestamp(day).date()) for day in new.Date})
    report["prices_ready"] = True
    report["prices_through"] = str(as_of.date())
    report["price_sha256"] = sha256(PRICES)
    from scripts.news_update import observation_time
    report["price_fetched_at"] = observation_time(clock)


def extend_sentiment(as_of, now, report, *, clock=None):
    from scripts.news_update import (MODEL_ID, fetch_day, matches, to_signal_date, SentimentScorer,
                                    article_provenance, observation_time, load_news_cache,
                                    append_provenance, append_news_cache, provenance_path, cache_row_hash,
                                    PROVENANCE_FIELDS, scorer_provenance)

    manifest = _read_json(MANIFEST)
    existing = pd.read_csv(SENTIMENT)
    existing["Date"] = pd.to_datetime(existing["Date"]).dt.normalize()
    as_of = pd.Timestamp(as_of).normalize()
    seed_end = pd.Timestamp(manifest["sentiment_seed_through"]).normalize()
    locked = locked_signal_dates()
    needed = list(sessions(seed_end + pd.Timedelta(days=1), as_of))
    missing = []
    for day in needed:
        if day in locked:
            if not sentiment_day_ready(existing, day):
                raise RuntimeError(f"{pd.Timestamp(day).date()} 已有訊號，但情緒不完整")
            continue
        if not sentiment_day_ready(existing, day):
            missing.append(pd.Timestamp(day).normalize())
    report["sentiment_through"] = str(existing.Date.max().date())
    report["sentiment_sha256"] = sha256(SENTIMENT)
    report["appended_sentiment_dates"] = []
    if not missing:
        report["sentiment_ready"] = sentiment_day_ready(existing, as_of)
        if not report["sentiment_ready"]:
            raise RuntimeError(f"{as_of.date()} 的情緒未到齊")
        return
    # A Monday signal includes Friday after 13:30 and the whole weekend. The
    # same rule covers multi-day exchange holidays; subtracting one calendar
    # day from the missing signal would silently omit part of that interval.
    previous_sessions = sessions(missing[0] - pd.Timedelta(days=90), missing[0] - pd.Timedelta(days=1))
    if not len(previous_sessions):
        raise RuntimeError("找不到新聞來源期間的前一交易日")
    start = pd.Timestamp(previous_sessions[-1]).normalize()
    scanned = []
    coverage = {}
    for date in pd.date_range(start, as_of, freq="D"):
        entries = fetch_day(date)
        fetched_at = observation_time(clock)
        print(f"[V211][NEWS] {date.strftime('%Y-%m-%d')}: fetched {len(entries)} articles", flush=True)
        scanned.extend([{**entry, **article_provenance(entry, fetched_at)} for entry in entries])
        coverage[str(date.date())] = {"articles": len(entries), "completed_at": fetched_at,
                                    "timestamp_kind": "observed_completion"}
    old = load_news_cache(CACHE)
    already = set(zip(old.news_id.astype(str), old.ticker.astype(str))) if len(old) else set()
    new = []
    observations = []
    for article in scanned:
        signal = to_signal_date(article["publish_time"])
        if signal <= seed_end:
            continue
        content = article["title"] + " " + article["summary"]
        for ticker in TICKERS:
            key = (article["news_id"], ticker)
            if not matches(content, ticker):
                continue
            evidence = {field: article.get(field) for field in PROVENANCE_FIELDS
                        if field not in ("available_at", "cache_row_hash", "provenance_status")}
            observations.append({"record_type": "article_observation", "news_id": article["news_id"],
                                 "ticker": ticker, "Date": str(signal.date()),
                                 "publish_time": article["publish_time"], **evidence})
            if key in already:
                continue
            new.append({
                "news_id": article["news_id"],
                "ticker": ticker,
                "publish_time": article["publish_time"],
                "Date": str(signal.date()),
                "text": content,
                **evidence,
            })
            already.add(key)
    if new:
        scorer = SentimentScorer()
        scores = {}
        for row in new:
            if row["news_id"] not in scores:
                scores[row["news_id"]] = scorer.score(row["text"][:1600])
            row["Daily_Sentiment"] = scores[row["news_id"]]
            row["Model_ID"] = MODEL_ID
            row.update(scorer_provenance(scorer))
            del row["text"]
            row["available_at"] = observation_time(clock)
            row["cache_row_hash"] = cache_row_hash(row)
            row["provenance_status"] = "observed_at_recorded_time"
            observations.append({"record_type": "scored_cache_row", **row})
        new_frame = pd.DataFrame(new)
        old = pd.concat([old, new_frame], ignore_index=True) if len(old) else new_frame.copy()
    if len(old):
        old = old.drop_duplicates(["news_id", "ticker"]).sort_values(["Date", "ticker", "news_id"])
    built = rows_for_sessions(old, missing)
    for day in missing:
        if not sentiment_day_ready(built, day):
            raise RuntimeError(f"{pd.Timestamp(day).date()} 情緒仍不完整，不寫入")
    if existing.Date.isin(missing).any():
        raise RuntimeError("情緒檔含有未完成日期，拒絕改寫已保存的列")
    # Only a fully successful source interval may certify an empty news day.
    # Persist coverage before appending sentiment, so an interrupted append can
    # be retried without any already-published sentiment row being rewritten.
    scan_state = _read_json(SCAN_COVERAGE)
    scan_state.setdefault("calendar_days", {}).update(coverage)
    completed_at = observation_time(clock)
    provenance_file = provenance_path(CACHE)
    def session_evidence(day):
        part = old[old.Date.astype(str) == str(day.date())]
        verified = not len(part) or bool((part.provenance_status == "observed_at_recorded_time").all())
        scoring_verified = not len(part) or bool((part.scoring_provenance_status == "resolved_checkpoint_recorded").all())
        # Later retrieval cannot prove that an earlier historical decision had the articles.
        timely = pd.Timestamp(completed_at).date() <= day.date()
        return {"source_start": str(start.date()), "source_end": str(as_of.date()),
                "completed_at": completed_at, "available_at": completed_at,
                "timestamp_kind": "observed_completion",
                "provenance_status": "observed_at_recorded_time" if verified and timely else "retrospective_unverified",
                "scoring_provenance_status": ("resolved_checkpoint_recorded" if len(part) and scoring_verified
                                              else "not_applicable_no_articles" if not len(part)
                                              else "checkpoint_unverified_legacy_or_custom_scorer"),
                "news_provenance_path": str(provenance_file),
                "unverified_article_rows": int((part.provenance_status != "observed_at_recorded_time").sum()) if len(part) else 0}
    scan_state.setdefault("signal_sessions", {}).update({
        str(day.date()): session_evidence(day)
        for day in missing
    })
    assert_v211(provenance_file)
    append_provenance(CACHE, observations)
    if new:
        append_news_cache(CACHE, new_frame)
    _write_json(SCAN_COVERAGE, scan_state)
    _append_csv(SENTIMENT, built)
    report["sentiment_ready"] = True
    report["sentiment_through"] = str(as_of.date())
    report["sentiment_sha256"] = sha256(SENTIMENT)
    report["appended_sentiment_dates"] = [str(pd.Timestamp(day).date()) for day in missing]
    report["sentiment_fetched_at"] = completed_at
    report["news_provenance_status"] = scan_state["signal_sessions"][str(as_of.date())]["provenance_status"]
    report["news_scoring_provenance_status"] = scan_state["signal_sessions"][str(as_of.date())]["scoring_provenance_status"]
    report["news_provenance_path"] = str(provenance_file)
    report["new_articles"] = len(new)
    report["news_source_start"] = str(start.date())
    report["news_source_end"] = str(as_of.date())


def load_market_context():
    if not market_files_ready():
        raise FileNotFoundError("V2.11 市場快照尚未建立")
    manifest = _read_json(MANIFEST)
    assert_history_intact(manifest)
    raw_max = pd.to_datetime(pd.read_csv(PRICES, usecols=["Date"]).Date).max()
    panel = load_panel(PRICES)
    if pd.Timestamp(panel.index.max()) != pd.Timestamp(raw_max).normalize():
        raise RuntimeError("價格最新日有缺列，拒絕把較早收盤當成最新資料")
    sentiment = load_sentiment(SENTIMENT, panel.index)
    if sentiment.loc[pd.Timestamp("2025-08-01"), TICKERS].isna().any():
        raise RuntimeError("實驗情緒副本仍缺少 2025-08-01，拒絕用 0 補上")
    last = panel.index.max()
    if sentiment.loc[last, TICKERS].isna().any():
        raise RuntimeError(f"{last.date()} 情緒未到齊，拒絕沿用前一日情緒")
    frames, risk = build_frames(panel, sentiment)
    one, five, excess = executable_labels(panel)
    return panel, frames, risk, one, five, excess


def _market_report(now):
    if now.tzinfo is None:
        raise ValueError("時間必須帶時區")
    completed = pd.Timestamp(current_completed_session(now)).normalize()
    return {
        "ready_to_predict": False,
        "completed_session": str(completed.date()),
        "clock": now.astimezone(TZ).isoformat(),
        "run_started_at": now.astimezone(TZ).isoformat(),
        "prices_ready": False,
        "sentiment_ready": False,
        "price_source": PRICE_SOURCE,
        "sentiment_source": SENTIMENT_SOURCE,
        "calendar": CALENDAR,
        "used_previous_day_sentiment": False,
        "v211_module_downloads": False,
        "error": None,
        "price_error": None,
        "sentiment_error": None,
        "historical_sentiment_provenance": "retrospective_unverified",
        "news_provenance_schema_version": 1,
        "appended_price_dates": [],
        "appended_sentiment_dates": [],
    }


def inspect_market(now=None):
    """Read the canonical ledger without creating files, seeding, or fetching."""
    now = now or datetime.now(TZ)
    report = _market_report(now)
    completed = pd.Timestamp(report["completed_session"])
    try:
        if not market_files_ready():
            report["gate"] = "market_files_missing"
            report["error"] = "共用市場帳本尚未建立"
            return report
        manifest = _read_json(MANIFEST)
        assert_history_intact(manifest)
        for field, path, validator, label in (
            ("prices", PRICES, price_day_ready, "price"),
            ("sentiment", SENTIMENT, sentiment_day_ready, "sentiment"),
        ):
            try:
                frame = pd.read_csv(path)
                report[f"{field}_through"] = str(pd.to_datetime(frame.Date).max().date())
                report[f"{label}_sha256"] = sha256(path)
                report[f"{field}_ready"] = validator(frame, completed)
                if not report[f"{field}_ready"]:
                    report[f"{label}_error"] = f"{completed.date()} 的{field}未完整到齊"
            except Exception as exc:
                report[f"{label}_error"] = f"{type(exc).__name__}: {exc}"
        report["ready_to_predict"] = bool(report["prices_ready"] and report["sentiment_ready"])
        report["gate"] = "ready" if report["ready_to_predict"] else ("prices_missing" if not report["prices_ready"] else "sentiment_waiting")
        errors = [report[key] for key in ("price_error", "sentiment_error") if report[key]]
        report["error"] = " | ".join(errors) if errors else None
        stored = _read_json(STATE).get("sessions", {}).get(str(completed.date()), {})
        report["fetched_at"] = stored.get("ready_at")
        report["fetched_at_kind"] = stored.get("fetched_at_kind", "unverified_legacy_record")
        evidence = _read_json(SCAN_COVERAGE).get("signal_sessions", {}).get(str(completed.date()), {})
        report["news_provenance_status"] = evidence.get("provenance_status", "retrospective_unverified")
        report["news_scoring_provenance_status"] = evidence.get("scoring_provenance_status", "checkpoint_unverified_legacy")
        from scripts.news_update import provenance_path
        report["news_provenance_path"] = str(provenance_path(CACHE))
        report["news_available_at"] = evidence.get("available_at")
        if evidence.get("available_at") and pd.Timestamp(evidence["available_at"]) > pd.Timestamp(now):
            report["ready_to_predict"] = False
            report["gate"] = "news_not_yet_available_at_decision"
            report["error"] = "新聞特徵完成時間晚於決策時點"
        if (report["ready_to_predict"] and report["fetched_at_kind"] == "observed_completion" and report.get("fetched_at")
                and pd.Timestamp(report["fetched_at"]) > pd.Timestamp(now)):
            report["ready_to_predict"] = False
            report["gate"] = "market_not_yet_available_at_decision"
            report["error"] = "市場快照完成時間晚於決策時點"
        report["official_price_changed_since_seed"] = sha256(OFFICIAL_PRICES) != manifest["official_price_sha256_at_seed"]
        report["official_sentiment_changed_since_seed"] = sha256(OFFICIAL_SENTIMENT) != manifest["official_sentiment_sha256_at_seed"]
    except Exception as exc:
        report["prices_ready"] = False
        report["sentiment_ready"] = False
        report["ready_to_predict"] = False
        report["gate"] = "integrity_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def update_market(now=None, *, clock=None):
    """Fetch once into the append-only canonical ledger, independently per source."""
    now = now or datetime.now(TZ)
    report = _market_report(now)
    completed = pd.Timestamp(report["completed_session"])
    try:
        manifest = seed_market()
        assert_history_intact(manifest)
        fetch_errors = {}
        for name, extend in (("price", extend_prices), ("sentiment", extend_sentiment)):
            try:
                if clock is None:
                    extend(completed, now, report)
                else:
                    extend(completed, now, report, clock=clock)
            except Exception as exc:
                fetch_errors[name] = f"{type(exc).__name__}: {exc}"
        # Validate the actual saved rows even after a source failed. A complete
        # current day with verified provenance may still be used; stale rows
        # never acquire the requested date simply because a fetch was attempted.
        from scripts.news_update import observation_time
        completed_at = observation_time(clock)
        saved = inspect_market(pd.Timestamp(completed_at).to_pydatetime())
        appended = {key: report[key] for key in ("appended_price_dates", "appended_sentiment_dates")}
        report.update(saved)
        report.update(appended)
        report["run_started_at"] = now.astimezone(TZ).isoformat()
        report["completed_at"] = completed_at
        for name, error in fetch_errors.items():
            report[f"{name}_fetch_error"] = error
        if fetch_errors:
            errors = [f"{name}: {error}" for name, error in fetch_errors.items()]
            if saved.get("error"):
                errors.append(saved["error"])
            report["error"] = " | ".join(errors)
            report["used_existing_current_data"] = bool(saved["ready_to_predict"])
        if report["ready_to_predict"]:
            if not report.get("fetched_at"):
                # Existing unversioned files do not acquire a made-up historical fetch time.
                observed = report.get("appended_price_dates") or report.get("appended_sentiment_dates")
                if observed:
                    report["fetched_at"] = completed_at
                    report["fetched_at_kind"] = "observed_completion"
            _remember(completed, report)
    except Exception as exc:
        report["prices_ready"] = False
        report["sentiment_ready"] = False
        report["ready_to_predict"] = False
        report["gate"] = "integrity_error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    _append_audit(report)
    return report
