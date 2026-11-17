"""Read saved research files. This module does not train, publish, or write."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2603.TW": "長榮",
    "2002.TW": "中鋼",
    "2891.TW": "中信金",
    "^TWII": "台灣加權股價指數",
}


def stable_text(path):
    path = Path(path)
    first = path.read_bytes()
    second = path.read_bytes()
    if not first or first != second:
        raise RuntimeError(f"檔案不完整或正在寫入：{path}")
    return first.decode("utf-8-sig")


def csv_rows(path):
    return list(csv.DictReader(stable_text(path).splitlines()))


def json_file(path):
    return json.loads(stable_text(path))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def historical_official():
    path = ROOT / "reports" / "daily" / "2026-11-04" / "strategy_comparison.csv"
    table = {row["Strategy_ID"]: row for row in csv_rows(path)}
    return {
        "source": rel(path),
        "protocol": "2026-11-04 已保存比較；444 與當時六特徵 555",
        "status": "HISTORICAL_OFFICIAL",
        "as_of": table["444"]["As_Of_Date"],
        "rows": table,
        "twii_id": "^TWII",
        "twii_note": "此列是同一比較檔中的理論價格指數，不是可交易 ETF 淨報酬。",
    }


def paired_metrics():
    path = ROOT / "reports" / "research_revision" / "2026-11-06" / "paired" / "metrics.csv"
    protocol = json_file(path.parent / "protocol.json")
    return {
        "source": rel(path),
        "protocol": protocol["protocol_version"],
        "status": protocol["result_status"],
        "causal_validated": protocol["causal_validated"],
        "rows": csv_rows(path),
        "rules": protocol["rules"],
    }


def paired_nav(arm=None):
    path = ROOT / "reports" / "research_revision" / "2026-11-06" / "paired" / "nav.csv"
    data = csv_rows(path)
    if arm:
        data = [row for row in data if row["Arm"] == arm]
    return {"source": rel(path), "rows": data}


def paired_weights(arm):
    path = ROOT / "reports" / "research_revision" / "2026-11-06" / "paired" / "weights.csv"
    return [row for row in csv_rows(path) if row["Arm"] == arm]


def paired_returns(arm):
    path = ROOT / "reports" / "research_revision" / "2026-11-06" / "paired" / "analysis" / "daily_returns.csv"
    return [row for row in csv_rows(path) if row["Arm"] == arm]


def v2_comparison():
    path = ROOT / "reports" / "v210_comparison" / "comparison.json"
    payload = json_file(path)
    books = payload["books"]
    quality = [
        row for row in payload["quality"]
        if row["Model"] == "rolling_selected"
    ]
    return {
        "source": rel(path),
        "candidate": payload["candidate"],
        "weights_saved": payload["weights_saved"],
        "coverage": payload["coverage"],
        "books": books,
        "ranked": [row for row in books if row.get("ROI_Comparable_With_Equal") is True],
        "blocked": [row for row in books if row.get("ROI_Comparable_With_Equal") is not True],
        "quality": quality,
        "status": "EXPLORATORY_RESULT",
    }


def v211_status():
    """Prefer verified local prospective snapshot over the stale clone copy."""
    from dashboard_v2.data_access.prospective import v211_status_local

    data = v211_status_local()
    data["source_commit"] = "local-prospective-snapshot"
    return data


def v211_signals():
    from dashboard_v2.data_access.prospective import v211_signals_local

    return v211_signals_local()


def v211_fills():
    from dashboard_v2.data_access.prospective import v211_fills_local

    return v211_fills_local()


def v211_nav():
    from dashboard_v2.data_access.prospective import v211_nav_local

    return v211_nav_local()


def news_observations():
    # Share the main portal's score-row hash join. An article observation alone
    # must never acquire another version's score or its earlier fetched time.
    from dashboard.news import load_news_archive

    archive = load_news_archive(ROOT)
    rows = []
    for item in archive["articles"].to_dict("records"):
        score = item.get("Daily_Sentiment")
        if score is None or not math.isfinite(float(score)):
            score = ""
        rows.append({
            "news_id": item.get("news_id", ""),
            "ticker": item.get("ticker", ""),
            "title": item.get("title", ""),
            "title_status": "DISPLAY_TITLE_LATER_RETRIEVAL" if item.get("title") else "TITLE_NOT_STORED",
            "title_observed_at": item.get("title_observed_at"),
            "source": item.get("cache_sources", ""),
            "article_url": item.get("source_url", ""),
            "publish_time": item.get("publish_time", ""),
            "first_seen_at": item.get("first_seen_at", ""),
            "fetched_at": item.get("fetched_at", ""),
            "available_at": item.get("available_at", ""),
            "sentiment": score,
            "sentiment_class": sentiment_class(score),
            "decision_use": "UNVERIFIED",
            "evidence": item.get("provenance_status", "retrospective_unverified"),
            "cache_row_hash": item["cache_row_hash"],
        })
    return {
        "source": "experiments/results/v2/v211/market/news_cache_provenance.jsonl",
        "score_source": "；".join(archive["sources"]),
        "rows": rows,
    }


def sentiment_class(value):
    if value is None or str(value).strip() == "":
        return "UNVERIFIED"
    number = float(value)
    if number == 0:
        return "ZERO_UNVERIFIED"
    if number >= 0.3:
        return "正向"
    if number <= -0.3:
        return "負向"
    return "門檻內"
