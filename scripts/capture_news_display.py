"""Capture current public article titles for display, never for historical features.

Default mode reads local metadata only. Explicit --capture appends titles for
already-scored article IDs; original scores and provenance are never rewritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
METADATA = "data/news_display_metadata.jsonl"
PURPOSE = "display_enrichment_current_retrieval"
CACHES = ("data/news_scored_cache.csv", "experiments/results/v2/v211/market/news_cache.csv")


class TitleParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.metadata = {}

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "meta":
            values = dict(attrs)
            name = values.get("property") or values.get("name")
            if name in ("og:title", "og:url"):
                self.metadata[name] = values.get("content", "")


def article_url(news_id):
    if not re.fullmatch(r"[0-9]{1,16}", str(news_id)):
        raise ValueError("Article ID must be numeric")
    return "https://news.cnyes.com/news/id/" + str(news_id)


def saved_ids(root=ROOT):
    result = set()
    for relative in CACHES:
        path = Path(root) / relative
        if path.is_symlink():
            raise ValueError("Scored cache must not be a symlink")
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    identity = row.get("news_id", "")
                    article_url(identity)
                    result.add(identity)
    return sorted(result, key=int, reverse=True)


def validate_record(row):
    allowed = {"schema_version", "news_id", "title", "url", "observed_at", "purpose", "title_sha256"}
    if not isinstance(row, dict) or set(row) != allowed:
        raise ValueError("Only compact display-title fields may be published")
    title = row["title"]
    observed = datetime.fromisoformat(row["observed_at"])
    if (row["schema_version"] != 1 or row["purpose"] != PURPOSE or row["url"] != article_url(row["news_id"])
            or not isinstance(title, str) or not title.strip() or len(title) > 2000
            or observed.tzinfo is None or any(ord(c) < 32 for c in title)
            or row["title_sha256"] != hashlib.sha256(title.encode()).hexdigest()):
        raise ValueError("Invalid display title, origin, timestamp or hash")


def validate_metadata(path, previous=None):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Display metadata must not be a symlink")
    raw = path.read_bytes() if path.is_file() else b""
    if previous is not None and not raw.startswith(previous):
        raise ValueError("Committed title metadata must remain an unchanged prefix")
    rows = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        validate_record(row)
        rows.append(row)
    return rows


def parse_title(html, news_id):
    parser = TitleParser()
    parser.feed(html)
    title = parser.metadata.get("og:title", "").strip()
    canonical = parser.metadata.get("og:url", "").rstrip("/")
    if not title or title in ("鉅亨網", "Anue鉅亨") or (canonical and canonical != article_url(news_id)):
        raise ValueError("Source lacks an identifiable article title")
    return title


def fetch_title(news_id):
    import requests

    url = article_url(news_id)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (university research display)"},
                            timeout=(5, 12), allow_redirects=False)
    if response.status_code != 200 or response.url != url:
        raise ValueError(f"Article response HTTP {response.status_code}")
    if len(response.content) > 4 * 1024 * 1024:
        raise ValueError("Article response exceeds display-title size bound")
    return parse_title(response.text, news_id)


def capture(root=ROOT, *, limit=50, fetcher=fetch_title, clock=None):
    root = Path(root).resolve()
    path = root / METADATA
    before = path.read_bytes() if path.is_file() else b""
    existing = validate_metadata(path)
    done = {row["news_id"] for row in existing}
    pending = [identity for identity in saved_ids(root) if identity not in done]
    if not 1 <= limit <= 100:
        raise ValueError("Limit must be between 1 and 100")

    def acquire(identity):
        try:
            title = fetcher(identity)
            observed = (clock() if clock else datetime.now(ZoneInfo("Asia/Taipei"))).isoformat()
            return {"schema_version": 1, "news_id": identity, "title": title,
                    "url": article_url(identity), "observed_at": observed, "purpose": PURPOSE,
                    "title_sha256": hashlib.sha256(title.encode()).hexdigest()}, None
        except Exception as exc:
            return None, {"news_id": identity, "error_type": type(exc).__name__}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(acquire, pending[:limit]))
    rows = [row for row, error in results if row is not None]
    errors = [error for row, error in results if error is not None]
    if rows:
        if before and not before.endswith(b"\n"):
            raise ValueError("Existing metadata lacks its record separator")
        encoded = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows)
        # Validate the complete batch before touching a published file.
        for row in rows:
            validate_record(row)
        if (path.read_bytes() if path.exists() else b"") != before:
            raise ValueError("Metadata changed during capture; refusing concurrent append")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(encoded)
        validate_metadata(path, before)
    return {"captured": len(rows), "already_saved": len(done), "pending_after": len(pending) - len(rows),
            "failures": errors, "purpose": PURPOSE, "historical_availability_proof": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    result = capture(limit=args.limit) if args.capture else {
        "read_only": True, "scored_article_ids": len(saved_ids()),
        "display_titles": len(validate_metadata(ROOT / METADATA)), "historical_availability_proof": False}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
