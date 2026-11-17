"""News display preserves scored versions and separates later title retrieval."""
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from dashboard import news

ROOT = Path(__file__).resolve().parents[1]


def test_real_archive_deduplicates_same_scored_row_without_dropping_stock_matches():
    archive = news.load_news_archive()
    frame = archive["articles"]
    assert len(frame) == 44 and frame.news_id.nunique() == 40
    assert frame.cache_row_hash.is_unique
    assert frame.available_at.notna().sum() == 23
    assert news.LEGACY in set(frame.timing_status) and news.LATE in set(frame.timing_status)
    assert archive["has_page_screenshots"] is False
    assert not archive["calendar"].empty and not archive["daily"].empty


def test_later_title_does_not_authenticate_original_score_or_overwrite_cache(tmp_path):
    path = tmp_path / "data/news_scored_cache.csv"
    path.parent.mkdir()
    shutil.copyfile(ROOT / "data/news_scored_cache.csv", path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    row = pd.read_csv(path).iloc[0]
    metadata = {"news_id": str(row.news_id), "title": "目前取得的展示標題", "url": "javascript:alert(1)",
                "observed_at": "2026-11-17T15:00:00+08:00"}
    (path.parent / "news_display_metadata.jsonl").write_text(json.dumps(metadata) + "\n")
    frame = news.load_news_archive(tmp_path)["articles"]
    shown = frame[frame.news_id.eq(str(row.news_id))].iloc[0]
    assert shown.title == metadata["title"]
    assert shown.provenance_status == "retrospective_unverified" and shown.available_at is None
    assert shown.title_observed_at == metadata["observed_at"]
    assert shown.source_url.startswith("https://news.cnyes.com/")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_news_filters_are_joint_literal_search_and_keep_missing_distinct_from_zero():
    frame = news.load_news_archive()["articles"]
    row = frame.iloc[0]
    result = news.filter_articles(frame, day=row.Date, tickers=[row.ticker], keyword=row.news_id)
    assert len(result) == 1 and result.iloc[0].cache_row_hash == row.cache_row_hash
    assert news.filter_articles(frame, keyword="[unmatched regex").empty
    changed = frame.iloc[:2].copy()
    changed["Daily_Sentiment"] = [0.0, float("nan")]
    assert len(news.filter_articles(changed, score_kind="中性")) == 1
    assert len(news.filter_articles(changed, score_kind="缺評分")) == 1


def test_changed_scored_provenance_fails_closed(tmp_path):
    relative = Path("experiments/results/v2/v211/market")
    target = tmp_path / relative
    target.mkdir(parents=True)
    for name in ("news_cache.csv", "news_cache_provenance.jsonl"):
        shutil.copyfile(ROOT / relative / name, target / name)
    path = target / "news_cache_provenance.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    scored = next(row for row in records if row.get("record_type") == "scored_cache_row")
    scored["Daily_Sentiment"] += 0.01
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(news.NewsEvidenceError, match="雜湊"):
        news.load_news_archive(tmp_path)


def test_empty_archive_and_unsafe_source_links_are_explicit(tmp_path):
    empty = news.load_news_archive(tmp_path)
    assert empty["articles"].empty and empty["warnings"]
    for url in ("javascript:alert(1)", "https://user:secret@example.com", "https://example.com\n", "/local"):
        assert news.safe_source_url(url) == ""
    assert news.safe_source_url("https://news.cnyes.com/news/id/123")
    assert b"'=formula" in news.safe_csv(pd.DataFrame({"title": ["=formula"]}))


def test_actual_news_ui_filters_and_article_details_without_model_runtime(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import sys
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    at = AppTest.from_string("from dashboard.news import render_news\nrender_news()", default_timeout=30).run()
    assert not at.exception and not at.error
    at.selectbox(key="news_score_kind").set_value("負向").run()
    assert not at.exception and not at.error
    at.text_input(key="news_keyword").set_value("unmatched-news-display").run()
    assert not at.exception
    assert any("符合條件：0" in item.value for item in at.markdown)
