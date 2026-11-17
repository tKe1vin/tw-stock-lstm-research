"""Regression coverage for the actual delivered Streamlit entry points."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("relative", ["streamlit_v2_app.py", "dashboard_v2/app.py"])
def test_entry_renders_once_and_all_research_sections_work(relative):
    at = AppTest.from_file(str(ROOT / relative), default_timeout=30).run()
    assert not at.exception
    assert len(at.title) == 1
    area = next(item for item in at.radio if item.label == "研究入口")
    area.set_value("555 升級研究與每日交易").run()
    assert not at.exception
    sections = next(item for item in at.radio if item.label == "升級研究子頁")
    for section in sections.options:
        sections.set_value(section).run()
        assert not at.exception, (relative, section, [exc.message for exc in at.exception])


def test_news_score_versions_and_scoring_time_match_main_evidence():
    from dashboard.news import load_news_archive
    from dashboard_v2.data_access.load import news_observations
    current = load_news_archive(ROOT)["articles"]
    rows = news_observations()["rows"]
    assert len(rows) == len(current)
    matched = current.set_index("cache_row_hash")
    differs = False
    for row in rows:
        source = matched.loc[row["cache_row_hash"]]
        assert row["available_at"] == source.available_at
        assert row["fetched_at"] == source.fetched_at
        assert row["title_observed_at"] == source.title_observed_at
        if source.available_at and source.fetched_at and source.available_at != source.fetched_at:
            differs = True
            assert row["available_at"] != row["fetched_at"]
    assert differs


def test_auto_refresh_can_be_enabled_and_navigated_without_repeated_main_calls():
    at = AppTest.from_file(str(ROOT / "streamlit_v2_app.py"), default_timeout=30).run()
    at.checkbox[0].check().run()
    assert not at.exception and len(at.title) == 1
    next(item for item in at.radio if item.label == "研究入口").set_value("555 升級研究與每日交易").run()
    assert not at.exception
    next(item for item in at.radio if item.label == "升級研究子頁").set_value("交易日誌").run()
    assert not at.exception
