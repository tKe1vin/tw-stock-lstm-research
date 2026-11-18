"""V2 and V3 share native comparison controls without mixing their evidence."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("altair")
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/app.py"
TABS = ["結果總覽", "預測與驗證", "成本與風險", "每日運作", "優化實驗", "來源與限制"]


def portal(page):
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at = at.radio(key="project_page").set_value(page).run()
    if page == "V2研究":
        at = at.radio(key="v2_research_scope").set_value("tracking").run()
    return at


def test_both_research_versions_have_identical_six_native_tabs():
    for page in ("V2研究", "V3研究"):
        at = portal(page)
        assert not at.exception and not at.error
        assert [tab.label for tab in at.tabs] == TABS
        assert at.metric and at.dataframe
        assert at.get("arrow_vega_lite_chart"), page
        assert len(at.title) == 1
        styles = "\n".join(item.value for item in at.markdown)
        assert ".stApp {background:#101418" not in styles
        assert 'svg[role="img"] {background:#101418' not in styles


def test_v2_wealth_relative_and_drawdown_views_preserve_frozen_v3_conclusion():
    at = portal("V2研究")
    assert not at.exception and not at.error
    for view in ("成本後淨值", "相對基準", "回撤路徑"):
        at.radio(key="v2_curve_view").set_value(view).run()
        assert not at.exception and not at.error
        assert at.radio(key="v2_curve_view").value == view
        assert at.get("arrow_vega_lite_chart")
        assert at.radio(key="v2_data_mode").value == "saved"
    at.radio(key="project_page").set_value("V3研究").run()
    assert not at.exception and not at.error
    values = {metric.label: metric.value for metric in at.metric}
    assert values["新聞候選淨報酬 · 未採用"] == "13.00%"
    assert values["0050基準淨報酬"] == "15.90%"
    assert any("尚未通過採用門檻" in item.value for item in at.warning)
