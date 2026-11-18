"""The historical V2 book stays readable, immutable and isolated from demos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("altair")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/app.py"
REPORT = ROOT / "reports/research_v2_common/2026-11-17"
TABS = ["結果總覽", "預測與驗證", "成本與風險", "每日運作", "優化實驗", "來源與限制"]
ARMS = {
    "static_equal444", "dynamic_news_risk555", "v2_news_lstm",
    "v2_no_news_lstm", "v2_price_lstm", "v2_news_no_risk",
    "v2_matched_equal", "full_equal_5",
}


def portal():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    return at.radio(key="project_page").set_value("V2研究").run()


def displayed(at):
    return "\n".join(item.value for kind in (at.markdown, at.caption, at.info, at.warning)
                     for item in kind)


def metrics(at):
    return {item.label: item.value for item in at.metric}


def saved_evidence():
    folders = ("dashboard/assets", "dashboard_v2/fixtures", "reports", "data", "models",
               "outputs", "experiments/results/v2", "research")
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in folders for path in (ROOT / folder).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts}


def encoded_fields(value, channel):
    """Inspect native Vega encodings without depending on generated dataset IDs."""
    if isinstance(value, dict):
        encoding = value.get("encoding", {}).get(channel, {})
        if isinstance(encoding, dict) and "field" in encoding:
            yield encoding["field"]
        for child in value.values():
            yield from encoded_fields(child, channel)
    elif isinstance(value, list):
        for child in value:
            yield from encoded_fields(child, channel)


def test_saved_defaults_to_complete_historical_book_with_six_native_tabs():
    # This committed report is required evidence, never a generated demo fixture.
    assert (REPORT / "manifest.json").is_file()
    expected = pd.read_csv(REPORT / "metrics.csv").set_index("Arm")
    summary = json.loads((REPORT / "summary.json").read_text())
    at = portal()
    assert not at.exception and not at.error
    assert at.radio(key="v2_data_mode").value == "saved"
    assert at.radio(key="v2_research_scope").value == "historical"
    assert [tab.label for tab in at.tabs] == TABS
    assert len(at.title) == 1
    assert "HISTORICAL_EXPLORATORY" in displayed(at)
    assert "SYNTHETIC_DEMO" not in displayed(at)
    values = metrics(at)
    assert values["V2.12新聞模型淨報酬"] == f"{expected.loc['v2_news_lstm', 'ROI_Pct']:.2f}%"
    assert values["444固定持股淨報酬"] == f"{expected.loc['static_equal444', 'ROI_Pct']:.2f}%"
    assert values["測試訊號日"] == str(summary["test_signal_days"]) == "150"
    assert values["每模型已成熟訊號日"] == str(summary["matured_signal_days"]) == "145"
    assert values["每模型待成熟訊號日"] == str(summary["pending_signal_days"]) == "5"
    assert any("Arm" in item.value and set(item.value.Arm) == ARMS
               for item in at.tabs[0].dataframe)
    assert at.tabs[0].get("arrow_vega_lite_chart")
    assert at.tabs[1].dataframe
    assert at.tabs[2].get("arrow_vega_lite_chart")
    assert at.tabs[4].dataframe and at.tabs[5].dataframe
    # Daily operations retain the separately saved V2.11 timeline.
    assert "最後保存訊號日" in values
    assert "以下每日運作讀取V2.11正式凍結追蹤" in displayed(at)


def test_historical_curve_views_and_arm_controls_preserve_the_saved_summary():
    at = portal()
    assert not at.exception and not at.error
    original = metrics(at)
    names = pd.read_csv(REPORT / "metrics.csv").set_index("Arm").Display_Name.to_dict()
    for view, field in (("成本後淨值", "Wealth_Index"),
                        ("相對基準", "Relative_Wealth_Pct"),
                        ("回撤路徑", "Drawdown_Pct")):
        at.radio(key="v2_curve_view").set_value(view).run()
        assert not at.exception and not at.error
        assert at.radio(key="v2_research_scope").value == "historical"
        chart = at.tabs[0].get("arrow_vega_lite_chart")[0]
        assert field in set(encoded_fields(json.loads(chart.proto.spec), "y"))
        assert metrics(at) == original
    for selected in ([], sorted(ARMS - {"static_equal444"})):
        at.multiselect(key="v2_history_arms").set_value(selected).run()
        assert not at.exception and not at.error
        chart = at.tabs[0].get("arrow_vega_lite_chart")[0]
        spec = json.loads(chart.proto.spec)
        domains = {name for layer in spec["layer"]
                   for name in layer.get("encoding", {}).get("color", {}).get("scale", {}).get("domain", [])}
        assert domains == {names[arm] for arm in ["static_equal444", *selected]}
        assert metrics(at) == original
    for arm in ("static_equal444", "dynamic_news_risk555", "v2_news_lstm"):
        at.selectbox(key="v2_history_exposure").set_value(arm).run()
        assert not at.exception and not at.error
        assert at.selectbox(key="v2_history_exposure").value == arm
        assert metrics(at) == original
    at.radio(key="v2_research_scope").set_value("tracking").run()
    assert not at.exception and not at.error
    assert "V2.11成本後估值報酬" in metrics(at)
    assert "V2.12新聞模型淨報酬" not in metrics(at)
    at.radio(key="v2_research_scope").set_value("historical").run()
    assert not at.exception and not at.error
    assert metrics(at) == original


def test_demo_cannot_read_history_or_leak_into_v3_or_a_fresh_session(monkeypatch):
    import dashboard.v2 as v2

    at = portal()
    assert not at.exception and not at.error
    assert at.radio(key="v2_research_scope").value == "historical"

    def reject(*_args, **_kwargs):
        raise AssertionError("DEMO cannot read the saved historical report")

    with monkeypatch.context() as isolated:
        isolated.setattr(v2, "load_v2_history", reject)
        at.radio(key="v2_data_mode").set_value("demo").run()
        assert not at.exception and not at.error
        assert at.radio(key="v2_research_scope").value == "tracking"
        assert at.radio(key="v2_research_scope").disabled
        assert "SYNTHETIC_DEMO" in displayed(at)
        assert "V2.12新聞模型淨報酬" not in metrics(at)
        assert not at.multiselect
        at.radio(key="project_page").set_value("V3研究").run()
        assert not at.exception and not at.error
        assert "SYNTHETIC_DEMO" not in displayed(at)
        assert metrics(at)["新聞候選淨報酬 · 未採用"] == "13.00%"
        assert metrics(at)["0050基準淨報酬"] == "15.90%"
    fresh = portal()
    assert not fresh.exception and not fresh.error
    assert fresh.radio(key="v2_data_mode").value == "saved"
    assert fresh.radio(key="v2_research_scope").value == "historical"


def test_missing_report_defaults_to_tracking_without_reading_or_synthesizing_history(monkeypatch):
    import dashboard.v2 as v2

    def reject(*_args, **_kwargs):
        raise AssertionError("An unavailable history must not be loaded")

    monkeypatch.setattr(v2, "history_available", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(v2, "load_v2_history", reject)
    at = portal()
    assert not at.exception and not at.error
    assert at.radio(key="v2_data_mode").value == "saved"
    assert at.radio(key="v2_research_scope").value == "tracking"
    assert "V2.11成本後估值報酬" in metrics(at)
    assert "V2.12新聞模型淨報酬" not in metrics(at)
    assert "SYNTHETIC_DEMO" not in displayed(at)


def test_invalid_history_fails_closed_and_tracking_remains_accessible(monkeypatch):
    import dashboard.v2 as v2
    from dashboard.v2_history import V2HistoryError

    def reject(*_args, **_kwargs):
        raise V2HistoryError("QA: historical report hash mismatch")

    monkeypatch.setattr(v2, "history_available", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(v2, "load_v2_history", reject)
    at = portal()
    assert not at.exception and at.error
    assert at.radio(key="v2_research_scope").value == "historical"
    assert "V2.12新聞模型淨報酬" not in metrics(at)
    assert "hash mismatch" in displayed(at)
    at.radio(key="v2_research_scope").set_value("tracking").run()
    assert not at.exception and not at.error
    assert "V2.11成本後估值報酬" in metrics(at)


def test_historical_browsing_uses_no_runtime_or_provider_and_preserves_saved_sources(monkeypatch):
    import requests

    def reject(*_args, **_kwargs):
        raise AssertionError("Historical browsing cannot fetch or train")

    before = saved_evidence()
    monkeypatch.setattr(requests.sessions.Session, "request", reject)
    for module in ("tensorflow", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, module, None)
    at = portal()
    assert not at.exception and not at.error
    assert at.radio(key="v2_research_scope").value == "historical"
    for section in list(at.selectbox(key="v2_section").options):
        at.selectbox(key="v2_section").set_value(section).run()
        assert not at.exception and not at.error, section
    for section in list(at.selectbox(key="v2_archive_section").options):
        at.selectbox(key="v2_archive_section").set_value(section).run()
        assert not at.exception and not at.error, section
    assert saved_evidence() == before
