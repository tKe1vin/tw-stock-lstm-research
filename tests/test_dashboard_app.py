"""Actual Streamlit interactions preserve research scope and never execute trades."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from dashboard import data


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/app.py"


def app():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    return at.radio(key="project_page").set_value("V3研究").run()


def values(at):
    return {metric.label: metric.value for metric in at.metric}


def protected_snapshot():
    """Presentation interactions may not write any retained research evidence."""
    paths = []
    for folder in ("dashboard/assets", "reports", "data", "models", "outputs", "experiments/results/v2/v211"):
        paths.extend(path for path in (ROOT / folder).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts)
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def test_default_study_renders_actual_metrics_and_rejected_news_status():
    at = app()
    assert not at.exception
    result = values(at)
    assert result["新聞候選淨報酬 · 未採用"] == "13.00%"
    assert result["0050基準淨報酬"] == "15.90%"
    assert result["新聞對0050差額"] == "-2.90 個百分點"
    assert result["新聞對同架構價格控制"] == "+0.12 個百分點"
    assert any("尚未通過採用門檻" in message.value for message in at.warning)
    assert any("B0_benchmark" in str(item.value) for item in at.json)
    assert len(at.tabs) == 6


def test_version_switch_updates_both_period_and_metrics_without_cross_period_ranking():
    at = app()
    at.selectbox(key="study_version").set_value("v3.2").run()
    assert not at.exception
    assert at.selectbox(key="study_version").value == "v3.2"
    assert values(at)["新聞候選淨報酬 · 未採用"] == "-0.58%"
    assert values(at)["0050基準淨報酬"] == "-0.58%"
    periods = [item.value for item in at.markdown if 'class="subline"' in item.value]
    assert len(periods) == 1
    assert "2026-08-04" in periods[0] and "2026-11-03" in periods[0]
    assert "2026-04-15" not in periods[0]
    at.selectbox(key="study_version").set_value("v3.3").run()
    assert not at.exception
    assert values(at)["新聞候選淨報酬 · 未採用"] == "13.00%"


def test_all_chart_views_and_equal_weight_control_render_without_changing_selection():
    at = app()
    for view in ("相對0050", "回撤路徑", "成本後淨值"):
        at.radio(key="curve_view").set_value(view).run()
        assert not at.exception
        assert at.radio(key="curve_view").value == view
    at.checkbox(key="show_equal").check().run()
    assert not at.exception and at.checkbox(key="show_equal").value is True
    assert values(at)["新聞對0050差額"] == "-2.90 個百分點"
    assert any("B0_benchmark" in str(item.value) for item in at.json)


@pytest.mark.parametrize("error", [data.DataIntegrityError("altered retained artifact"),
                                  FileNotFoundError("missing retained artifact")])
def test_invalid_or_missing_evidence_stops_before_any_performance_cards(monkeypatch, error):
    def reject(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(data, "load_study", reject)
    at = app()
    assert not at.exception
    assert len(at.error) == 1
    assert not at.metric and not at.dataframe


def test_interactions_are_read_only_and_work_without_provider_or_model_runtime(monkeypatch):
    import requests

    def no_network(*_args, **_kwargs):
        raise AssertionError("Presentation must not access a provider")

    before = protected_snapshot()
    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    at = app()
    at.checkbox(key="show_equal").check().run()
    at.radio(key="curve_view").set_value("相對0050").run()
    at.selectbox(key="study_version").set_value("v3.2").run()
    assert not at.exception
    assert protected_snapshot() == before
