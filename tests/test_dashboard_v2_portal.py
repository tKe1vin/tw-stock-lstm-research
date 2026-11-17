"""The unified portal uses saved V2 evidence and keeps explicit demos scoped."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/app.py"


def portal():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    return at.radio(key="project_page").set_value("V2研究").run()


def displayed(at):
    texts = [item.value for kind in (at.markdown, at.caption, at.info, at.warning)
             for item in kind]
    texts.extend(f"{item.label}：{item.value}" for item in at.metric)
    return "\n".join(texts)


def saved_evidence():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("dashboard/assets", "dashboard_v2/fixtures", "reports", "data", "models",
                           "outputs", "experiments/results/v2/v211", "research/v211")
            for path in (ROOT / folder).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts}


def choose_section(at, token):
    """Historical sources and daily operations have independent selectors."""
    key = "v2_archive_section" if token in {"Ridge", "V2 總覽", "原 555 與 V2", "歷史實驗", "限制"} else "v2_section"
    options = at.selectbox(key=key).options
    value = next(option for option in options if token in option)
    return at.selectbox(key=key).set_value(value).run()


def test_v2_portal_defaults_to_saved_v211_and_missing_ridge_not_demo_metrics(monkeypatch):
    for key in ("TW_V211_LOCAL", "TW_RIDGE_ROOT", "TW_GITHUB_REVIEW_ROOT"):
        monkeypatch.delenv(key, raising=False)
    at = portal()
    assert not at.exception and not at.error
    assert at.radio(key="v2_data_mode").value == "saved"
    at = choose_section(at, "V2.11")
    assert not at.exception and not at.error
    signal_root = ROOT / "experiments/results/v2/v211/daily"
    latest = max(path.stem for path in signal_root.glob("????-??-??.csv"))
    assert any(metric.label == "最後保存訊號日" and metric.value == latest for metric in at.metric)
    assert "experiments/results/v2/v211" in displayed(at)
    at = choose_section(at, "Ridge")
    assert not at.exception and not at.error
    text = displayed(at)
    assert "DATA_UNAVAILABLE" in text
    ridge_metrics = [metric for metric in at.metric if "Ridge" in metric.label and "MAE" in metric.label]
    assert ridge_metrics
    assert all(metric.value in {"未提供", "—", "None", "DATA_UNAVAILABLE"} for metric in ridge_metrics)


def test_demo_is_explicit_and_does_not_leak_into_v3_or_new_portal_session(monkeypatch):
    from dashboard_v2.data_access import prospective

    for key in ("TW_V211_LOCAL", "TW_RIDGE_ROOT", "TW_GITHUB_REVIEW_ROOT"):
        monkeypatch.delenv(key, raising=False)
    standalone_root = prospective.v211_root()
    at = portal()
    at = choose_section(at, "交易日誌")
    latest_day = max(option for option in at.selectbox(key="v2_journal_day").options if option != "全部")
    at.selectbox(key="v2_journal_day").set_value(latest_day).run()
    at.radio(key="v2_data_mode").set_value("demo").run()
    assert not at.exception and not at.error
    assert "SYNTHETIC_DEMO" in displayed(at)
    at = choose_section(at, "V2.11")
    assert not at.exception and not at.error
    assert "fixtures/display_snapshots" in displayed(at)
    assert prospective.v211_root() == standalone_root
    at.radio(key="project_page").set_value("V3研究").run()
    assert not at.exception and not at.error
    assert "SYNTHETIC_DEMO" not in displayed(at)
    metrics = {item.label: item.value for item in at.metric}
    assert metrics["新聞候選淨報酬 · 未採用"] == "13.00%"
    another = portal()
    assert not another.exception and not another.error
    assert another.radio(key="v2_data_mode").value == "saved"
    another = choose_section(another, "Ridge")
    assert "DATA_UNAVAILABLE" in displayed(another)
    assert prospective.v211_root() == standalone_root


def test_next_strategy_separates_forecast_entry_from_actual_rebalance_eligibility():
    from dashboard_v2.data_access.prospective import display_sources, next_day_strategy
    with display_sources(mode="saved", root=ROOT):
        expected = next_day_strategy()
    at = choose_section(portal(), "隔日策略")
    assert not at.exception and not at.error
    text = displayed(at)
    assert "預測假設進場" in text and "再平衡排程" in text and "交易資格" in text
    assert expected["forecast_entry"] in text
    assert expected["scheduled_rebalance"] in text
    if str(expected["trade_eligible"]).lower() != "true":
        assert "未具備已保存的再平衡交易資格" in text
        assert expected["execution_status"] in text


def test_all_v2_sections_use_no_model_or_provider_and_preserve_every_source(monkeypatch):
    import requests
    import dashboard_v2.app as standalone

    def reject(*_args, **_kwargs):
        raise AssertionError("Portal browsing cannot fetch, train, or invoke standalone main")

    before = saved_evidence()
    monkeypatch.setattr(requests.sessions.Session, "request", reject)
    monkeypatch.setattr(standalone, "main", reject)
    for module in ("tensorflow", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, module, None)
    at = portal()
    assert not at.exception and len(at.title) == 1
    for mode in ("saved", "demo"):
        at.radio(key="v2_data_mode").set_value(mode).run()
        assert not at.exception and not at.error
        if mode == "demo":
            assert any(metric.label == "已發生成交費" and metric.value == "未提供" for metric in at.metric)
            fee_metrics = [metric for metric in at.metric if "費" in metric.label]
            assert all("nan" not in metric.value.lower() and "None" not in metric.value
                       for metric in fee_metrics)
            assert any("不以0元補齊" in item.value for item in at.info)
        for key in ("v2_section", "v2_archive_section"):
            for section in list(at.selectbox(key=key).options):
                at.selectbox(key=key).set_value(section).run()
                assert not at.exception, (mode, section, [item.message for item in at.exception])
                assert not at.error, (mode, section, [item.value for item in at.error])
                assert len(at.title) == 1
                assert ".stApp {background:#101418" not in displayed(at)
                assert 'svg[role="img"] {background:#101418' not in displayed(at)
    assert saved_evidence() == before
