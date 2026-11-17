"""The project portal preserves independent evidence and can be browsed offline."""
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
PAGES = ["專案總覽", "模型與結果", "研究日誌", "新聞與情緒", "V2研究", "V3研究", "文件與重現"]


def project_app():
    return AppTest.from_file(str(APP), default_timeout=30).run()


def retained_evidence():
    paths = (path for folder in ("dashboard/assets", "reports", "data", "models", "outputs",
                                 "experiments/results/v2/v211")
             for path in (ROOT / folder).rglob("*")
             if path.is_file() and "__pycache__" not in path.parts)
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def test_project_overview_is_default_and_all_seven_sections_render():
    at = project_app()
    assert not at.exception
    navigation = at.radio(key="project_page")
    assert navigation.value == "專案總覽"
    assert list(navigation.options) == PAGES
    # The whole-project landing page must not present the rejected V3 model as
    # though it were the only system or a live strategy.
    assert "新聞候選淨報酬 · 未採用" not in {metric.label for metric in at.metric}
    for page in PAGES[1:]:
        at.radio(key="project_page").set_value(page).run()
        assert not at.exception, (page, [exc.message for exc in at.exception])
        assert not at.error, (page, [error.value for error in at.error])
        assert at.radio(key="project_page").value == page


def test_v3_integrity_failure_isolated_from_other_project_evidence(monkeypatch):
    def reject(*_args, **_kwargs):
        raise data.DataIntegrityError("altered V3 presentation snapshot")

    monkeypatch.setattr(data, "load_study", reject)
    at = project_app()
    assert not at.exception and not at.error
    for page in ("研究日誌", "新聞與情緒", "V2研究", "文件與重現"):
        at.radio(key="project_page").set_value(page).run()
        assert not at.exception and not at.error
    at.radio(key="project_page").set_value("V3研究").run()
    assert not at.exception
    assert len(at.error) == 1
    assert "封存資料檢查未通過" in at.error[0].value
    assert "新聞候選淨報酬 · 未採用" not in {metric.label for metric in at.metric}
    at.radio(key="project_page").set_value("研究日誌").run()
    assert not at.exception and not at.error


def test_entire_project_browsing_is_offline_and_preserves_all_saved_evidence(monkeypatch):
    import requests

    def no_network(*_args, **_kwargs):
        raise AssertionError("Dashboard browsing must use retained local evidence")

    before = retained_evidence()
    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    for module in ("tensorflow", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, module, None)
    at = project_app()
    assert not at.exception
    for page in PAGES[1:]:
        at.radio(key="project_page").set_value(page).run()
        assert not at.exception, (page, [exc.message for exc in at.exception])
    assert retained_evidence() == before
