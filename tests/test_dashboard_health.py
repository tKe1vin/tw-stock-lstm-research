"""Operational dashboard checks beyond the individual page contracts.

These use private copies of retained evidence. They never repair or rewrite a
published report, change a model pack, or interpret screenshots as research.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

# The daily model environment deliberately does not install presentation extras.
pytest.importorskip("streamlit")
pytest.importorskip("altair")
from streamlit.testing.v1 import AppTest

from dashboard import v2_history as history


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard/app.py"


@pytest.fixture
def retained_history(tmp_path):
    report = ROOT / history.REPORT_DIR
    shutil.copytree(report, tmp_path / history.REPORT_DIR)
    manifest = json.loads((report / "manifest.json").read_text())
    archive = Path(manifest["models"]["archive_path"])
    (tmp_path / archive).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / archive, tmp_path / archive)
    return tmp_path, manifest


def portal():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    return app.radio(key="project_page").set_value("V2研究").run()


def metric_values(app):
    return {item.label: item.value for item in app.metric}


def displayed(app):
    return "\n".join(item.value for group in (app.markdown, app.caption, app.info, app.warning)
                     for item in group)


def test_retained_history_is_independent_of_missing_or_future_live_inputs(retained_history):
    root, manifest = retained_history
    original = history.load_v2_history(root)
    assert not (root / "experiments").exists()
    # A live checkout can advance without reopening a frozen 150-day study or
    # maturing the historical study's five deliberately pending signal days.
    live = root / "experiments/results/v2/v211/market"
    live.mkdir(parents=True)
    (live / "prices.csv").write_text("Date,Ticker,Close\n2099-01-05,2330.TW,999999\n")
    (live / "sentiment.csv").write_text("Date,Ticker,Daily_Sentiment\n2099-01-05,2330.TW,1\n")
    # Export provenance is retained as evidence, rather than re-read from live
    # files that may be absent in a presentation-only deployment.
    for relative in manifest["sources"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"future live checkout; not the frozen report input")
    advanced = history.load_v2_history(root)
    assert (advanced["period_start"], advanced["period_end"]) == ("2026-04-10", "2026-11-17")
    assert advanced["summary"]["valuation_sessions"] == 150
    assert advanced["summary"]["pending_signal_days"] == 5
    for table in history.TABLES:
        pd.testing.assert_frame_equal(original[table], advanced[table])
    archive = root / manifest["models"]["archive_path"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["models"]["sha256"]


@pytest.mark.parametrize("target", ["nav.csv", "manifest.json", "model"])
def test_partial_deployment_during_read_hides_history_and_other_pages_recover(
        retained_history, monkeypatch, target):
    import dashboard.v2 as v2

    root, manifest = retained_history
    relative = manifest["models"]["archive_path"] if target == "model" else history.REPORT_DIR / target
    watched = (root / relative).resolve()
    original_read = Path.read_bytes
    calls = 0

    def changing_read(path):
        nonlocal calls
        raw = original_read(path)
        if path.resolve() == watched:
            calls += 1
            if calls >= 2:
                return raw + b"\npartial deployment"
        return raw

    # Use the real loader against a private copy, including its final complete
    # read check. A fake loader raising an error would miss this race.
    monkeypatch.setattr(v2, "load_v2_history", lambda _root: history.load_v2_history(root))
    with monkeypatch.context() as racing:
        racing.setattr(Path, "read_bytes", changing_read)
        app = portal()
        assert calls >= 2
        assert not app.exception and len(app.error) == 1
        assert "載入期間變動" in displayed(app)
        assert "V2.12新聞模型淨報酬" not in metric_values(app)
        app.radio(key="v2_research_scope").set_value("tracking").run()
        assert not app.exception and not app.error
        assert "V2.11成本後估值報酬" in metric_values(app)
        for page in ("專案總覽", "模型與結果", "研究日誌", "新聞與情緒", "V3研究", "文件與重現"):
            app.radio(key="project_page").set_value(page).run()
            assert not app.exception and not app.error, page
    # The next read uses the now-coherent source; a failed render must not leave
    # cached metrics or a permanent error behind.
    app.radio(key="project_page").set_value("V2研究").run()
    app.radio(key="v2_research_scope").set_value("historical").run()
    assert not app.exception and not app.error
    assert metric_values(app)["V2.12新聞模型淨報酬"] == "9.05%"


def test_unified_portal_ignores_standalone_overlay_environment_and_restores_it(monkeypatch):
    from dashboard_v2.data_access import prospective

    fixtures = ROOT / "dashboard_v2/fixtures/display_snapshots"
    monkeypatch.setenv("TW_V211_LOCAL", str(fixtures / "v211_local"))
    monkeypatch.setenv("TW_RIDGE_ROOT", str(fixtures / "ridge"))
    monkeypatch.setenv("TW_GITHUB_REVIEW_ROOT", str(fixtures / "github_latest"))
    overlay = prospective.v211_root()
    assert overlay == fixtures / "v211_local"
    app = portal()
    assert not app.exception and not app.error
    assert app.radio(key="v2_research_scope").value == "historical"
    assert metric_values(app)["V2.12新聞模型淨報酬"] == "9.05%"
    app.radio(key="v2_research_scope").set_value("tracking").run()
    assert not app.exception and not app.error
    assert "SYNTHETIC_DEMO" not in displayed(app)
    latest = max(path.stem for path in (ROOT / "experiments/results/v2/v211/daily").glob("????-??-??.csv"))
    assert metric_values(app)["最後保存訊號日"] == latest
    app.selectbox(key="v2_archive_section").set_value("Ridge 前瞻").run()
    assert not app.exception and not app.error
    assert "DATA_UNAVAILABLE" in displayed(app)
    assert prospective.v211_root() == overlay
    app.radio(key="v2_data_mode").set_value("demo").run()
    assert not app.exception and not app.error
    assert "SYNTHETIC_DEMO" in displayed(app)
    assert prospective.v211_root() == overlay
    fresh = portal()
    assert not fresh.exception and not fresh.error
    assert fresh.radio(key="v2_data_mode").value == "saved"
    assert fresh.radio(key="v2_research_scope").value == "historical"
    assert "SYNTHETIC_DEMO" not in displayed(fresh)
    assert prospective.v211_root() == overlay
