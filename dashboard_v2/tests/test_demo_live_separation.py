"""DEMO fixtures must never be treated as live formal performance."""
from __future__ import annotations

from pathlib import Path

from dashboard_v2.data_access.prospective import (
    DEMO_META_PATH,
    fixture_data_kind,
    ridge_status,
    source_status,
    v211_root,
    v211_status_local,
)


def test_default_paths_are_synthetic_demo():
    assert DEMO_META_PATH.exists()
    assert "SYNTHETIC_DEMO" in DEMO_META_PATH.read_text(encoding="utf-8")
    root = v211_root()
    assert "fixtures" in root.as_posix()
    assert fixture_data_kind(root) == "SYNTHETIC_DEMO"
    status = v211_status_local()
    assert status["data_kind"] == "SYNTHETIC_DEMO"
    assert status["not_official_performance"] is True
    assert status["official_claim_blocked"] is True
    assert status["performance"] == "PENDING"
    ridge = ridge_status()
    assert ridge["data_kind"] == "SYNTHETIC_DEMO"
    assert ridge["official_claim_blocked"] is True
    assert ridge["status"] == "PENDING"


def test_live_env_overrides_fixture_root(tmp_path, monkeypatch):
    live = tmp_path / "live_v211"
    (live / "daily").mkdir(parents=True)
    (live / "status.json").write_text(
        '{"ok":true,"demo_flag":null,"model_sha256":"x","model_version":"x",'
        '"data":{"prices_through":"2026-11-16"},'
        '"scoring":{"matured_signal_days":0,"pending_predictions":0,"pending_labels":[],"metrics":null},'
        '"book":{"open_nav":100000},'
        '"signal":{"created_at":"2026-11-16T12:00:00+08:00"}}',
        encoding="utf-8",
    )
    (live / "signal_registry.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TW_V211_LOCAL", str(live))
    assert v211_root() == live
    assert fixture_data_kind(live) == "LIVE_READONLY"


def test_source_status_exposes_demo_separation():
    status = source_status()
    assert status["data_kind"] == "SYNTHETIC_DEMO"
    assert status["not_official_performance"] is True
    # Display path must stay under fixtures when live env is unset.
    assert "fixtures/display_snapshots" in status["display_root"].replace("\\", "/")
