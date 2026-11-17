"""Per-render source selection cannot leak DEMO or private paths across sessions."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
from threading import Barrier

import pytest

from dashboard_v2.data_access import prospective as data


def test_saved_context_reads_current_published_ledger_and_missing_ridge_without_demo():
    root = data.ROOT
    published = json.loads((root / "experiments/results/v2/v211/status.json").read_text())
    registry = json.loads((root / "experiments/results/v2/v211/signal_registry.json").read_text())
    with data.display_sources(mode="saved", root=root):
        assert data.v211_root() == root / "experiments/results/v2/v211"
        assert data._github_root() == root
        status = data.v211_status_local()
        assert status["data_kind"] == "LIVE_READONLY"
        assert status["prices_through"] == published["data"]["prices_through"]
        assert status["prices_through"] >= "2026-11-17"
        assert status["matured_signal_days"] == published["scoring"]["matured_signal_days"]
        assert len(data.v211_signals_local()) == len(registry) * 5
        ridge = data.ridge_status()
        assert ridge["status"] == "DATA_UNAVAILABLE" and ridge["scorable_days"] == 0
        assert ridge["source_signals"] is ridge["source_labels"] is None
        assert ridge["official_claim_blocked"] is True
        assert data._prediction_sha([{"target_date": "2026-11-10"}]) is None


def test_explicit_modes_ignore_private_environment_and_never_mutate_it(tmp_path, monkeypatch):
    for name in ("TW_V211_LOCAL", "TW_RIDGE_ROOT", "TW_GITHUB_REVIEW_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name))
    environment = dict(os.environ)
    standalone = data.v211_root()
    with data.display_sources(mode="saved"):
        assert data.fixture_data_kind() == "LIVE_READONLY"
        assert data.v211_status_local()["data_kind"] == "LIVE_READONLY"
        assert data.ridge_status()["status"] == "DATA_UNAVAILABLE"
        assert data._github_root() == data.ROOT
    with data.display_sources(mode="demo"):
        assert data.v211_root() == data.SNAP / "v211_local"
        assert data._github_root() is None
        assert data.fixture_data_kind() == "SYNTHETIC_DEMO"
        status = data.v211_status_local()
        assert status["matured_signal_days"] == 2 and len(data.v211_signals_local()) == 40
        ridge = data.ridge_status()
        assert ridge["data_kind"] == "SYNTHETIC_DEMO" and ridge["scorable_days"] == 5
        assert data.github_research_status()["source_summary"].startswith("dashboard_v2/fixtures/")
    assert data.v211_root() == standalone
    assert dict(os.environ) == environment


def test_nested_contexts_restore_on_exception_and_invalid_mode_does_not_leak():
    outside = data.v211_root()
    with data.display_sources(mode="saved"):
        saved = data.v211_root()
        with pytest.raises(RuntimeError):
            with data.display_sources(mode="demo"):
                assert data.is_synthetic_demo_path(data.v211_root())
                raise RuntimeError("renderer failure")
        assert data.v211_root() == saved
        with pytest.raises(ValueError, match="saved or demo"):
            with data.display_sources(mode="invalid"):
                pytest.fail("invalid mode must not enter")
        assert data.v211_root() == saved
    assert data.v211_root() == outside


def test_two_simultaneous_sessions_keep_distinct_modes_and_paths():
    barrier = Barrier(2)
    current = json.loads((data.ROOT / "experiments/results/v2/v211/status.json").read_text())

    def render(mode):
        with data.display_sources(mode=mode):
            barrier.wait(timeout=5)
            first = data.v211_root()
            result = (data.fixture_data_kind(), data.v211_status_local()["matured_signal_days"],
                      data.ridge_status()["data_kind"], data._github_root())
            barrier.wait(timeout=5)
            assert data.v211_root() == first
            return first, result

    outside = data.v211_root()
    with ThreadPoolExecutor(max_workers=2) as workers:
        saved = workers.submit(render, "saved")
        demo = workers.submit(render, "demo")
        saved_path, saved_result = saved.result(timeout=15)
        demo_path, demo_result = demo.result(timeout=15)
    assert saved_path != demo_path
    assert saved_result == ("LIVE_READONLY", current["scoring"]["matured_signal_days"], "LIVE_READONLY", data.ROOT)
    assert demo_result == ("SYNTHETIC_DEMO", 2, "SYNTHETIC_DEMO", None)
    assert data.v211_root() == outside


def test_context_root_controls_all_saved_locations_instead_of_module_defaults(tmp_path):
    with data.display_sources(mode="saved", root=tmp_path):
        assert data.v211_root() == tmp_path / "experiments/results/v2/v211"
        assert data._github_root() == tmp_path
        assert data.github_research_status()["source_meta"] is None
        assert data.github_research_status()["source_summary"] is None
        assert data.sync_manifest()["status"] == "MISSING"
        assert data.ridge_status()["status"] == "DATA_UNAVAILABLE"


def test_matured_label_remains_mature_next_morning_and_future_label_is_not_ready(monkeypatch):
    monkeypatch.setattr(data, "v211_matured_local", lambda: {"signal_days": ["2026-11-09"]})
    local = {"prices_through": "2026-11-17", "payload": {"scoring": {"pending_labels": []}}}
    before = data._label_maturity_gate(datetime(2026, 11, 17, 14, 59, tzinfo=data.TAIPEI), local)
    after = data._label_maturity_gate(datetime(2026, 11, 17, 15, 5, tzinfo=data.TAIPEI), local)
    morning = data._label_maturity_gate(datetime(2026, 11, 18, 9, tzinfo=data.TAIPEI), local)
    assert before["state"] == "PENDING"
    assert after["state"] == morning["state"] == "MATURED"
    assert morning["protocol_close_passed"] is True
    future = {"prices_through": "2026-11-20", "payload": {"scoring": {"pending_labels": [
        {"signal": "2026-11-09", "label_end": "2026-11-20"}]}}}
    gate = data._label_maturity_gate(datetime(2026, 11, 19, 16, tzinfo=data.TAIPEI), future)
    assert gate["state"] == "PENDING" and gate["protocol_close_passed"] is False
