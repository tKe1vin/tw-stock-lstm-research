"""Only the archived, paired cost replay may appear as a V3 optimization result."""
import hashlib
from pathlib import Path
import shutil

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from dashboard.optimization import REPORT, load_cost_experiment
from dashboard.data import ROOT, DataIntegrityError


def test_cost_experiment_retains_all_pairs_and_frozen_legacy_results():
    result = load_cost_experiment()
    assert result["available"] and result["manifest"]["legacy_replay_verified"] is True
    frame = result["metrics"]
    for _, group in frame.groupby(["Split", "Fold"]):
        assert len(group) == 5
        assert group.Start.nunique() == group.End.nunique() == 1
        assert set(group.Allocator) == {"benchmark", "legacy", "incremental"}
        assert not group.duplicated(["Allocator", "Arm"]).any()
    old = pd.read_csv(ROOT / "reports/research_v3_3/2026-11-17/holdout_metrics.csv").set_index("Arm")
    for row in frame[(frame.Split == "holdout") & (frame.Allocator == "legacy")].itertuples():
        assert row.ROI_Pct == pytest.approx(old.loc[row.Arm, "ROI_Pct"], abs=1e-8)
    assert result["registration"]["holdout_used_for_selection"] is False
    assert result["registration"]["selected"] == "B0_benchmark"


def test_missing_experiment_is_pending_and_changed_metrics_fail_closed(tmp_path):
    assert load_cost_experiment(tmp_path) == {"available": False}
    base = tmp_path / REPORT
    shutil.copytree(ROOT / REPORT, base)
    path = base / "comparison_metrics.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DataIntegrityError, match="hash"):
        load_cost_experiment(tmp_path)


def test_display_is_read_only_and_does_not_load_private_inputs_or_models(monkeypatch):
    import requests
    import sys
    monkeypatch.setattr(requests.sessions.Session, "request", lambda *a, **k: pytest.fail("no fetch"))
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    files = list((ROOT / REPORT).iterdir())
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    at = AppTest.from_string("from dashboard.optimization import render_cost_experiment\nrender_cost_experiment()", default_timeout=30).run()
    assert not at.exception and not at.error
    assert any("12.4340%" in item.value for item in at.markdown)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
