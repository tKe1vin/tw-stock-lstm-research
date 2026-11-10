import copy
import json

import numpy as np
import pandas as pd
import pytest

from scripts import research_v3 as v3


def validation_records():
    return [{"Candidate": candidate, "Fold": fold, "Split": "validation",
             "Relative_Wealth_Excess_Pct": 0.5 if candidate.endswith("_news") else 0.2,
             "Tracking_Error_Annualized": 0.03, "Stress_Relative_Wealth_Excess_Pct": 0.1}
            for candidate in v3.load_config()["candidate_order"] for fold in range(1, 4)]


def test_validation_selection_uses_matching_family_and_registered_tie_break():
    result = v3.select_news_candidate(validation_records(), v3.load_config())
    assert result["selected"] == "ridge_news"
    assert not result["holdout_used_for_selection"]
    assert not result["prospective_deployment_allowed"]


def test_all_rejected_candidates_fall_back_without_forcing_news_allocation():
    records = validation_records()
    for row in records:
        row["Relative_Wealth_Excess_Pct"] = -0.01
    result = v3.select_news_candidate(records, v3.load_config())
    assert result["selected"] == "B0_benchmark"
    assert all(row["rejection_reasons"] for row in result["assessments"])


@pytest.mark.parametrize("bad", ["test", "prospective"])
def test_holdout_or_future_results_cannot_enter_candidate_selection(bad):
    records = validation_records()
    records[0]["Split"] = bad
    with pytest.raises(ValueError, match="Only validation"):
        v3.select_news_candidate(records, v3.load_config())


@pytest.mark.parametrize("change", ["missing_fold", "duplicate", "nonfinite"])
def test_incomplete_candidate_comparison_fails_closed(change):
    records = validation_records()
    if change == "missing_fold":
        records.pop()
    elif change == "duplicate":
        records.append(copy.copy(records[0]))
    else:
        records[0]["Relative_Wealth_Excess_Pct"] = np.nan
    with pytest.raises(ValueError):
        v3.select_news_candidate(records, v3.load_config())


def test_relative_wealth_keeps_entry_cost_and_requires_identical_dates():
    a = pd.DataFrame({"Date": ["2026-01-01", "2026-01-02"], "NAV": [980, 1000]})
    b = pd.DataFrame({"Date": a.Date, "NAV": [990, 1100]})
    assert v3.relative_metrics(a, b)["Relative_Wealth_Excess_Pct"] == pytest.approx((1000/1100-1)*100)
    with pytest.raises(ValueError, match="dates differ"):
        v3.relative_metrics(a, b.assign(Date=["2026-01-01", "2026-01-03"]))


def test_default_check_does_not_fetch_train_or_create_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert v3.check()["mode"] == "read_only_check"
    assert not list(tmp_path.iterdir())


def test_finished_hold_ends_at_last_scheduled_label(monkeypatch):
    from research.v3 import portfolio
    days = pd.bdate_range("2026-01-01", periods=90)
    seen = {}
    def fake_run(bars, predictions, signal_dates, config, end_date=None):
        seen["end"] = end_date
        return seen
    monkeypatch.setattr(portfolio, "run_backtest", fake_run)
    v3._book(pd.DataFrame({"Date": days}), pd.DataFrame(), days[10:70], v3.load_config())
    assert seen["end"] == days[71]  # last scheduled signal65 +6; not daily69 +6


def test_published_artifact_mutation_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("x=1")
    item = tmp_path / "artifact.json"
    item.write_text("{}")
    manifest = {"source_sha256": {}, "files_sha256": {"artifact.json": v3.digest(item)}}
    (tmp_path / "reproduction_manifest.json").write_text(json.dumps(manifest))
    assert v3.verify(tmp_path)["verified"]
    item.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        v3.verify(tmp_path)
