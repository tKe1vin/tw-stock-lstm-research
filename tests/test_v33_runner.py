import copy
import json

import numpy as np
import pandas as pd
import pytest

from research.v3_3 import run as v3


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


@pytest.mark.parametrize("change", ["missing_fold", "duplicate", "nonfinite", "missing_split", "negative_te"])
def test_incomplete_candidate_comparison_fails_closed(change):
    records = validation_records()
    if change == "missing_fold":
        records.pop()
    elif change == "duplicate":
        records.append(copy.copy(records[0]))
    elif change == "nonfinite":
        records[0]["Relative_Wealth_Excess_Pct"] = np.nan
    elif change == "missing_split":
        for record in records:
            del record["Split"]
    else:
        records[0]["Tracking_Error_Annualized"] = -0.01
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
    from research.v3_3 import portfolio
    days = pd.bdate_range("2026-01-01", periods=120)
    seen = {}
    def fake_run(bars, predictions, signal_dates, config, end_date=None):
        seen["end"] = end_date
        return seen
    monkeypatch.setattr(portfolio, "run_backtest", fake_run)
    v3._book(pd.DataFrame({"Date": days}), pd.DataFrame(), days[10:70], v3.load_config())
    assert seen["end"] == days[71]  # last scheduled signal50 +21; not daily69 +21


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


def test_private_source_snapshots_cannot_be_written_under_git_root():
    with pytest.raises(ValueError, match="outside the repository"):
        v3.train(None, None, None, v3.ROOT / "reports/restricted-data-run")


def test_existing_private_run_is_immutable_before_any_read(tmp_path):
    with pytest.raises(FileExistsError, match="immutable"):
        v3.train(None, None, None, tmp_path)


@pytest.mark.parametrize("field", ["label_gap", "hold_sessions", "validation_samples"])
def test_ignored_or_unsupported_protocol_change_is_rejected(tmp_path, field):
    config = v3.load_config()
    config[field] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        v3.load_config(path)


def test_sources_snapshot_before_preparation_and_selection_before_holdout(tmp_path, monkeypatch):
    from research.v3 import data
    from research.v3_3 import models
    from research.v3_3 import delivery

    bars_path = tmp_path / "prices.csv"
    bars_path.write_text("restricted-provider-normalized-prices")
    metadata_path = tmp_path / "metadata.json"
    metadata = {"fetched_at": "2026-11-17T00:00:00+00:00", "close_basis": "truly_raw_unadjusted",
                "raw_manifest_sha256": "a" * 64, "suspensions": [], "action_date_overrides": [],
                "point_in_time_certified": False, "official_exhaustive_action_coverage_certified": False}
    metadata_path.write_text(json.dumps(metadata))
    sentiment_path = tmp_path / "sentiment.csv"
    sentiment_path.write_text("Date,ticker,Daily_Sentiment\n2026-01-01,2330.TW,0.25\n")
    output = tmp_path / "private-run"
    days = pd.bdate_range("2025-01-01", periods=160)
    fold = {"train": days[:20], "validation": days[50:52], "test": []}
    final = {"train": days[:50], "validation": [], "test": days[110:112]}
    observed = {}
    monkeypatch.setattr(v3, "SOURCES", ())

    def load_retained(path, **kwargs):
        assert path == output / "inputs/bars.csv"
        assert path.read_text() == "restricted-provider-normalized-prices"
        # A simultaneous daily append may change the live source, but not the
        # input snapshot used by this already-started experiment.
        sentiment_path.write_text("Date,ticker,Daily_Sentiment\n2026-01-01,2330.TW,-1.0\n")
        return pd.DataFrame({"Date": days})

    def prepare(bars, news, **kwargs):
        observed["news_score"] = float(news.Daily_Sentiment.iloc[0])
        return pd.DataFrame(), pd.DataFrame(), {"eligible_dates": days[30:-21]}

    def fit(features, target, split, candidate, **kwargs):
        part = "test" if len(split["test"]) else "validation"
        if part == "test":
            assert (output / "selection.json").is_file()
            assert json.loads((output / "selection.json").read_text())["selected"] == "B0_benchmark"
        return pd.DataFrame({"Date": split[part], "Ticker": "2330.TW", "Predicted": 0.,
                             "Actual": 0., "Split": part}), {"metadata": {"candidate": candidate}, "arrays": {}}

    def book(bars, predictions, dates, config, **kwargs):
        return {"nav": pd.DataFrame({"Date": [days[60], days[61]], "NAV": [1_000_000., 1_010_000.]}),
                "fills": pd.DataFrame({"Shares": [1]}), "weights": pd.DataFrame({"Weight": [1.]}),
                "metrics": {"ROI_Pct": 1., "Start": "2026-01-01", "End": "2026-06-30"}}

    monkeypatch.setattr(data, "load_dataset", load_retained)
    monkeypatch.setattr(data, "validate_data", lambda *a, **kw: {})
    monkeypatch.setattr(models, "prepare_features", prepare)
    monkeypatch.setattr(models, "make_splits", lambda *a, **kw: {"folds": [fold, fold, fold], "final": final, "test": final["test"]})
    monkeypatch.setattr(models, "fit_predict", fit)
    monkeypatch.setattr(v3, "_book", book)
    result = v3.train(bars_path, metadata_path, sentiment_path, output)
    assert observed["news_score"] == .25
    assert result["selection"]["selected"] == "B0_benchmark"
    assert v3.verify(output)["verified"]

    public = tmp_path / "public"
    delivery.export_public(output, public)
    assert not (public / "inputs").exists()
    assert all(b"restricted-provider-normalized-prices" not in path.read_bytes()
               for path in public.rglob("*") if path.is_file())
    assert not json.loads((public / "input_provenance.json").read_text())["clean_clone_contains_all_inputs"]
