"""Interrupted exports must be retried rather than kept by a success marker."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts.cached_exports import validate_cached_export, validate_forecast_frame
from scripts import daily_reports, run_all_daily
from scripts.common import TICKERS

ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-11-20"
NATIVE = ("evaluation_protocol.json", "test_predictions.csv", "next_day_recommendation.csv",
          "model_metrics.csv", "strategy_metrics.csv", "backtest_nav.csv")


def copied_cache(root, model):
    source = ROOT / "outputs" / model / DAY
    status = json.loads((source / "status.json").read_bytes())
    names = [source / "status.json", ROOT / status["unified_csv"]]
    if model in {"444", "555"}:
        folder = ROOT / status["daily_native_dir"]
        names += [folder / name for name in NATIVE]
        if model == "555":
            names.append(folder / "market_metrics.csv")
            baseline = ROOT / "outputs/444" / DAY / "status.json"
            names += [baseline, ROOT / json.loads(baseline.read_bytes())["daily_native_dir"] / "test_predictions.csv"]
    for path in names:
        target = root / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return status, root / status["unified_csv"]


def check(root, model, status, path):
    return validate_cached_export(root, model, DAY, status, path)


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("model", ["444", "555", "555_v2"])
def test_real_completed_exports_are_valid_and_no_source_files_change(tmp_path, model):
    status, path = copied_cache(tmp_path, model)
    before = snapshot(tmp_path)
    assert check(tmp_path, model, status, path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("model", ["444", "555", "555_v2"])
def test_orchestrator_retries_damaged_success_exports_and_accepts_restored_complete_files(tmp_path, monkeypatch, model):
    status, path = copied_cache(tmp_path, model)
    monkeypatch.setattr(run_all_daily, "ROOT", tmp_path)
    assert run_all_daily._completed_model(model, DAY, "2026-11-22") is not None
    original = path.read_bytes()
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[frame.index[-1], "predicted_return_pct"] = np.nan
    frame.to_csv(path, index=False)
    assert run_all_daily._completed_model(model, DAY, "2026-11-22") is None
    assert json.loads((tmp_path / "outputs" / model / DAY / "status.json").read_bytes())["status"] == "success"
    path.write_bytes(original)
    assert run_all_daily._completed_model(model, DAY, "2026-11-22") is not None


@pytest.mark.parametrize("model", ["444", "555", "555_v2"])
def test_every_partial_write_boundary_misses_cache_then_restored_export_is_reusable(tmp_path, model):
    status, path = copied_cache(tmp_path, model)
    original = path.read_bytes()
    # Header-only, truncated rows and a partial final record are distinct crash
    # outcomes, all sharing the original successful status file.
    ends = [0, original.index(b"\n") + 1, len(original) // 3, len(original) // 2,
            original.rfind(b"\n", 0, len(original) - 1) + 8]
    for end in ends:
        path.write_bytes(original[:end])
        before = snapshot(tmp_path)
        assert not check(tmp_path, model, status, path)
        assert snapshot(tmp_path) == before
    path.write_bytes(original)
    assert check(tmp_path, model, status, path)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "", "unfinished"])
@pytest.mark.parametrize("model", ["444", "555_v2"])
def test_successful_status_cannot_hide_a_nonfinite_or_unfinished_prediction(tmp_path, model, value):
    status, path = copied_cache(tmp_path, model)
    frame = pd.read_csv(path, keep_default_na=False).astype({"predicted_return_pct": object})
    frame.loc[frame.index[-1], "predicted_return_pct"] = value
    frame.to_csv(path, index=False)
    assert not check(tmp_path, model, status, path)


def test_final_record_truncated_after_horizon_is_not_a_completed_forecast(tmp_path):
    status, path = copied_cache(tmp_path, "555")
    lines = path.read_bytes().splitlines(keepends=True)
    # The valid prediction and horizon survive, but every later allocation and
    # execution field is absent. pandas still parses this as five rows.
    lines[-1] = b",".join(lines[-1].split(b",")[:10]) + b"\n"
    path.write_bytes(b"".join(lines))
    assert not check(tmp_path, "555", status, path)


@pytest.mark.parametrize("column", ["target_weight", "cash_weight", "execution_status", "data_hash"])
def test_blank_fields_after_valid_predictions_are_incomplete_exports(tmp_path, column):
    status, path = copied_cache(tmp_path, "555")
    frame = pd.read_csv(path, keep_default_na=False).astype({column: object})
    frame.loc[frame.index[-1], column] = ""
    frame.to_csv(path, index=False)
    assert not check(tmp_path, "555", status, path)


@pytest.mark.parametrize("model", ["444", "555", "555_v2"])
@pytest.mark.parametrize("column,value", [("target_weight", -.01), ("target_weight", 1.01),
                                          ("target_weight", np.inf), ("target_weight", "unfinished"),
                                          ("cash_weight", np.nan), ("cash_weight", -.01),
                                          ("cash_weight", 1.01), ("cash_weight", np.inf)])
def test_invalid_applied_allocations_never_become_cache_hits(tmp_path, model, column, value):
    status, path = copied_cache(tmp_path, model)
    frame = pd.read_csv(path, keep_default_na=False).astype({column: object})
    frame.loc[frame.index[-1], column] = value
    frame.to_csv(path, index=False)
    assert not check(tmp_path, model, status, path)


@pytest.mark.parametrize("column,value", [("rank", 0), ("rank", np.inf), ("trade_eligible", "unknown"),
                                          ("forecast_entry_date", DAY), ("execution_price", -1),
                                          ("execution_price", np.inf)])
def test_applicable_numeric_and_execution_fields_are_validated(tmp_path, column, value):
    status, path = copied_cache(tmp_path, "555_v2")
    frame = pd.read_csv(path, keep_default_na=False).astype({column: object})
    frame.loc[frame.index[-1], column] = value
    frame.to_csv(path, index=False)
    assert not check(tmp_path, "555_v2", status, path)


@pytest.mark.parametrize("column,value", [("signal_date", "2026-11-19"), ("data_asof", "2026-11-23"),
                                           ("model_id", "another-model"), ("prediction_horizon_days", 2),
                                           ("stock_id", TICKERS[0])])
def test_wrong_schema_session_or_stock_cross_section_cannot_be_reused(tmp_path, column, value):
    status, path = copied_cache(tmp_path, "555_v2")
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[frame.index[-1], column] = value
    frame.to_csv(path, index=False)
    assert not check(tmp_path, "555_v2", status, path)


@pytest.mark.parametrize("filename", NATIVE)
def test_interrupted_native_444_files_invalidate_its_success_cache(tmp_path, filename):
    status, path = copied_cache(tmp_path, "444")
    file = tmp_path / status["daily_native_dir"] / filename
    file.write_bytes(file.read_bytes()[:20])
    assert not check(tmp_path, "444", status, path)


@pytest.mark.parametrize("filename,column", [("strategy_metrics.csv", "ROI_Pct"), ("model_metrics.csv", "Price_MAE"),
                                             ("backtest_nav.csv", "Portfolio_Net"), ("test_predictions.csv", "Predicted_Return_Pct")])
def test_structurally_valid_native_files_with_invalid_values_cannot_be_reused(tmp_path, filename, column):
    status, path = copied_cache(tmp_path, "444")
    file = tmp_path / status["daily_native_dir"] / filename
    frame = pd.read_csv(file)
    frame.loc[frame.index[-1], column] = np.inf
    frame.to_csv(file, index=False)
    assert not check(tmp_path, "444", status, path)


def test_555_cache_is_invalidated_when_444_core_changes_even_if_the_new_csv_is_parseable(tmp_path):
    status, path = copied_cache(tmp_path, "555")
    assert check(tmp_path, "555", status, path)
    baseline = json.loads((tmp_path / "outputs/444" / DAY / "status.json").read_bytes())
    core = tmp_path / baseline["daily_native_dir"] / "test_predictions.csv"
    original = core.read_bytes()
    core.write_bytes(original + b"\n")
    assert not check(tmp_path, "555", status, path)
    core.write_bytes(original)
    assert check(tmp_path, "555", status, path)


@pytest.mark.parametrize("value", [None, [], "interrupted metadata", 1])
def test_malformed_paired_core_metadata_is_a_cache_miss(tmp_path, value):
    status, path = copied_cache(tmp_path, "555")
    status["primary_research"] = value
    assert not check(tmp_path, "555", status, path)


def test_unified_native_forecast_disagreement_is_not_a_cache_hit(tmp_path):
    status, path = copied_cache(tmp_path, "444")
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, "predicted_return_pct"] += 1
    frame.to_csv(path, index=False)
    assert not check(tmp_path, "444", status, path)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, ""])
def test_report_validation_rejects_nonfinite_values_before_publication(tmp_path, native, value):
    prediction = "Predicted_Return_Pct" if native else "predicted_return_pct"
    stock = "Ticker" if native else "stock_id"
    signal = "Signal_Date" if native else "signal_date"
    frame = pd.DataFrame([{stock: ticker, signal: DAY, prediction: 0.0} for ticker in TICKERS]).astype({prediction: object})
    frame.loc[frame.index[-1], prediction] = value
    path = tmp_path / "forecast.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid"):
        daily_reports._validate_forecast(path, DAY)


def test_report_validation_requires_predictions_but_retains_valid_negative_and_zero_forecasts(tmp_path):
    frame = pd.DataFrame([{"Ticker": ticker, "Signal_Date": DAY} for ticker in TICKERS])
    with pytest.raises(ValueError):
        validate_forecast_frame(frame, DAY)
    frame["Predicted_Return_Pct"] = [-2., 0., 1., 2., 3.]
    validate_forecast_frame(frame, DAY)
    frame["Target_Date"] = DAY
    with pytest.raises(ValueError, match="after"):
        validate_forecast_frame(frame, DAY)
