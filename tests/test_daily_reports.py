import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import daily_reports, journal
from scripts.common import TICKERS

DAY = "2026-11-06"


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_reports, "ROOT", tmp_path)
    monkeypatch.setattr(journal, "ROOT", tmp_path)
    return tmp_path


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def summary(statuses=None):
    statuses = statuses or {model: "waiting" for model in daily_reports.MODELS}
    return {
        "completed_session": DAY, "run_date": DAY,
        "started": "2026-11-06T17:37:00+08:00", "market": {},
        "steps": {model: {"status": value, "exit_code": 0 if value == "success" else 2}
                  for model, value in statuses.items()},
    }


def notebook(root, model, price_hash="same", day=DAY):
    native = root / "outputs" / model / DAY / "native"
    native.mkdir(parents=True)
    protocol = {
        "protocol_version": "lstm-evaluation-v1", "as_of_date": day,
        "price_auto_adjust": False, "initial_capital": 100000.,
        "broker_fee_rate": .001425, "sell_tax_rate": .003, "liquidate_at_end": True,
        "time_step": 10, "test_days": 2, "validation_days": 60,
        "annual_trading_days": 252, "risk_free_rate": 0.,
        "direction_zero_tolerance_pct": 1e-8,
        "test_target_dates_sha256": "same", "price_panel_sha256": price_hash,
        "test_target_dates": ["2026-11-05", DAY], "actual_split_audit": [],
    }
    save_json(native / "evaluation_protocol.json", protocol)
    pd.DataFrame([{
        "ROI_Pct": 1., "Sharpe": .25, "MDD_Pct": -4., "Direction_Accuracy_Pct": 56.,
        "Test_Days": 2, "As_Of_Date": day, "Run_ID": f"{model}-run",
        "Test_Start": "2026-11-05", "Test_End": DAY,
    }]).to_csv(native / "strategy_metrics.csv", index=False)
    nav = {"Date": ["2026-11-04", "2026-11-05", DAY],
           "Market_Index": [100000, 101000, 102000],
           "Portfolio_Net" if model == "444" else "NAV": [100000, 101000, 101000]}
    pd.DataFrame(nav).to_csv(native / "backtest_nav.csv", index=False)
    pd.DataFrame([{"Sharpe": .5, "MDD_Pct": -2}]).to_csv(native / "market_metrics.csv", index=False)
    pd.DataFrame([{
        "Ticker": ticker, "Price_MAE": 1., "Price_RMSE": 2., "Return_MAE_pp": .5,
        "Return_RMSE_pp": .7, "Direction_Accuracy_Pct": 55.,
    } for ticker in TICKERS]).to_csv(native / "model_metrics.csv", index=False)
    pd.DataFrame([{
        "Ticker": ticker, "Signal_Date": day, "Target_Date": "2026-11-09",
        "Predicted_Return_Pct": 1., "Target_Weight": .2, "Cash_Weight": 0,
    } for ticker in TICKERS]).to_csv(native / "next_day_recommendation.csv", index=False)
    save_json(root / "outputs" / model / DAY / "status.json", {
        "model": model, "status": "success", "completed_session": DAY,
        "daily_native_dir": native.relative_to(root).as_posix(),
        "native_output_dir": "outputs/missing-artifact",
    })
    return native


def test_partial_preserves_previous_results_and_marks_currentness(root):
    save_json(root / "reports/latest/four_systems.json", {
        "as_of_date": "2026-11-05", "status": "complete",
        "models": {model: {"current": True} for model in daily_reports.MODELS},
    })
    previous = root / "reports/latest/555/strategy_metrics.csv"
    previous.parent.mkdir(parents=True)
    previous.write_text("prior result", encoding="utf-8")
    notebook(root, "444")
    run = summary({"444": "success", "555": "waiting", "555_v2": "waiting", "v211": "waiting"})
    report = daily_reports.build_reports(run, DAY)
    assert report["status"] == "partial"
    assert report["models"]["444"]["current"] is True
    assert report["models"]["555"]["current"] is False
    assert report["last_successful_as_of_date"] == "2026-11-05"
    assert previous.read_text() == "prior result"
    assert (root / "reports/latest/444/next_day_recommendation.csv").exists()
    assert (root / f"reports/daily/{DAY}/status.json").exists()
    metadata = json.loads((root / "reports/latest/metadata.json").read_text())
    assert metadata["status"] == "partial"
    assert metadata["model_currentness"]["555"] is False
    assert not (root / f"reports/daily/{DAY}/comparison.md").exists()


def test_legacy_two_model_complete_is_not_a_four_system_success(root):
    save_json(root / "reports/latest/metadata.json", {
        "as_of_date": "2026-11-05", "status": "complete",
        "444_source": "outputs/444/runs/old", "555_source": "outputs/555/runs/old",
    })
    report = daily_reports.build_reports(summary(), DAY)
    assert report["last_successful_as_of_date"] is None
    assert report["report_schema"] == daily_reports.REPORT_SCHEMA
    assert "最近四套系統均完成" not in (root / "reports/latest/summary.md").read_text()


def test_partial_four_system_report_retains_verified_prior_success(root):
    save_json(root / "reports/latest/four_systems.json", {
        "report_schema": daily_reports.REPORT_SCHEMA,
        "as_of_date": "2026-11-05", "status": "partial",
        "last_successful_as_of_date": "2026-11-04",
        "models": {model: {"current": False} for model in daily_reports.MODELS},
    })
    report = daily_reports.build_reports(summary(), DAY)
    assert report["last_successful_as_of_date"] == "2026-11-04"


def test_current_notebooks_compare_separately_from_frozen_research(root):
    notebook(root, "444")
    notebook(root, "555")
    v2 = root / "outputs/555_v2" / DAY / "prediction.csv"
    v2.parent.mkdir(parents=True)
    pd.DataFrame([{"Ticker": ticker, "Signal_Date": DAY} for ticker in TICKERS]).to_csv(v2, index=False)
    save_json(v2.with_name("status.json"), {
        "model": "555_v2", "status": "success", "completed_session": DAY,
    })
    signal = root / "experiments/results/v2/v211/daily" / f"{DAY}.csv"
    signal.parent.mkdir(parents=True)
    pd.DataFrame([{"Ticker": ticker, "Signal_Date": DAY} for ticker in TICKERS]).to_csv(signal, index=False)
    save_json(root / "outputs/v211" / DAY / "status.json", {
        "status": "success", "completed_session": DAY, "ledger_mode": "imported_research",
    })
    run = summary({model: "success" for model in daily_reports.MODELS})
    report = daily_reports.build_reports(run, DAY)
    assert report["status"] == "complete"
    assert report["performance_protocols_combined"] is False
    assert report["comparison_status"] == "matched_test_dates_and_price_panel"
    comparison = pd.read_csv(root / "reports/latest/strategy_comparison.csv")
    assert set(comparison.Strategy_ID) == {"444", "555", "^TWII benchmark"}
    assert (root / "reports/journals/444.md").exists()
    assert "../.." in (root / "reports/latest/summary.md").read_text()


def test_mismatched_price_protocol_raises_instead_of_publishing_complete(root):
    notebook(root, "444")
    notebook(root, "555", price_hash="different")
    run = summary({"444": "success", "555": "success", "555_v2": "waiting", "v211": "waiting"})
    with pytest.raises(ValueError, match="price_panel_sha256"):
        daily_reports.build_reports(run, DAY)
    assert not (root / "reports/latest/metadata.json").exists()


def test_stale_native_files_cannot_be_called_a_current_success(root):
    notebook(root, "444", day="2026-11-05")
    run = summary({"444": "success", "555": "waiting", "555_v2": "waiting", "v211": "waiting"})
    with pytest.raises(ValueError, match="without complete outputs"):
        daily_reports.build_reports(run, DAY)


def test_missing_success_status_is_not_silently_accepted(root):
    run = summary({"444": "success", "555": "waiting", "555_v2": "waiting", "v211": "waiting"})
    with pytest.raises(ValueError, match="missing its current"):
        daily_reports.build_reports(run, DAY)


@pytest.mark.parametrize("created", ["2026-11-05T17:37:00+08:00", "2026-11-06T01:40:00+08:00"])
def test_journal_scores_saved_forecasts_against_shared_live_prices(root, created):
    history = root / "outputs/444/recommendation_history/old.csv"
    history.parent.mkdir(parents=True)
    pd.DataFrame([{
        "Ticker": ticker, "Signal_Date": "2026-11-05", "Target_Date": DAY,
        "Created_At": created, "Predicted_Return_Pct": 2.,
    } for ticker in TICKERS]).to_csv(history, index=False)
    prices = root / "shared_prices.csv"
    pd.DataFrame([{"ticker": ticker, "Date": date, "Close": value}
                  for ticker in TICKERS
                  for date, value in [("2026-11-05", 100.), (DAY, 110.)]]).to_csv(prices, index=False)
    review = journal.past_444(DAY, price_path=prices)
    assert "+10.00%" in review
    assert "+2.00%" in review


def test_journal_does_not_score_later_historical_replays_as_forward_predictions(root):
    history = root / "outputs/444/recommendation_history/replay.csv"
    history.parent.mkdir(parents=True)
    pd.DataFrame([{
        "Ticker": ticker, "Signal_Date": "2026-11-04", "Target_Date": "2026-11-05",
        "Created_At": "2026-11-06T17:37:00+08:00", "Predicted_Return_Pct": 2.,
    } for ticker in TICKERS]).to_csv(history, index=False)
    # No prices are needed: the recommendation was not available on its signal date.
    review = journal.past_444(DAY, price_path=root / "missing.csv")
    assert "暫不評分" in review
