"""Completed session publications survive successful retries with new evidence."""
import json

import pandas as pd
import pytest

from scripts import daily_reports, journal
from scripts.common import TICKERS


DAY = "2026-11-06"
NEXT_DAY = "2026-11-09"


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(daily_reports, "ROOT", tmp_path)
    monkeypatch.setattr(journal, "ROOT", tmp_path)
    return tmp_path


def sources(root, day=DAY, *, roi=1.0):
    """Write valid native contracts and both frozen-system forecast files."""
    for model in ("444", "555"):
        native = root / "outputs" / model / day / "native"
        native.mkdir(parents=True, exist_ok=True)
        save_json(native / "evaluation_protocol.json", {
            "protocol_version": "lstm-evaluation-v1", "as_of_date": day,
            "price_auto_adjust": False, "initial_capital": 100000,
            "broker_fee_rate": .001425, "sell_tax_rate": .003,
            "liquidate_at_end": True, "annual_trading_days": 252,
            "test_target_dates": [day], "price_panel_sha256": "shared-price-snapshot",
        })
        pd.DataFrame([{
            "As_Of_Date": day, "ROI_Pct": roi, "Sharpe": .25, "MDD_Pct": -4,
            "Direction_Accuracy_Pct": 55, "Test_Days": 1,
            "Test_Start": day, "Test_End": day, "Run_ID": f"{model}-{roi}",
        }]).to_csv(native / "strategy_metrics.csv", index=False)
        pd.DataFrame({
            "Date": ["2026-11-05", day], "Market_Index": [100000, 102000],
            "Portfolio_Net" if model == "444" else "NAV": [100000, 101000],
        }).to_csv(native / "backtest_nav.csv", index=False)
        pd.DataFrame([{
            "Ticker": ticker, "Price_MAE": 1, "Price_RMSE": 2,
            "Return_MAE_pp": .5, "Return_RMSE_pp": .7, "Direction_Accuracy_Pct": 55,
        } for ticker in TICKERS]).to_csv(native / "model_metrics.csv", index=False)
        pd.DataFrame([{
            "Ticker": ticker, "Signal_Date": day, "Target_Date": "2026-11-10",
            "Predicted_Return_Pct": roi, "Target_Weight": .2, "Cash_Weight": 0,
        } for ticker in TICKERS]).to_csv(native / "next_day_recommendation.csv", index=False)
        if model == "555":
            pd.DataFrame([{"Sharpe": .5, "MDD_Pct": -2}]).to_csv(
                native / "market_metrics.csv", index=False)
        save_json(native.parent / "status.json", {
            "status": "success", "completed_session": day,
            "daily_native_dir": native.relative_to(root).as_posix(),
        })
    for model in ("555_v2", "v211"):
        output = root / "outputs" / model / day
        save_json(output / "status.json", {"status": "success", "completed_session": day})
        signal = (output / "prediction.csv" if model == "555_v2" else
                  root / "experiments/results/v2/v211/daily" / f"{day}.csv")
        signal.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"Ticker": ticker, "Signal_Date": day}
                      for ticker in TICKERS]).to_csv(signal, index=False)


def run(day=DAY, *, hour=17, statuses=None):
    statuses = statuses or {model: "success" for model in daily_reports.MODELS}
    return {
        "completed_session": day, "run_date": day, "output_day": day,
        "started": f"{day}T{hour}:37:00+08:00", "force": True,
        "steps": {model: {"status": status, "exit_code": 0 if status == "success" else 2}
                  for model, status in statuses.items()},
    }


def publications(root):
    return {path.relative_to(root): path.read_bytes()
            for folder in ("reports/daily", "reports/latest", "reports/journals")
            for path in (root / folder).rglob("*") if path.is_file()}


def attempts(root):
    return {path: path.read_bytes() for path in (root / "reports/attempts").rglob("*.json")}


def test_successful_force_retry_keeps_completed_publication_and_records_new_attempt(root):
    sources(root)
    first = daily_reports.build_reports(run(), DAY)
    assert first["status"] == "complete" and first["publication_status"] == "published"
    before = publications(root)
    recorded = attempts(root)

    sources(root, roi=99)
    save_json(root / "experiments/results/v2/v211/market/news_scan_coverage.json", {
        "calendar_days": {DAY: {"articles": 7}}, "signal_sessions": {DAY: {}},
    })
    retry = daily_reports.build_reports(run(hour=21), DAY)

    assert retry["status"] == "complete" and retry["research_status"] == "complete"
    assert retry["publication_status"] == "attempt_only"
    assert retry["preserved_reason"] == "already_published_completed_session_report"
    assert publications(root) == before
    assert len(attempts(root)) == 2
    assert all(path.read_bytes() == raw for path, raw in recorded.items())
    newest = json.loads((root / retry["attempt_journal"]).read_text())
    assert newest["started"] == f"{DAY}T21:37:00+08:00"
    assert newest["report"]["news"]["articles_on_signal_calendar_day"] == 7
    assert newest["report"]["comparison_status"] == "not_recomputed_for_preserved_attempt"


def test_completed_invocation_is_idempotent_without_rewriting_its_attempt(root):
    sources(root)
    invocation = run()
    first = daily_reports.build_reports(invocation, DAY)
    before = publications(root)
    recorded = attempts(root)
    sources(root, roi=99)

    assert daily_reports.build_reports(invocation, DAY) == first
    assert publications(root) == before
    assert attempts(root) == recorded


def test_rejected_invocation_stays_immutable_when_its_sources_are_later_repaired(root):
    invocation = run()
    with pytest.raises(ValueError, match="missing its current"):
        daily_reports.build_reports(invocation, DAY)
    recorded = attempts(root)
    sources(root)

    with pytest.raises(ValueError, match="cannot be overwritten"):
        daily_reports.build_reports(invocation, DAY)
    assert attempts(root) == recorded
    assert publications(root) == {}

    completed = daily_reports.build_reports(run(hour=21), DAY)
    assert completed["publication_status"] == "published"
    assert all(path.read_bytes() == raw for path, raw in recorded.items())


def test_partial_can_complete_and_next_session_can_publish_without_changing_prior_day(root):
    sources(root)
    partial = daily_reports.build_reports(run(statuses={
        model: "success" if model == "444" else "waiting" for model in daily_reports.MODELS
    }), DAY)
    assert partial["status"] == "partial"
    assert (root / "reports/daily" / DAY / "444.md").is_file()
    assert not (root / "reports/daily" / DAY / "comparison.md").exists()
    partial_attempts = attempts(root)

    completed = daily_reports.build_reports(run(hour=21), DAY)
    assert completed["status"] == "complete" and completed["publication_status"] == "published"
    assert (root / "reports/daily" / DAY / "comparison.md").is_file()
    assert all(path.read_bytes() == raw for path, raw in partial_attempts.items())
    old_daily = {path: path.read_bytes() for path in (root / "reports/daily" / DAY).rglob("*")
                 if path.is_file()}
    old_attempts = attempts(root)

    sources(root, NEXT_DAY, roi=2)
    next_report = daily_reports.build_reports(run(NEXT_DAY), NEXT_DAY)
    assert next_report["publication_status"] == "published"
    assert next_report["as_of_date"] == NEXT_DAY
    assert next_report["last_successful_as_of_date"] == NEXT_DAY
    assert (root / "reports/daily" / NEXT_DAY / "comparison.md").is_file()
    assert all(path.read_bytes() == raw for path, raw in old_daily.items())
    assert all(path.read_bytes() == raw for path, raw in old_attempts.items())


def test_legacy_two_system_complete_does_not_freeze_unfinished_four_system_report(root):
    sources(root)
    save_json(root / "reports/daily" / DAY / "status.json", {
        "as_of_date": DAY, "status": "complete",
        "models": {model: {"current": True} for model in ("444", "555")},
    })
    report = daily_reports.build_reports(run(), DAY)
    assert report["publication_status"] == "published"
    assert report["status"] == "complete"
    assert all(report["models"][model]["current"] for model in daily_reports.MODELS)
