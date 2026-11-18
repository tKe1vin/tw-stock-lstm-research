"""The historical website verifies saved evidence without fetching or fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("altair")

from dashboard import v2_history as history


MODELS = ("price_lstm", "no_news_lstm", "news_lstm")
TICKERS = ("2330.TW", "2317.TW", "2454.TW", "2308.TW", "2412.TW")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(report, name):
    path = report / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][name] = _digest(report / name)
    path.write_text(json.dumps(manifest))


@pytest.fixture
def public_report(tmp_path):
    """A labelled test fixture, never copied into the published research tree."""
    report = tmp_path / history.REPORT_DIR
    report.mkdir(parents=True)
    calendar = pd.bdate_range("2025-11-03", periods=300)
    dates, signals = calendar[150:], calendar[149:299]
    summary = {
        "period_start": str(dates[0].date()), "period_end": str(dates[-1].date()),
        "valuation_sessions": 150, "test_signal_days": 150, "matured_signal_days": 145,
        "pending_signal_days": 5, "forecasts_per_model": 750, "fold_count": 3,
        "initial_capital": 100000, "benchmark_arm": "static_equal444",
        "model_arm": "v2_news_lstm", "matched_arm": "v2_matched_equal",
        "study_status": "historical_exploratory", "reproducible_model_pack": True,
    }
    nav_rows, curve_rows, metric_rows = [], [], []
    for arm_number, arm in enumerate(history.ARMS):
        wealth = [99.85 + i * (.01 + arm_number * .005) for i in range(150)]
        equity = [.7] * 149 + [0.]
        costs = [150.] + [0.] * 148 + [80.]
        peak, drawdowns = 100., []
        name = "TEST FIXTURE " + arm
        for date, value, weight, cost in zip(dates, wealth, equity, costs):
            peak = max(peak, value)
            drawdown = (value / peak - 1) * 100
            drawdowns.append(drawdown)
            nav_rows.append({"Date": date.date(), "Arm": arm, "Display_Name": name,
                             "NAV": value * 1000, "Cash": value * 1000 * (1 - weight),
                             "Equity_Weight": weight, "Transaction_Cost": cost})
            curve_rows.append({"Date": date.date(), "Arm": arm, "Display_Name": name,
                               "Wealth_Index": value, "Drawdown_Pct": drawdown})
        metric_rows.append({"Arm": arm, "Display_Name": name, "ROI_Pct": wealth[-1] - 100,
                            "MDD_Pct": min(drawdowns), "Sharpe": 1., "Total_Transaction_Cost": sum(costs),
                            "Average_Equity_Weight_Pct": sum(equity) / 150 * 100,
                            "Average_Cash_Weight_Pct": (1 - sum(equity) / 150) * 100,
                            "Rebalance_Days": 30, "Sessions": 150})
    metrics = pd.DataFrame(metric_rows)
    roi = metrics.set_index("Arm").ROI_Pct
    for field, arm in (("Gap_Vs_444_Pp", "static_equal444"), ("Gap_Vs_555_Pp", "dynamic_news_risk555"),
                       ("Gap_Vs_Matched_Equal_Pp", "v2_matched_equal")):
        metrics[field] = metrics.ROI_Pct - roi[arm]
    timeline_rows, threshold_rows = [], []
    for model in MODELS:
        for fold in range(3):
            first = 149 + fold * 50
            for split, positions in (("test", range(first, first + 50)), ("validation", range(first - 66, first - 6))):
                for position in positions:
                    matured = position + 6 < 300
                    timeline_rows.append({"Model": model, "Fold": fold, "Split": split,
                                          "Signal_Date": calendar[position].date(), "Execution_Date": calendar[position + 1].date(),
                                          "Label_End": (calendar[position + 6] if matured else calendar[-1] + pd.offsets.BDay(position + 6 - 299)).date(),
                                          "Label_State": "matured" if matured else "pending", "Rows": 5,
                                          "Actual_Count": 5 if matured else 0})
            for ticker in TICKERS:
                threshold_rows.append({"Model": model, "Fold": fold, "Ticker": ticker, "Tau": 1.,
                                       "Validation_Rows": 60, "Validation_Signal_Days": 60,
                                       "Latest_Validation_Label_End": calendar[first - 1].date(),
                                       "Test_First_Signal": calendar[first].date()})
    timeline = pd.DataFrame(timeline_rows)
    diagnostics = []
    groups = [(model, "all", split, part) for (model, split), part in timeline.groupby(["Model", "Split"])]
    groups += [(model, fold, split, part) for (model, fold, split), part in timeline.groupby(["Model", "Fold", "Split"])]
    for model, fold, split, part in groups:
        mature = part[part.Label_State.eq("matured")]
        pending = part[part.Label_State.eq("pending")]
        for forecast in ("model", "zero_return_baseline"):
            diagnostics.append({"Model": model, "Fold": fold, "Split": split, "Forecast": forecast,
                                "Rows": int(part.Rows.sum()), "Signal_Days": part.Signal_Date.nunique(),
                                "Matured_Rows": int(mature.Rows.sum()), "Matured_Signal_Days": mature.Signal_Date.nunique(),
                                "Pending_Rows": int(pending.Rows.sum()), "Pending_Signal_Days": pending.Signal_Date.nunique(),
                                "MAE_Pp": 1., "RMSE_Pp": 1.2,
                                "Direction_Accuracy_Pct": 50. if forecast == "model" else None,
                                "Independent_Sample_Count": None})
    tables = {
        "nav": pd.DataFrame(nav_rows), "curves": pd.DataFrame(curve_rows), "metrics": metrics,
        "prediction_metrics": pd.DataFrame(diagnostics), "prediction_timeline": timeline,
        "thresholds": pd.DataFrame(threshold_rows),
        "paired_effects": pd.DataFrame([
            {"Contrast": contrast, "N_Days": 149, "Mean_Daily_Difference_Pp": .001,
             "CI_Lower_Pp": -.01, "CI_Upper_Pp": .01}
            for contrast in ("news_vs_no_news", "risk_with_news", "selection_at_same_gross")]),
        "daily_returns": pd.concat([
            pd.DataFrame({"Date": part.Date.iloc[1:], "Arm": arm,
                          "Net_Return_Pct": (part.NAV.pct_change().iloc[1:] * 100)})
            for arm, part in pd.DataFrame(nav_rows).groupby("Arm")
        ], ignore_index=True),
        "weights": pd.DataFrame({"Fixture_Only": [True]}),
        "fills": pd.DataFrame({"Fixture_Only": [True]}),
    }
    for name, frame in tables.items():
        frame.to_csv(report / (name + ".csv"), index=False)
    (report / "summary.json").write_text(json.dumps(summary))
    (report / "protocol.json").write_text(json.dumps({
        "fixture_only": True, "prospective_validated": False,
        "shared_target_dates": [str(day.date()) for day in dates],
        "shared_signal_dates": [str(day.date()) for day in signals],
    }))
    (report / "summary.md").write_text("TEST FIXTURE — historical exploration")
    pack = tmp_path / "research/v2_common/models/2026-11-17/fold_models.zip"
    pack.parent.mkdir(parents=True)
    pack.write_bytes(b"test-only stand-in for an immutable model archive")
    manifest = {"schema": history.SCHEMA, "period_start": summary["period_start"],
                "period_end": summary["period_end"], "initial_capital": 100000,
                "files": {name: _digest(report / name) for name in history.REQUIRED_FILES},
                "models": {"archive_path": pack.relative_to(tmp_path).as_posix(), "sha256": _digest(pack)},
                "sources": {"growing_live_file.csv": "provenance only; current file may have grown"}}
    (report / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, report


def test_loader_preserves_entry_cost_common_calendar_and_closed_endpoint(public_report, monkeypatch):
    import requests

    root, _ = public_report
    def reject(*args, **kwargs):
        pytest.fail("Historical views cannot fetch or train")
    monkeypatch.setattr(requests.sessions.Session, "request", reject)
    for module in ("tensorflow", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, module, None)
    study = history.load_v2_history(root)
    assert study["mode"] == "saved" and study["is_demo"] is False
    assert study["benchmark_arm"] == "static_equal444"
    assert study["matched_arm"] == "v2_matched_equal"
    assert set(study["metrics"].Arm) == set(history.ARMS)
    assert study["curves"].groupby("Arm").first().Wealth_Index.lt(100).all()
    assert study["curves"].groupby("Arm").Date.nunique().eq(150).all()
    assert study["nav"].groupby("Arm").last().Equity_Weight.eq(0).all()
    assert study["prediction_metrics"].Independent_Sample_Count.isna().all()
    assert study["prediction_metrics"].query("Forecast == 'zero_return_baseline'").Direction_Accuracy_Pct.isna().all()


def test_missing_and_changed_report_files_fail_closed(public_report):
    root, report = public_report
    (report / "nav.csv").write_bytes((report / "nav.csv").read_bytes() + b"\n")
    with pytest.raises(history.V2HistoryError, match="hash"):
        history.load_v2_history(root)
    (report / "manifest.json").unlink()
    assert not history.history_available(root)
    with pytest.raises(history.V2HistoryError, match="缺少"):
        history.load_v2_history(root)


@pytest.mark.parametrize("target", ["report", "model"])
def test_manifest_cannot_read_outside_report_or_repository(public_report, target):
    root, report = public_report
    path = report / "manifest.json"
    manifest = json.loads(path.read_text())
    if target == "report":
        manifest["files"]["../summary.json"] = "0" * 64
    else:
        manifest["models"]["archive_path"] = "/etc/passwd"
    path.write_text(json.dumps(manifest))
    with pytest.raises(history.V2HistoryError, match="路徑不安全"):
        history.load_v2_history(root)


@pytest.mark.parametrize("mutation", ["date", "roi", "drawdown", "entry_rebase", "cash", "terminal", "cost", "pending", "maturity_count", "tau_leak", "forecast_dates", "model_hash"])
def test_rehashed_semantic_changes_remain_blocked(public_report, mutation):
    root, report = public_report
    if mutation == "model_hash":
        pack = root / "research/v2_common/models/2026-11-17/fold_models.zip"
        pack.write_bytes(pack.read_bytes() + b"changed")
        expected = "模型包hash"
    else:
        name = {"date": "curves", "roi": "metrics", "drawdown": "curves", "entry_rebase": "curves",
                "cash": "nav", "terminal": "nav", "cost": "nav", "pending": "prediction_timeline",
                "maturity_count": "prediction_metrics", "tau_leak": "thresholds", "forecast_dates": "prediction_timeline"}[mutation] + ".csv"
        frame = pd.read_csv(report / name)
        expected = ".*"
        if mutation == "date":
            frame.loc[0, "Date"] = "2025-01-01"
            expected = "共同估值日"
        elif mutation == "roi":
            frame.loc[0, "ROI_Pct"] += 1
            expected = "ROI"
        elif mutation == "drawdown":
            frame.loc[0, "Drawdown_Pct"] = 0
            expected = "回撤"
        elif mutation == "entry_rebase":
            frame.loc[0, "Wealth_Index"] = 100
            expected = "重設"
        elif mutation == "cash":
            frame.loc[0, "Cash"] += 100
            expected = "無法對帳"
        elif mutation == "terminal":
            frame.loc[149, "Equity_Weight"] = .5
            frame.loc[149, "Cash"] = frame.loc[149, "NAV"] * .5
            expected = "真正清算"
        elif mutation == "cost":
            frame.loc[1, "Transaction_Cost"] += 1
            expected = "成本"
        elif mutation == "pending":
            frame.loc[frame.Label_State.eq("pending"), "Actual_Count"] = 5
            expected = "待成熟"
        elif mutation == "maturity_count":
            frame.loc[0, "Matured_Rows"] = 500
            expected = "成熟列數"
        elif mutation == "tau_leak":
            frame.loc[0, "Latest_Validation_Label_End"] = frame.loc[0, "Test_First_Signal"]
            expected = "測試前成熟"
        elif mutation == "forecast_dates":
            for column in ("Signal_Date", "Execution_Date"):
                frame[column] = (pd.to_datetime(frame[column]) - pd.offsets.BDay(1)).dt.strftime("%Y-%m-%d")
            expected = "共同帳本"
        frame.to_csv(report / name, index=False)
        _checkpoint(report, name)
    with pytest.raises(history.V2HistoryError, match=expected):
        history.load_v2_history(root)
