"""The normal 555 command reuses today's 444 and publishes a matched ledger."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import daily_reports, run_model
from scripts.common import TICKERS, TZ, sessions


def setup_sources(root):
    days = pd.bdate_range("2024-01-02", periods=110)
    day = days[-1]
    prices = pd.DataFrame([{"Date": d, "ticker": t, "Close": 100.0}
                           for d in days for t in [*TICKERS, "^TWII"]])
    news = pd.DataFrame([{"Date": d, "ticker": t, "Daily_Sentiment": 0.0, "News_Count": 0,
                         "Latest_News_Timestamp": None, "Source_Status": "verified_no_matching_article",
                         "Scoring_Model": "neutral_no_news"} for d in days for t in TICKERS])
    sources = root / "data/snapshots" / str(day.date())
    sources.mkdir(parents=True)
    prices.to_csv(sources / "prices.csv", index=False)
    news.to_csv(sources / "sentiment.csv", index=False)
    market = root / "experiments/results/v2/v211/market"
    market.mkdir(parents=True)
    prices.to_csv(market / "prices.csv", index=False)
    news.to_csv(market / "sentiment.csv", index=False)
    baseline = root / "outputs/444" / str(day.date()) / "native"
    baseline.mkdir(parents=True)
    dates = days[-3:]
    protocol = {"as_of_date": str(day.date()), "protocol_version": "lstm-evaluation-v1",
                "test_target_dates": dates.strftime("%Y-%m-%d").tolist(), "price_auto_adjust": False,
                "initial_capital": 100000, "broker_fee_rate": .001425, "sell_tax_rate": .003,
                "liquidate_at_end": True, "annual_trading_days": 252}
    (baseline / "evaluation_protocol.json").write_text(json.dumps(protocol))
    predictions = pd.DataFrame([{"Date": d, "Signal_Date": days[days.get_loc(d)-1], "Ticker": t,
                                 "Previous_Close": 100., "Actual_Close": 100., "Predicted_Close": 101.,
                                 "Predicted_Return_Pct": 1., "Direction_Correct": False}
                                for d in dates for t in TICKERS])
    predictions.to_csv(baseline / "test_predictions.csv", index=False)
    target = sessions(day + pd.Timedelta(days=1), day + pd.Timedelta(days=14))[0]
    rec = pd.DataFrame([{"Ticker": t, "Signal_Date": day, "Target_Date": target, "Previous_Close": 100.,
                         "Predicted_Close": 101., "Created_At": datetime.combine(day.date(), datetime.min.time(), TZ).replace(hour=16).isoformat()}
                        for t in TICKERS])
    rec.to_csv(baseline / "next_day_recommendation.csv", index=False)
    pd.DataFrame([{"Ticker": t, "Direction_Accuracy_Pct": 0., "Return_MAE_pp": 1., "Return_RMSE_pp": 1.,
                   "Price_MAE": 1., "Price_RMSE": 1.} for t in TICKERS]).to_csv(baseline / "model_metrics.csv", index=False)
    pd.DataFrame([{"As_Of_Date": str(day.date()), "ROI_Pct": 999}]).to_csv(baseline / "strategy_metrics.csv", index=False)
    pd.DataFrame({"Date": dates, "Portfolio_Net": [1, 2, 3], "Market_Index": [1, 1, 1]}).to_csv(baseline / "backtest_nav.csv", index=False)
    (baseline.parent / "status.json").write_text(json.dumps({"status": "success", "completed_session": str(day.date()),
                                                            "daily_native_dir": baseline.relative_to(root).as_posix()}))
    return day, baseline


def test_default_555_and_reports_use_shared_core_and_next_close_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    day, baseline = setup_sources(tmp_path)
    now = datetime.combine(day.date(), datetime.min.time(), TZ).replace(hour=17)
    monkeypatch.setattr(run_model, "ROOT", tmp_path)
    monkeypatch.setattr(daily_reports, "ROOT", tmp_path)
    monkeypatch.setattr(run_model, "run_day", lambda now=None: (now or datetime.combine(day.date(), datetime.min.time(), TZ).replace(hour=17), day, day))
    before = {p.name: p.read_bytes() for p in baseline.iterdir()}
    monkeypatch.setattr(run_model, "run_notebook", lambda *a, **k: (_ for _ in ()).throw(AssertionError("normal 555 must not train a different model")))
    assert run_model.run("555", "full") == 0
    native = tmp_path / "outputs/555" / str(day.date()) / "native"
    assert json.loads((native / "evaluation_protocol.json").read_text())["core_model"] == "444"
    assert pd.read_csv(native / "strategy_metrics.csv").ROI_Pct.iloc[0] < 0
    np.testing.assert_allclose(pd.read_csv(native / "next_day_recommendation.csv").Predicted_Return_Pct, 1., atol=1e-12)
    assert {p.name: p.read_bytes() for p in baseline.iterdir()} == before
    assert not (tmp_path / "outputs/555/recommendation_history").exists()
    assert len(list((tmp_path / "outputs/555/paired_recommendation_history").glob("*.csv"))) == 1
    summary = {"run_date": str(day.date()), "steps": {m: {"status": "success" if m in {"444", "555"} else "waiting",
                                                           "exit_code": 0 if m in {"444", "555"} else 2}
                                                       for m in daily_reports.MODELS}}
    report = daily_reports.build_reports(summary, day)
    assert report["comparison_status"] == "paired_444_core_next_close"
    comparison = pd.read_csv(tmp_path / "reports/latest/strategy_comparison.csv").set_index("Strategy_ID")
    assert comparison.loc["444", "ROI_Pct"] < 0  # 999% legacy ROI was not mixed into the main study.
    np.testing.assert_allclose(comparison.loc["555", "ROI_Pct"], comparison.loc["444", "ROI_Pct"], atol=1e-10)


def test_555_waits_for_current_444_and_legacy_remains_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    day, _ = setup_sources(tmp_path)
    monkeypatch.setattr(run_model, "ROOT", tmp_path)
    monkeypatch.setattr(run_model, "run_day", lambda now=None: (datetime.combine(day.date(), datetime.min.time(), TZ).replace(hour=17), day, day))
    (tmp_path / "outputs/444" / str(day.date()) / "status.json").unlink()
    assert run_model.run_paired_555("full") == 2
    monkeypatch.setattr(run_model, "run_notebook", lambda *a, **k: 17)
    assert run_model.run("555", "full", legacy_555=True) == 17
