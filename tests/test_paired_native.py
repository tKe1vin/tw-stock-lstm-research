"""The default 555 export shares model predictions while changing only allocation."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.paired_native import ARM, PROTOCOL, export_paired_native
from scripts.paired_research import TICKERS, paired_daily_recommendation, run_paired_research


def fixture(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=110)
    panel = pd.DataFrame(100.0, index=dates, columns=[*TICKERS, "^TWII"])
    predictions, news = [], []
    for i in (106, 107, 108):
        for ticker in TICKERS:
            predictions.append({"Signal_Date": dates[i], "Date": dates[i+1], "Ticker": ticker,
                                "Predicted_Return_Pct": 1., "Actual_Return_Pct": 0., "Direction_Correct": False})
            news.append({"Date": dates[i], "ticker": ticker, "Daily_Sentiment": .5})
    news.extend({"Date": dates[-1], "ticker": ticker, "Daily_Sentiment": .5} for ticker in TICKERS)
    predictions, news = pd.DataFrame(predictions), pd.DataFrame(news)
    day = str(dates[-1].date())
    baseline = tmp_path / "444/native"
    baseline.mkdir(parents=True)
    protocol = {"protocol_version": "lstm-evaluation-v1", "as_of_date": day,
                "initial_capital": 100000., "broker_fee_rate": .001425, "sell_tax_rate": .003,
                "annual_trading_days": 252, "liquidate_at_end": True,
                "test_target_dates": [str(dates[i+1].date()) for i in (106, 107, 108)]}
    (baseline / "evaluation_protocol.json").write_text(json.dumps(protocol))
    predictions.to_csv(baseline / "test_predictions.csv", index=False)
    pd.DataFrame({"Ticker": TICKERS, "Direction_Accuracy_Pct": 0.0,
                  "Return_MAE_pp": 1., "Return_RMSE_pp": 1., "Price_MAE": 1., "Price_RMSE": 1.}).to_csv(
        baseline / "model_metrics.csv", index=False)
    recommendation = pd.DataFrame({"Ticker": TICKERS, "Signal_Date": dates[-1],
                                   "Target_Date": dates[-1] + pd.offsets.BDay(1),
                                   "Previous_Close": 100., "Predicted_Return_Pct": 1.})
    result = run_paired_research(predictions, panel, news, protocol)
    daily = paired_daily_recommendation(recommendation, panel, news, protocol)
    return baseline, result, daily, day


def test_native_uses_paired_metrics_and_same_444_core_without_changing_source(tmp_path):
    baseline, result, daily, day = fixture(tmp_path)
    before = {p: p.read_bytes() for p in baseline.rglob("*") if p.is_file()}
    output = tmp_path / "555/native"
    report = export_paired_native(result, daily, baseline, output, day, f"{day}T17:37:00+08:00")
    assert before == {p: p.read_bytes() for p in before}
    assert report["training_executed"] is False
    assert report["protocol_version"] == PROTOCOL
    metric = pd.read_csv(output / "strategy_metrics.csv").iloc[0]
    expected = result["metrics"].set_index("Arm").loc[ARM]
    assert metric.ROI_Pct == pytest.approx(expected.ROI_Pct)
    assert metric.Direction_Accuracy_Pct == 0
    assert metric.Core_Model == 444
    copied = pd.read_csv(output / "test_predictions.csv")
    source = pd.read_csv(baseline / "test_predictions.csv")
    pd.testing.assert_frame_equal(copied[source.columns], source, check_dtype=False)
    assert copied.Core_Model.eq(444).all()
    rec = pd.read_csv(output / "next_day_recommendation.csv")
    assert len(rec) == len(TICKERS)
    assert rec.Arm.eq(ARM).all() and rec.Run_Mode.eq("daily").all()
    assert rec.Execution_Status.eq("research_target_only_no_order").all()
    assert rec.Execution_Price.isna().all()
    np.testing.assert_allclose(rec.Predicted_Return_Pct, 1.)
    nav = pd.read_csv(output / "backtest_nav.csv")
    assert len(nav) == int(expected.Observations) + 1
    assert nav.NAV.iloc[0] == 100000.
    market = pd.read_csv(output / "market_metrics.csv").iloc[0]
    assert market.Total_Transaction_Cost == 0
    assert "theoretical_price_index" in market.Return_Basis
    protocol = json.loads((output / "evaluation_protocol.json").read_text())
    assert protocol["protocol_version"] == PROTOCOL
    assert protocol["source_444_evaluation_protocol"]["protocol_version"] == "lstm-evaluation-v1"
    assert not list((tmp_path / "555").glob("*recommendation_history*"))


def test_later_export_is_a_historical_replay_without_rewriting_444(tmp_path):
    baseline, result, daily, day = fixture(tmp_path)
    export_paired_native(result, daily, baseline, tmp_path / "555/native", day, "2026-11-09T18:00:00+08:00")
    rec = pd.read_csv(tmp_path / "555/native/next_day_recommendation.csv")
    assert rec.Run_Mode.eq("historical_replay").all()
    assert rec.Created_At.eq("2026-11-09T18:00:00+08:00").all()


@pytest.mark.parametrize("failure", ["baseline_destination", "stale", "naive_time", "duplicate"])
def test_invalid_native_export_fails_before_creating_output(tmp_path, failure):
    baseline, result, daily, day = fixture(tmp_path)
    output = tmp_path / "555/native"
    created = f"{day}T17:37:00+08:00"
    if failure == "baseline_destination":
        output = baseline / "nested"
    elif failure == "stale":
        day = "2026-11-06"
    elif failure == "naive_time":
        created = f"{day}T17:37:00"
    else:
        daily = pd.concat([daily, daily[daily.Arm == ARM].iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        export_paired_native(result, daily, baseline, output, day, created)
    assert not output.exists()
