"""Candidate forecasts never inherit the live ledger's pending-trade status."""
from datetime import datetime
import json

import pandas as pd
import pytest

from experiments.v2 import v211 as tracker
from experiments.v2 import v211_market as market
from scripts import model_bundle, run_model as runner


@pytest.mark.parametrize("day,generated_at,eligible,expected_status", [
    ("2026-11-19", "2026-11-19T17:00:00+08:00", True, "forecast_only"),
    ("2026-11-18", "2026-11-18T17:00:00+08:00", False, "forecast_only"),
    ("2026-11-19", "2026-11-20T14:00:00+08:00", True, "forecast_expired"),
])
def test_candidate_export_stays_forecast_only_on_a_scheduled_rebalance(
        tmp_path, monkeypatch, day, generated_at, eligible, expected_status):
    generated = datetime.fromisoformat(generated_at)
    selected = pd.Timestamp(day)
    eligibility = tracker.execution_eligibility(selected, "2026-11-06")
    assert eligibility["trade_eligible"] is eligible

    class ExportClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return generated.astimezone(tz) if tz is not None else generated.replace(tzinfo=None)

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "datetime", ExportClock)
    monkeypatch.setattr(runner, "run_day", lambda: (generated, pd.Timestamp(generated.date()), selected))
    monkeypatch.delenv("TW_LSTM_OUTPUT_DAY", raising=False)
    monkeypatch.delenv("TW_LSTM_V211_DIR", raising=False)
    weights = tmp_path / "weights"
    weights.mkdir()
    weight_path = weights / "weights.npz"
    weight_path.write_bytes(b"isolated-candidate-weights")
    (weights / "model_card.json").write_text("{}")
    digest = runner.sha256(weight_path)
    monkeypatch.setattr(runner, "EXPECTED_WEIGHTS", digest)
    monkeypatch.setattr(runner, "_frozen_card", lambda: (digest, {}))
    monkeypatch.setattr(tracker, "WEIGHT_DIR", weights)
    monkeypatch.setattr(model_bundle, "verify_bundle", lambda **kwargs: {"bundle_sha256": "fixture-bundle"})
    prices = tmp_path / "prices.csv"
    prices.write_text("Date,ticker,Close\n" + day + ",2330.TW,100\n")
    sentiment = tmp_path / "sentiment.csv"
    sentiment.write_text("Date,ticker,Daily_Sentiment\n" + day + ",2330.TW,0\n")
    monkeypatch.setattr(market, "PRICES", prices)
    monkeypatch.setattr(market, "SENTIMENT", sentiment)
    monkeypatch.setattr(market, "market_files_ready", lambda: True)
    monkeypatch.setattr(market, "prediction_gate", lambda *_args: (True, "ready"))
    monkeypatch.setattr(market, "load_market_context", lambda: ())
    rows = pd.DataFrame([
        {"Ticker": ticker, "Predicted_Return_Pct": 1.0, "Rank": rank,
         "Target_Weight": .2, "Cash_Weight": 0., "Risk_Off": False}
        for rank, ticker in enumerate(runner.NAMES, 1)
    ])
    monkeypatch.setattr(tracker, "predict_rows", lambda *_args: (rows, eligibility["forecast_entry_date"], None))
    monkeypatch.setattr(tracker, "execution_eligibility", lambda _day: eligibility.copy())

    assert runner.run_555_v2("full") == 0
    folder = tmp_path / "outputs/555_v2" / str(generated.date())
    status = json.loads((folder / "status.json").read_text())
    exported = pd.read_csv(folder / "unified_signal.csv", keep_default_na=False)
    assert not status["signal_published"] and status["research_candidate"]
    assert status["recommendation_execution_status"] == expected_status
    assert exported.execution_status.eq(expected_status).all()
    assert exported.execution_price.eq(runner.NOT_APPLICABLE).all()
    assert exported.trade_eligible.eq(eligible and expected_status != "forecast_expired").all()
    assert exported.forecast_entry_date.eq(eligibility["forecast_entry_date"]).all()
    assert exported.scheduled_rebalance_date.eq(eligibility["scheduled_rebalance_date"]).all()
