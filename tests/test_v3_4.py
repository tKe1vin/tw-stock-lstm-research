"""Allocation changes retain accounting and charge real planned trading legs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.v3_3.portfolio import _settings, run_backtest as legacy
from research.v3_4.allocator import estimate_trade_cost, incremental_weights
from research.v3_4.portfolio import run_backtest

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "research/v3_3/config.json").read_text())


def synthetic():
    dates = pd.bdate_range("2024-01-02", periods=65)
    cfg = config()
    records = [{"Date": day, "Ticker": asset, "Close": 100., "Volume": 1_000_000,
                "Dividends": 0., "Stock_Splits": 0.}
               for day in dates for asset in ["0050.TW", *cfg["satellite_tickers"]]]
    signals = dates[:40]
    prediction = pd.DataFrame([{"Date": day, "Ticker": asset, "Predicted": .01}
                               for day in signals for asset in cfg["satellite_tickers"]])
    return pd.DataFrame(records), signals, prediction


def test_no_turnover_has_no_fee_and_small_sale_has_one_minimum_fee_plus_correct_tax():
    settings = _settings(config(), None)
    prices = {asset: 100. for asset in [settings.benchmark, *settings.stocks]}
    assert estimate_trade_cost({settings.benchmark: 1000}, {settings.benchmark: 1000}, prices, settings) == 0
    stock = settings.stocks[0]
    sale = estimate_trade_cost({stock: 1}, {stock: 0}, prices, settings)
    etf = estimate_trade_cost({settings.benchmark: 1}, {settings.benchmark: 0}, prices, settings)
    assert sale == pytest.approx(20 + 99.95 * .003 + .05)
    assert sale - etf == pytest.approx(99.95 * (.003 - .001))


def test_low_forecasts_keep_0050_and_exactly_reproduce_control_accounting():
    bars, signals, prediction = synthetic()
    a = run_backtest(bars, prediction, signals, config())
    b = legacy(bars, None, signals, {**config(), "allocation_mode": "benchmark"})
    pd.testing.assert_frame_equal(a["nav"], b["nav"])
    assert a["metrics"]["ROI_Pct"] == b["metrics"]["ROI_Pct"]
    assert a["metrics"]["ROI_Pct"] < 0  # Flat quotes do not erase entry/exit costs.


def test_large_forecast_keeps_policy_limits_next_session_execution_and_cash_constraints():
    bars, signals, prediction = synthetic()
    prediction["Predicted"] = 20.
    result = run_backtest(bars, prediction, signals, config())
    weights = result["weights"]
    for _, part in weights.groupby("Signal_Date"):
        assert part.Policy_Target_Weight.sum() == pytest.approx(1.)
        assert part.loc[part.Ticker.eq("0050.TW"), "Policy_Target_Weight"].iloc[0] >= .8 - 1e-12
        direct = part[part.Ticker.ne("0050.TW")].Policy_Target_Weight
        assert direct.max() <= .1 and (direct > 0).sum() <= 2
    assert pd.to_datetime(result["fills"].Execution_Date).min() > signals[0]
    assert result["nav"].Cash.min() >= -1e-8
    assert result["metrics"]["ROI_Pct"] < 0


def test_cost_gate_uses_only_signal_quotes_and_rejects_invalid_quote():
    settings = _settings(config(), None)
    prices = {asset: 100. for asset in [settings.benchmark, *settings.stocks]}
    info = pd.DataFrame({"Predicted": [20.] * 5}, index=settings.stocks)
    weights, _, reason = incremental_weights(info, {}, 1_000_000, settings, prices, {})
    assert sum(weights[asset] for asset in settings.stocks) == pytest.approx(.2)
    assert reason == "incremental_turnover_gate_passed"
    prices[settings.stocks[0]] = float("nan")
    weights, _, reason = incremental_weights(info, {}, 1_000_000, settings, prices, {})
    assert weights[settings.benchmark] == 1.
    assert "fallback" in reason


def test_default_cli_does_not_replay_or_write(monkeypatch, capsys):
    from research.v3_4 import run
    monkeypatch.setattr(run, "run_experiment", lambda *a: pytest.fail("read-only default"))
    run.main([])
    assert json.loads(capsys.readouterr().out)["read_only"] is True
