"""Cash, action, timing, and cost invariants for the 0050 research book."""
import copy

import numpy as np
import pandas as pd
import pytest

from research.v3.portfolio import DEFAULT_TICKERS, run_backtest


def fixture(periods=22):
    dates = pd.bdate_range("2025-01-02", periods=periods)
    bars = pd.DataFrame([
        {"Date": day, "Ticker": ticker, "Open": 100., "High": 100., "Low": 100.,
         "Close": 100., "Volume": 10_000_000., "Dividends": 0., "Stock_Splits": 0.,
         "Return_of_Capital": 0., "Tradable": True}
        for day in dates for ticker in ("0050.TW", *DEFAULT_TICKERS)
    ])
    forecasts = pd.DataFrame([
        {"Date": day, "Ticker": ticker, "Predicted": 10. if ticker == "2330.TW" else -10.}
        for day in dates[:-6] for ticker in DEFAULT_TICKERS
    ])
    config = {
        "benchmark": {"ticker": "0050.TW"}, "initial_capital": 1_000_000,
        "allocation_mode": "model",
        "execution": {"broker_fee_rate": 0., "minimum_broker_fee": 0.,
                      "stock_sell_tax": 0., "equity_etf_sell_tax": 0.,
                      "slippage_bps_each_side": 0.},
    }
    return bars, forecasts, dates, config


def test_next_close_gap_is_not_earned_and_signal_share_quantity_is_recorded():
    bars, forecasts, dates, config = fixture()
    bars.loc[(bars.Ticker == "2330.TW") & bars.Date.ge(dates[1]), "Close"] = 200.
    bars.loc[(bars.Ticker == "2330.TW") & bars.Date.ge(dates[2]), "Close"] = 220.
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    assert result["nav"].NAV.iloc[0] == 1_000_000
    assert result["metrics"]["Final_NAV"] == 1_010_000
    stock_fill = result["fills"].query("Ticker == '2330.TW' and Side == 'BUY'").iloc[0]
    assert stock_fill.Execution_Date == dates[1]
    assert stock_fill.Requested_Shares == 1000  # Sized at the signal's known close.
    assert stock_fill.Shares == 500  # Cannot borrow to fill an overnight gap.


def test_constant_prices_exact_integer_entry_exit_fee_and_cash_conservation():
    bars, forecasts, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    config["execution"].update(broker_fee_rate=.001425, minimum_broker_fee=20., equity_etf_sell_tax=.001)
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    fills = result["fills"]
    quantity = int(np.floor(1_000_000 / (100 * 1.001425)))
    assert fills.Shares.tolist() == [quantity, quantity]
    entry_fee = quantity * 100 * .001425
    exit_fee = quantity * 100 * .001425
    exit_tax = quantity * 100 * .001
    expected_cost = entry_fee + exit_fee + exit_tax
    assert result["metrics"]["Final_NAV"] == pytest.approx(1_000_000 - expected_cost)
    assert result["metrics"]["Total_Transaction_Cost"] == pytest.approx(expected_cost)
    assert result["metrics"]["Relative_Wealth_Excess_Pct"] == 0.
    assert (result["nav"].Cash >= -1e-8).all()
    assert result["nav"].iloc[-1].Holdings_Value == 0.


def test_split_changes_shares_without_creating_profit_and_returned_capital_is_counted_once():
    bars, forecasts, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 25.
    bars.loc[affected & bars.Date.eq(dates[3]), "Stock_Splits"] = 4.
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    assert result["metrics"]["Final_NAV"] == 1_000_000
    assert result["fills"].Shares.tolist() == [10_000, 40_000]

    # A 40% remaining share count plus cash return must not be mistaken for
    # a price loss or counted as both ordinary dividend and returned capital.
    bars, forecasts, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 235.
    bars.loc[affected & bars.Date.eq(dates[3]), "Stock_Splits"] = .4
    bars.loc[affected & bars.Date.eq(dates[3]), "Return_of_Capital"] = 6.
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    assert result["metrics"]["Final_NAV"] == 1_000_000
    assert result["nav"].Return_of_Capital_Cash.sum() == 60_000


def test_dividend_is_receivable_until_payment_and_cannot_finance_orders_early():
    bars, forecasts, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars["Dividend_Payment_Date"] = pd.NaT
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 90.
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividends"] = 10.
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividend_Payment_Date"] = dates[9]
    result = run_backtest(bars, forecasts, dates[:11], config, dates[16])
    nav = result["nav"].set_index("Date")
    assert nav.loc[dates[3], "NAV"] == 1_000_000
    assert nav.loc[dates[3], "Cash"] == 0.
    assert nav.loc[dates[3], "Dividend_And_Capital_Receivables"] == 100_000
    assert nav.loc[dates[8], "Cash"] == 0.
    assert nav.loc[dates[9], "Cash"] == 100_000
    assert nav.loc[dates[9], "Dividend_And_Capital_Receivables"] == 0.
    assert result["metrics"]["Unknown_Payment_Date_Event_Count"] == 0
    assert result["metrics"]["Valuation_Protocol"] == "raw_price_corporate_action_receivable_ledger"


def test_unknown_payment_date_is_disclosed_and_receivables_survive_liquidation():
    bars, forecasts, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 90.
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividends"] = 10.
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    assert result["metrics"]["Unknown_Payment_Date_Event_Count"] == 1
    assert result["metrics"]["Valuation_Protocol"] == "retrospective_ex_date_cash_wealth_proxy"
    assert len(result["protocol"]["unknown_payment_events"]) == 1
    bars["Dividend_Payment_Date"] = pd.NaT
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividend_Payment_Date"] = dates[15]
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    assert result["metrics"]["Terminal_Receivables"] == 100_000
    assert result["metrics"]["Final_NAV"] == 1_000_000
    assert result["nav"].Cash.iloc[-1] == 900_000


def test_cost_gate_includes_both_etf_switch_legs_and_slippage():
    bars, forecasts, dates, config = fixture()
    config["execution"].update(broker_fee_rate=.001425, minimum_broker_fee=20.,
                               stock_sell_tax=.003, equity_etf_sell_tax=.001,
                               slippage_bps_each_side=5.)
    forecasts["Predicted"] = 1.7  # Below 1.5*(4*.1425+.3+.1+4*.05)=1.755 pp.
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    assert set(result["fills"].Ticker) == {"0050.TW"}
    assert result["weights"].Minimum_Net_Edge_Pct.iloc[0] == pytest.approx(1.755)
    forecasts["Predicted"] = 2.
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    active = result["weights"].query("Ticker != '0050.TW' and Policy_Target_Weight > 0")
    assert len(active) == 2
    assert active.Policy_Target_Weight.max() == .1
    assert result["weights"].set_index("Ticker").loc["0050.TW", "Policy_Target_Weight"] == pytest.approx(.8)


def test_future_price_and_volume_do_not_change_earlier_orders_or_policy():
    bars, forecasts, dates, config = fixture()
    before = run_backtest(bars, forecasts, dates[:11], config, dates[16])
    later = bars.copy()
    later.loc[later.Date.ge(dates[7]), "Close"] *= 3
    later.loc[later.Date.ge(dates[7]), "Volume"] *= 100
    after = run_backtest(later, forecasts, dates[:11], config, dates[16])
    pd.testing.assert_frame_equal(before["weights"].query("Signal_Date < @dates[7]").reset_index(drop=True),
                                  after["weights"].query("Signal_Date < @dates[7]").reset_index(drop=True))
    pd.testing.assert_frame_equal(before["fills"].query("Execution_Date < @dates[7]").reset_index(drop=True),
                                  after["fills"].query("Execution_Date < @dates[7]").reset_index(drop=True))


def test_suspension_is_valued_explicitly_and_zero_volume_orders_expire():
    bars, forecasts, dates, config = fixture()
    # Five scheduled signals: missing signal information at day5 must not
    # postpone the rebalance anchor to day6.
    affected = bars.Ticker.eq("0050.TW") & bars.Date.eq(dates[6])
    bars.loc[affected, ["Close", "Volume", "Tradable"]] = [np.nan, 0., False]
    result = run_backtest(bars, forecasts, dates[:11], config, dates[16])
    assert "0050.TW" in result["nav"].set_index("Date").loc[dates[6], "Stale_Valuations"]
    if len(result["fills"].query("Execution_Date == @dates[6] and Ticker == '0050.TW'")):
        assert result["fills"].query("Execution_Date == @dates[6] and Ticker == '0050.TW'").Shares.eq(0).all()
    assert dates[7] not in set(result["fills"].Execution_Date)
    assert set(result["weights"].Signal_Date) == set(dates[[0, 5, 10]])
    assert result["metrics"]["Live_Execution_Proven"] is False


def test_previous_volume_limits_orders_and_does_not_carry_partial_fill_forward():
    bars, forecasts, dates, config = fixture()
    bars.loc[bars.Date.eq(dates[0]), "Volume"] = 100.
    config["allocation_mode"] = "benchmark"
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    assert result["fills"].Shares.tolist() == [1, 1]
    assert set(result["fills"].Execution_Date) == {dates[1], dates[6]}
    assert result["fills"].Status.iloc[0] == "partial"


def test_last_daily_prediction_does_not_extend_last_scheduled_five_day_holding_period():
    bars, forecasts, dates, config = fixture()
    result = run_backtest(bars, forecasts, dates[:10], config)
    assert result["metrics"]["End"] == str(dates[11].date())  # signal5 ->entry6 ->exit11.
    assert result["nav"].Date.max() < dates[15]  # signal9 was never a rebalance.
    assert set(result["weights"].Signal_Date) == set(dates[[0, 5]])


def test_missing_nonfinite_or_duplicate_inputs_are_rejected():
    bars, forecasts, dates, config = fixture()
    with pytest.raises(ValueError, match="Missing market"):
        run_backtest(bars.iloc[1:], forecasts, dates[:1], config, dates[6])
    with pytest.raises(ValueError, match="Duplicate"):
        run_backtest(pd.concat([bars, bars.iloc[:1]]), forecasts, dates[:1], config, dates[6])
    bad = bars.copy()
    bad.loc[bad.index[0], "Close"] = np.nan
    with pytest.raises(ValueError, match="invalid tradable"):
        run_backtest(bad, forecasts, dates[:1], config, dates[6])
    bad_forecasts = forecasts.copy()
    bad_forecasts.loc[0, "Predicted"] = np.nan
    with pytest.raises(ValueError, match="Non-finite predicted"):
        run_backtest(bars, bad_forecasts, dates[:1], config, dates[6])


def test_missing_prediction_returns_budget_to_benchmark_without_moving_schedule():
    bars, forecasts, dates, config = fixture()
    forecasts = forecasts[~(forecasts.Date.eq(dates[5]) & forecasts.Ticker.eq("2330.TW"))]
    result = run_backtest(bars, forecasts, dates[:11], config, dates[16])
    policies = result["weights"]
    missing = policies[policies.Signal_Date.eq(dates[5])]
    assert missing.loc[missing.Ticker.eq("0050.TW"), "Policy_Target_Weight"].iloc[0] == 1.
    assert set(policies.Signal_Date) == set(dates[[0, 5, 10]])


def test_equal_universe_control_and_model_have_disclosed_different_stock_counts():
    bars, forecasts, dates, config = fixture()
    config["allocation_mode"] = "equal"
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    assert len(result["weights"].query("Ticker != '0050.TW' and Policy_Target_Weight > 0")) == 5
    assert result["protocol"]["equal_control_has_five_stocks_model_has_at_most_two"]


def test_cost_stress_reduces_same_fixed_policy_terminal_wealth():
    bars, forecasts, dates, config = fixture()
    base = run_backtest(bars, forecasts, dates[:1], config, dates[6], stress_bps=5)
    stress = run_backtest(bars, forecasts, dates[:1], config, dates[6], stress_bps=15)
    assert stress["metrics"]["Final_NAV"] < base["metrics"]["Final_NAV"]
    assert stress["metrics"]["Slippage_Bps_Each_Side"] == 15.


def test_corporate_action_fractional_entitlement_is_not_silently_sold():
    bars, forecasts, dates, config = fixture()
    config["initial_capital"] = 1_000_100
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 250.
    bars.loc[affected & bars.Date.eq(dates[3]), "Stock_Splits"] = .4
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    assert (result["fills"].Shares % 1 == 0).all()
    assert result["metrics"]["Terminal_Fractional_Entitlement_Value"] == pytest.approx(100.)
    assert result["metrics"]["Terminal_Residual_Holdings_Value"] == pytest.approx(100.)
    assert result["metrics"]["Final_NAV"] == pytest.approx(1_000_100.)


def test_invalid_payment_date_and_unregistered_risk_budget_are_rejected():
    bars, forecasts, dates, config = fixture()
    bars["Dividend_Payment_Date"] = pd.NaT
    bars.loc[0, "Dividends"] = 1.
    bars.loc[0, "Dividend_Payment_Date"] = dates[0] - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="precedes"):
        run_backtest(bars, forecasts, dates[:1], config, dates[6])
    config["allocation"] = {"minimum_benchmark_weight": .5, "maximum_satellite_weight": .5}
    with pytest.raises(ValueError, match=">=80%"):
        run_backtest(bars, forecasts, dates[:1], config, dates[6])


def test_suspended_cash_reduction_detaches_receivable_from_stale_share_mark():
    bars, forecasts, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    halted = affected & bars.Date.isin(dates[3:5])
    bars.loc[halted, ["Close", "Volume", "Tradable"]] = [np.nan, 0., False]
    bars.loc[affected & bars.Date.ge(dates[5]), "Close"] = 235.
    bars["Return_of_Capital_Payment_Date"] = pd.NaT
    event = affected & bars.Date.eq(dates[3])
    bars.loc[event, "Stock_Splits"] = .4
    bars.loc[event, "Return_of_Capital"] = 6.
    bars.loc[event, "Return_of_Capital_Payment_Date"] = dates[9]
    result = run_backtest(bars, forecasts, dates[:1], config, dates[6])
    nav = result["nav"].set_index("Date")
    assert nav.loc[dates[3], "Dividend_And_Capital_Receivables"] == 60_000
    assert nav.loc[dates[3], "Holdings_Value"] == 940_000
    assert nav.loc[dates[3], "NAV"] == 1_000_000
    assert nav.loc[dates[4], "NAV"] == 1_000_000


def test_benchmark_holds_shares_and_only_sweeps_available_cash_without_weekly_sales():
    bars, forecasts, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars["Dividend_Payment_Date"] = pd.NaT
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 90.
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividends"] = 10.
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividend_Payment_Date"] = dates[9]
    result = run_backtest(bars, forecasts, dates[:11], config, dates[16])
    filled = result["fills"].query("Shares > 0")
    assert filled[filled.Side.eq("SELL")].Execution_Date.tolist() == [dates[16]]
    assert filled[filled.Side.eq("BUY")].Execution_Date.tolist() == [dates[1], dates[11]]
    assert result["metrics"]["Final_NAV"] == 1_000_000


def test_conflicting_config_aliases_and_malformed_eligibility_fail_closed():
    bars, forecasts, dates, config = fixture()
    config["satellite_tickers"] = list(DEFAULT_TICKERS)
    config["initial_satellite_tickers"] = list(reversed(DEFAULT_TICKERS))
    with pytest.raises(ValueError, match="Conflicting satellite"):
        run_backtest(bars, forecasts, dates[:1], config, dates[6])
    del config["initial_satellite_tickers"]
    config["execution"]["maximum_previous_volume_participation"] = .02
    config["execution"]["maximum_previous_session_volume_fraction"] = .01
    with pytest.raises(ValueError, match="Conflicting volume"):
        run_backtest(bars, forecasts, dates[:1], config, dates[6])
    del config["execution"]["maximum_previous_session_volume_fraction"]
    forecasts["Eligible"] = "False"
    with pytest.raises(ValueError, match="explicit booleans"):
        run_backtest(bars, forecasts, dates[:1], config, dates[6])
