"""Twenty-session execution, accounting and no-lookahead checks for V3.3."""
import copy

import numpy as np
import pandas as pd
import pytest

from research.v3_3.portfolio import DEFAULT_TICKERS, run_backtest


def fixture(periods=84):
    dates = pd.bdate_range("2025-01-02", periods=periods)
    bars = pd.DataFrame([
        {"Date": day, "Ticker": ticker, "Open": 100., "High": 100., "Low": 100.,
         "Close": 100., "Volume": 10_000_000., "Dividends": 0., "Stock_Splits": 0.,
         "Return_of_Capital": 0., "Tradable": True}
        for day in dates for ticker in ("0050.TW", *DEFAULT_TICKERS)
    ])
    predictions = pd.DataFrame([
        {"Date": day, "Ticker": ticker, "Predicted": 10. if ticker == "2330.TW" else -10.}
        for day in dates[:-21] for ticker in DEFAULT_TICKERS
    ])
    config = {"benchmark": {"ticker": "0050.TW"}, "initial_capital": 1_000_000,
              "hold_sessions": 20, "label_gap": 21, "allocation_mode": "model",
              "execution": {"rebalance_every_sessions": 20, "broker_fee_rate": 0.,
                            "minimum_broker_fee": 0., "stock_sell_tax": 0.,
                            "equity_etf_sell_tax": 0., "slippage_bps_each_side": 0.}}
    return bars, predictions, dates, config


def test_five_day_or_inconsistent_label_config_is_rejected():
    bars, predictions, dates, config = fixture()
    for key, value in (("hold_sessions", 5), ("label_gap", 6), ("hold_sessions", 20.5)):
        bad = copy.deepcopy(config)
        bad[key] = value
        with pytest.raises(ValueError):
            run_backtest(bars, predictions, dates[:1], bad, dates[21])
    bad = copy.deepcopy(config)
    bad["execution"]["rebalance_every_sessions"] = 5
    with pytest.raises(ValueError, match="twenty"):
        run_backtest(bars, predictions, dates[:1], bad, dates[21])


def test_last_daily_prediction_does_not_extend_monthly_holding_period():
    bars, predictions, dates, config = fixture()
    result = run_backtest(bars, predictions, dates[:31], config)
    assert result["metrics"]["Start"] == str(dates[1].date())
    assert result["metrics"]["End"] == str(dates[41].date())
    assert set(result["weights"].Signal_Date) == set(dates[[0, 20]])
    assert result["protocol"]["hold_sessions"] == 20
    assert result["protocol"]["label_gap_sessions"] == 21
    assert result["protocol"]["rebalance_every_sessions"] == 20


def test_missing_scheduled_signal_never_moves_the_full_session_calendar():
    bars, predictions, dates, config = fixture()
    dates_without_twenty = dates[:51].difference(dates[[20]])
    result = run_backtest(bars, predictions, dates_without_twenty, config, dates[61])
    assert set(result["weights"].Signal_Date) == set(dates[[0, 40]])
    assert dates[22] not in set(result["fills"].Execution_Date)


def test_next_close_gap_is_not_earned_and_integer_orders_cannot_borrow():
    bars, predictions, dates, config = fixture()
    bars.loc[bars.Ticker.eq("2330.TW") & bars.Date.ge(dates[1]), "Close"] = 200.
    bars.loc[bars.Ticker.eq("2330.TW") & bars.Date.ge(dates[2]), "Close"] = 220.
    result = run_backtest(bars, predictions, dates[:1], config, dates[21])
    assert result["nav"].NAV.iloc[0] == 1_000_000
    assert result["metrics"]["Final_NAV"] == 1_010_000
    row = result["fills"].query("Ticker == '2330.TW' and Side == 'BUY'").iloc[0]
    assert row.Requested_Shares == 1000
    assert row.Shares == 500
    assert (result["nav"].Cash >= -1e-8).all()
    assert (result["fills"].Shares % 1 == 0).all()


def test_constant_benchmark_prices_exact_full_entry_and_exit_costs():
    bars, predictions, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    config["execution"].update(broker_fee_rate=.001425, minimum_broker_fee=20., equity_etf_sell_tax=.001)
    result = run_backtest(bars, predictions, dates[:1], config, dates[21])
    quantity = int(np.floor(1_000_000 / (100 * 1.001425)))
    expected_cost = quantity * 100 * (.001425 + .001425 + .001)
    assert result["fills"].Shares.tolist() == [quantity, quantity]
    assert result["metrics"]["Final_NAV"] == pytest.approx(1_000_000 - expected_cost)
    assert result["metrics"]["Total_Transaction_Cost"] == pytest.approx(expected_cost)
    assert result["metrics"]["Relative_Wealth_Excess_Pct"] == 0.
    assert result["nav"].Holdings_Value.iloc[-1] == 0.


def test_split_and_suspended_cash_reduction_preserve_total_value():
    bars, predictions, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 25.
    bars.loc[affected & bars.Date.eq(dates[3]), "Stock_Splits"] = 4.
    split = run_backtest(bars, predictions, dates[:1], config, dates[21])
    assert split["metrics"]["Final_NAV"] == 1_000_000
    assert split["fills"].Shares.tolist() == [10_000, 40_000]

    bars, predictions, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars["Return_of_Capital_Payment_Date"] = pd.NaT
    bars.loc[affected & bars.Date.isin(dates[3:5]), ["Close", "Volume", "Tradable"]] = [np.nan, 0., False]
    bars.loc[affected & bars.Date.ge(dates[5]), "Close"] = 235.
    bars.loc[affected & bars.Date.eq(dates[3]), "Stock_Splits"] = .4
    bars.loc[affected & bars.Date.eq(dates[3]), "Return_of_Capital"] = 6.
    bars.loc[affected & bars.Date.eq(dates[3]), "Return_of_Capital_Payment_Date"] = dates[9]
    reduced = run_backtest(bars, predictions, dates[:1], config, dates[21])
    nav = reduced["nav"].set_index("Date")
    assert nav.loc[dates[3], "Holdings_Value"] == 940_000
    assert nav.loc[dates[3], "Dividend_And_Capital_Receivables"] == 60_000
    assert nav.loc[dates[3], "NAV"] == 1_000_000
    assert reduced["metrics"]["Final_NAV"] == 1_000_000


def test_dividend_cash_cannot_be_reinvested_until_payment_and_next_monthly_order():
    bars, predictions, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars["Dividend_Payment_Date"] = pd.NaT
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 90.
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividends"] = 10.
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividend_Payment_Date"] = dates[9]
    result = run_backtest(bars, predictions, dates[:31], config, dates[41])
    nav = result["nav"].set_index("Date")
    assert nav.loc[dates[3], "Dividend_And_Capital_Receivables"] == 100_000
    assert nav.loc[dates[8], "Cash"] == 0.
    assert nav.loc[dates[9], "Cash"] == 100_000
    filled = result["fills"].query("Shares > 0")
    assert filled[filled.Side.eq("BUY")].Execution_Date.tolist() == [dates[1], dates[21]]
    assert filled[filled.Side.eq("SELL")].Execution_Date.tolist() == [dates[41]]
    assert result["metrics"]["Final_NAV"] == 1_000_000


def test_terminal_receivables_and_fractional_action_rights_are_not_claimed_as_cash():
    bars, predictions, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    affected = bars.Ticker.eq("0050.TW")
    bars["Dividend_Payment_Date"] = pd.NaT
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 90.
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividends"] = 10.
    bars.loc[affected & bars.Date.eq(dates[3]), "Dividend_Payment_Date"] = dates[40]
    result = run_backtest(bars, predictions, dates[:1], config, dates[21])
    assert result["metrics"]["Final_NAV"] == 1_000_000
    assert result["metrics"]["Terminal_Receivables"] == 100_000
    assert result["nav"].Cash.iloc[-1] == 900_000

    bars, predictions, dates, config = fixture()
    config.update(initial_capital=1_000_100, allocation_mode="benchmark")
    affected = bars.Ticker.eq("0050.TW")
    bars.loc[affected & bars.Date.ge(dates[3]), "Close"] = 250.
    bars.loc[affected & bars.Date.eq(dates[3]), "Stock_Splits"] = .4
    result = run_backtest(bars, predictions, dates[:1], config, dates[21])
    assert result["metrics"]["Terminal_Fractional_Entitlement_Value"] == pytest.approx(100.)
    assert result["metrics"]["Final_NAV"] == pytest.approx(1_000_100.)


def test_monthly_cost_gate_still_prices_all_four_switch_legs():
    bars, predictions, dates, config = fixture()
    config["execution"].update(broker_fee_rate=.001425, minimum_broker_fee=20.,
                               stock_sell_tax=.003, equity_etf_sell_tax=.001,
                               slippage_bps_each_side=5.)
    predictions["Predicted"] = 1.7
    blocked = run_backtest(bars, predictions, dates[:1], config, dates[21])
    assert set(blocked["fills"].Ticker) == {"0050.TW"}
    assert blocked["weights"].Minimum_Net_Edge_Pct.iloc[0] == pytest.approx(1.755)
    predictions["Predicted"] = 2.
    allowed = run_backtest(bars, predictions, dates[:1], config, dates[21])
    active = allowed["weights"].query("Ticker != '0050.TW' and Policy_Target_Weight > 0")
    assert len(active) == 2 and active.Policy_Target_Weight.eq(.1).all()
    assert allowed["weights"].query("Ticker == '0050.TW'").Policy_Target_Weight.iloc[0] == pytest.approx(.8)


def test_future_prices_volumes_and_missing_predictions_do_not_rewrite_earlier_orders():
    bars, predictions, dates, config = fixture()
    before = run_backtest(bars, predictions, dates[:51], config, dates[61])
    changed = bars.copy()
    changed.loc[changed.Date.ge(dates[25]), "Close"] *= 3
    changed.loc[changed.Date.ge(dates[25]), "Volume"] *= 100
    after = run_backtest(changed, predictions, dates[:51], config, dates[61])
    pd.testing.assert_frame_equal(before["weights"].query("Signal_Date < @dates[25]").reset_index(drop=True),
                                  after["weights"].query("Signal_Date < @dates[25]").reset_index(drop=True))
    pd.testing.assert_frame_equal(before["fills"].query("Execution_Date < @dates[25]").reset_index(drop=True),
                                  after["fills"].query("Execution_Date < @dates[25]").reset_index(drop=True))
    missing = predictions[~(predictions.Date.eq(dates[20]) & predictions.Ticker.eq("2330.TW"))]
    result = run_backtest(bars, missing, dates[:51], config, dates[61])
    assert result["weights"].query("Signal_Date == @dates[20] and Ticker == '0050.TW'").Policy_Target_Weight.iloc[0] == 1.
    assert set(result["weights"].Signal_Date) == set(dates[[0, 20, 40]])


def test_zero_volume_partial_order_expires_and_does_not_move_the_monthly_anchor():
    bars, predictions, dates, config = fixture()
    config["allocation_mode"] = "benchmark"
    bars.loc[bars.Date.eq(dates[0]), "Volume"] = 100.
    result = run_backtest(bars, predictions, dates[:1], config, dates[21])
    assert result["fills"].Shares.tolist() == [1, 1]
    assert result["fills"].Status.iloc[0] == "partial"
    assert set(result["fills"].Execution_Date) == {dates[1], dates[21]}
    bars, predictions, dates, config = fixture()
    halted = bars.Ticker.eq("0050.TW") & bars.Date.eq(dates[21])
    bars.loc[halted, ["Close", "Volume", "Tradable"]] = [np.nan, 0., False]
    result = run_backtest(bars, predictions, dates[:51], config, dates[61])
    assert dates[22] not in set(result["fills"].Execution_Date)
    assert set(result["weights"].Signal_Date) == set(dates[[0, 20, 40]])


def test_cost_stress_same_dates_and_equal_control_disclosures_are_preserved():
    bars, predictions, dates, config = fixture()
    base = run_backtest(bars, predictions, dates[:31], config, dates[41], stress_bps=5)
    stress = run_backtest(bars, predictions, dates[:31], config, dates[41], stress_bps=15)
    assert stress["metrics"]["Final_NAV"] < base["metrics"]["Final_NAV"]
    assert list(base["nav"].Date) == list(stress["nav"].Date)
    assert base["protocol"]["unfilled_orders_carried_forward"] is False
    config["allocation_mode"] = "equal"
    equal = run_backtest(bars, predictions, dates[:1], config, dates[21])
    assert len(equal["weights"].query("Ticker != '0050.TW' and Policy_Target_Weight > 0")) == 5
    assert equal["protocol"]["equal_control_has_five_stocks_model_has_at_most_two"]


def test_malformed_bars_and_prediction_cross_sections_fail_closed():
    bars, predictions, dates, config = fixture()
    with pytest.raises(ValueError, match="Missing market"):
        run_backtest(bars.iloc[1:], predictions, dates[:1], config, dates[21])
    with pytest.raises(ValueError, match="Duplicate"):
        run_backtest(pd.concat([bars, bars.iloc[:1]]), predictions, dates[:1], config, dates[21])
    predictions.loc[0, "Predicted"] = np.nan
    with pytest.raises(ValueError, match="Non-finite predicted"):
        run_backtest(bars, predictions, dates[:1], config, dates[21])
