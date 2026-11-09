"""Counterexamples for research timing, calibration, and matched trade costs."""
import numpy as np
import pandas as pd
import pytest

from experiments import calendar_rules
from experiments.horizon_experiment import TICKERS
from experiments.risk_experiment import BROKER_FEE_RATE, INITIAL_CAPITAL, MARKET
from experiments.v2.books import run_book, validation_tau
from experiments.v2.signals import calibrate


def market_inputs():
    dates = pd.bdate_range("2024-01-02", periods=50)
    panel = pd.DataFrame(100.0, index=dates, columns=[*TICKERS, MARKET])
    risk = pd.DataFrame({"Risk_Off": False}, index=dates)
    sentiment = pd.DataFrame(0.0, index=dates, columns=TICKERS)
    return dates, panel, risk, sentiment


def validation_rows(dates, fold=1, positions=(0, 1), predicted=0.1):
    return pd.DataFrame([
        {"Fold": fold, "Split": "validation", "Signal_Date": dates[pos],
         "Label_End": dates[pos + 3], "Ticker": ticker, "Predicted": predicted}
        for pos in positions for ticker in TICKERS
    ])


def test_future_fold_validation_does_not_change_earlier_trades():
    dates, panel, risk, sentiment = market_inputs()
    tests = pd.DataFrame([
        {"Fold": 1, "Signal_Date": dates[pos], "Ticker": ticker,
         "Predicted": 1.0 if j == 0 else 0.0}
        for pos in (6, 7) for j, ticker in enumerate(TICKERS)
    ])
    past = validation_rows(dates)
    future = validation_rows(dates, fold=2, positions=(20, 21), predicted=10.0)
    before = run_book(panel, risk, tests, sentiment, dates[7], dates[8], 1, "threshold", past)
    after = run_book(panel, risk, tests, sentiment, dates[7], dates[8], 1, "threshold",
                     pd.concat([past, future], ignore_index=True))
    pd.testing.assert_frame_equal(before, after)
    assert before[f"Target_{TICKERS[0]}"].iloc[0] > 0.2


def test_same_fold_unmatured_validation_is_rejected():
    dates, *_ = market_inputs()
    rows = pd.concat([validation_rows(dates), validation_rows(dates, positions=(6,))])
    with pytest.raises(ValueError, match="尚未成熟"):
        validation_tau(rows, fold=1, as_of=dates[6])


def test_static_threshold_requires_explicit_pre_signal_freeze():
    dates, panel, risk, sentiment = market_inputs()
    tests = pd.DataFrame({"Signal_Date": [dates[6]] * 5, "Ticker": TICKERS, "Predicted": 0.0})
    with pytest.raises(ValueError, match="tau_frozen_at"):
        run_book(panel, risk, tests, sentiment, dates[7], dates[8], 1, "threshold", np.ones(5))
    book = run_book(panel, risk, tests, sentiment, dates[7], dates[8], 1, "threshold",
                    np.ones(5), tau_frozen_at=dates[5])
    assert book.Tau_Protocol.iloc[0] == "explicitly_frozen_before_signal"


def test_next_close_book_uses_signal_day_risk_switch():
    dates, panel, risk, sentiment = market_inputs()
    risk.loc[dates[1], "Risk_Off"] = True
    book = run_book(panel, risk, None, sentiment, dates[1], dates[3], 1, "equal")
    assert book.Signal_Date.tolist() == [dates[0], dates[1], dates[2]]
    weights = book[[f"Target_{ticker}" for ticker in TICKERS]].sum(axis=1)
    np.testing.assert_allclose(weights, [1.0, 0.5, 0.5])
    assert book.Risk_Off.tolist() == [False, True, False]
    assert not book.Trade.iloc[-1]


def test_terminal_close_liquidates_without_new_entry_or_rebalance_fee():
    dates, panel, risk, sentiment = market_inputs()
    book = run_book(panel, risk, None, sentiment, dates[1], dates[3], 1, "equal")
    assert not book.Trade.iloc[-1]
    assert book.Rebalance_Cost.iloc[-1] == 0
    assert book.Liquidation_Cost.iloc[-1] > 0


def test_verified_exchange_closures_are_not_treated_as_missing_price_sessions():
    from scripts.common import sessions

    assert pd.Timestamp("2023-01-18") not in sessions("2023-01-16", "2023-01-30")
    assert pd.Timestamp("2024-10-31") not in sessions("2024-10-28", "2024-11-04")


def test_equal_and_prediction_books_share_missing_signal_calendar_and_costs():
    dates, panel, risk, sentiment = market_inputs()
    tests = pd.DataFrame({"Fold": 1, "Signal_Date": [dates[6]] * 5,
                          "Ticker": TICKERS, "Predicted": 0.0})
    kwargs = {"signal_calendar": [dates[6]]}
    equal = run_book(panel, risk, None, sentiment, dates[6], dates[8], 1, "equal", **kwargs)
    predicted = run_book(panel, risk, tests, sentiment, dates[6], dates[8], 1,
                         "threshold", validation_rows(dates), **kwargs)
    assert equal.Trade.tolist() == predicted.Trade.tolist() == [False, True, False]
    for column in ("NAV_Reported", "Rebalance_Cost", "Liquidation_Cost", "Stock_PnL"):
        np.testing.assert_allclose(equal[column], predicted[column])


def test_v28_comparison_uses_fold_thresholds_and_does_not_backfill_global_choice(monkeypatch):
    from experiments.v2 import v28

    dates, panel, risk, sentiment = market_inputs()
    tests = pd.DataFrame([
        {"Fold": 1, "Split": "test", "Signal_Date": dates[pos],
         "Execution_Date": dates[pos + 1], "Label_End": dates[pos + 6],
         "Ticker": ticker, "Predicted": 0.0}
        for pos in (6, 7, 8) for ticker in TICKERS
    ])
    model = pd.concat([validation_rows(dates), tests], ignore_index=True)
    monkeypatch.setattr(v28, "load_sentiment", lambda *_: sentiment)
    result = v28._books(panel, risk, {"mse_b": model}, "mse_b", dates[7], dates[14]).set_index("Name")
    assert "adopted_threshold_weekly" not in result.index
    equal = result.loc["equal_weekly"]
    predicted = result.loc["mse_b_threshold_weekly"]
    for field in ("ROI_Pct", "Total_Transaction_Cost", "Rebalance_Days", "Skipped_Rebalance_Days"):
        assert predicted[field] == pytest.approx(equal[field])
    assert not bool(predicted.Global_Adoption_Backfilled)


def test_calendar_rules_fill_next_close_with_preceding_signal_risk():
    dates, panel, risk, _ = market_inputs()
    # A same-close fill would incorrectly capture the 100 -> 200 jump.
    panel.loc[dates[1]:, TICKERS] = 200.0
    risk.loc[dates[1], "Risk_Off"] = True
    predictions = pd.DataFrame([
        {"Signal_Date": dates[pos], "Date": dates[pos + 1], "Ticker": ticker,
         "Predicted_Return_Pct": 0.0, "Daily_Sentiment": 0.0}
        for pos in range(3) for ticker in TICKERS
    ])
    tau = pd.DataFrame([
        {"oos_start": dates[1], "oos_end": dates[3], "Ticker": ticker,
         "Tau_Pct": 0.1, "Validation_End": dates[0]}
        for ticker in TICKERS
    ])
    book = calendar_rules.run_book(predictions, panel, risk, "B", tau, gap=1, liquidate=False)
    assert book.Date.tolist() == list(dates[1:4])
    assert book.Execution.eq("next_session_close").all()
    assert book.NAV_Reported.iloc[0] == pytest.approx(INITIAL_CAPITAL / (1 + BROKER_FEE_RATE))
    weights = book[[f"Target_{ticker}" for ticker in TICKERS]].sum(axis=1)
    np.testing.assert_allclose(weights.iloc[:2], [1.0, 0.5])


def test_calibration_purges_labels_crossing_validation_holdout():
    dates, *_ = market_inputs()
    rows = pd.DataFrame([
        {"Model": "B", "Fold": 1, "Ticker": TICKERS[0], "Split": "validation",
         "Signal_Date": dates[pos], "Label_End": dates[pos + 6],
         "Predicted": float(pos + 1), "Actual_5d_Pct": float(2 * (pos + 1) + 1)}
        for pos in range(30)
    ])
    before, audit = calibrate(rows)
    changed = rows.copy()
    changed.loc[9:14, "Actual_5d_Pct"] = 1e9
    after, _ = calibrate(changed)
    np.testing.assert_allclose(before.Predicted, after.Predicted)
    assert audit.Fit_Days.iloc[0] == 9
    assert audit.Purged_Fit_Days.iloc[0] == 6
    assert pd.Timestamp(audit.Fit_Last_Label_End.iloc[0]) < dates[15]
    with pytest.raises(ValueError, match="Label_End"):
        calibrate(rows.drop(columns="Label_End"))
