"""Economic and time-order invariants for a shared saved-prediction allocation study."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.paired_research import (
    ARMS, MARKET, TICKERS, ResearchConfig, allocation_weights, market_risk,
    paired_daily_recommendation, rebalance, run_paired_research,
)


def fixture_data():
    dates = pd.bdate_range("2024-01-02", periods=110)
    panel = pd.DataFrame(100.0, index=dates, columns=list(TICKERS) + [MARKET])
    rows, news = [], []
    for i in (100, 101, 102):
        for ticker in TICKERS:
            rows.append({"Signal_Date": dates[i], "Date": dates[i + 1], "Ticker": ticker,
                         "Predicted_Return_Pct": 1.0})
            news.append({"Date": dates[i], "ticker": ticker, "Daily_Sentiment": 0.0})
    return panel, pd.DataFrame(rows), pd.DataFrame(news)


def test_first_close_gap_is_not_earned_and_only_post_fill_move_counts():
    panel, predictions, news = fixture_data()
    panel.loc[panel.index[101]:, list(TICKERS)] = 200
    panel.loc[panel.index[102]:, list(TICKERS)] = 220
    cfg = ResearchConfig(broker_fee_rate=0, sell_tax_rate=0)
    result = run_paired_research(predictions, panel, news, config=cfg)
    baseline = result["nav"].query("Arm == 'static_equal444'")
    assert baseline.NAV.iloc[0] == pytest.approx(cfg.initial_capital)
    assert baseline.NAV.iloc[-1] == pytest.approx(cfg.initial_capital * 1.1)
    assert len(baseline) == 3
    fills = result["fills"].query("Arm == 'static_equal444'")
    assert set(fills.Execution_Date) == {panel.index[101], panel.index[103]}
    assert fills[fills.Execution_Date == panel.index[103]].Share_Change.lt(0).all()


def test_constant_prices_charge_exact_entry_and_liquidation_costs():
    panel, predictions, news = fixture_data()
    cfg = ResearchConfig()
    result = run_paired_research(predictions, panel, news, config=cfg)
    expected = cfg.initial_capital / (1 + cfg.broker_fee_rate) * (
        1 - cfg.broker_fee_rate - cfg.sell_tax_rate)
    row = result["metrics"].set_index("Arm").loc["static_equal444"]
    assert row.ROI_Pct == pytest.approx((expected / cfg.initial_capital - 1) * 100)
    assert row.Total_Transaction_Cost == pytest.approx(cfg.initial_capital - expected)
    assert set(result["nav"].groupby("Arm").size()) == {3}


def test_allocation_factors_and_caps_are_independently_controlled():
    pred = np.array([1, 1, -1, 1, 1], dtype=float)
    news = np.array([0.9, -0.9, 0.9, 0, 0], dtype=float)
    no_news = allocation_weights(pred)
    with_news = allocation_weights(pred, news, use_news=True)
    assert no_news == pytest.approx([.2, .2, 0, .2, .2])
    assert with_news == pytest.approx([.3, 0, 0, .2, .2])
    capped = allocation_weights(pred, news, use_news=True, use_risk=True, risk_off=True)
    assert capped.sum() == pytest.approx(.5)
    assert capped.max() <= .3
    assert "dynamic_no_news_risk" in ARMS


def test_matched_equal_has_the_same_gross_exposure_and_common_core():
    panel, predictions, news = fixture_data()
    news.loc[news.ticker == TICKERS[0], "Daily_Sentiment"] = .8
    result = run_paired_research(predictions, panel, news)
    table = result["weights"]
    gross = table.groupby(["Arm", "Target_Date"]).Policy_Target_Weight.sum().unstack(0)
    np.testing.assert_allclose(gross.dynamic_news_risk555, gross.risk_matched_equal_reference)
    assert table.groupby(["Target_Date", "Ticker"]).Core_Predicted_Return_Pct.nunique().eq(1).all()
    assert result["protocol"]["causal_validated"] is False
    assert result["protocol"]["news_evidence"]["news_point_in_time_verified"] is False


def test_future_prices_and_future_news_do_not_change_earlier_decisions():
    panel, predictions, news = fixture_data()
    before = run_paired_research(predictions, panel, news)
    altered = panel.copy()
    altered.loc[altered.index[103]:, :] *= 100
    altered_news = news.copy()
    altered_news.loc[altered_news.Date == panel.index[102], "Daily_Sentiment"] = -.9
    after = run_paired_research(predictions, altered, altered_news)
    columns = ["Arm", "Signal_Date", "Ticker", "Policy_Target_Weight", "Risk_Off"]
    a = before["weights"].loc[before["weights"].Signal_Date < panel.index[102], columns]
    b = after["weights"].loc[after["weights"].Signal_Date < panel.index[102], columns]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    risk_before, risk_after = market_risk(panel), market_risk(altered)
    pd.testing.assert_frame_equal(risk_before.iloc[:103], risk_after.iloc[:103])


def test_missing_news_and_skipped_session_are_rejected():
    panel, predictions, news = fixture_data()
    with pytest.raises(ValueError, match="Missing or invalid"):
        run_paired_research(predictions, panel, news.iloc[1:])
    gap = predictions.copy()
    gap.loc[gap.Date == panel.index[102], "Date"] = panel.index[104]
    with pytest.raises(ValueError, match="next observed"):
        run_paired_research(gap, panel, news)


def test_acquisition_after_decision_or_signal_day_is_rejected():
    panel, predictions, news = fixture_data()
    news["Fetched_At"] = news.Date.dt.strftime("%Y-%m-%d") + "T17:37:00+08:00"
    decisions = {day: day.tz_localize("Asia/Taipei") + pd.Timedelta(hours=17) for day in news.Date.unique()}
    with pytest.raises(ValueError, match="later than"):
        run_paired_research(predictions, panel, news, decision_at=decisions)
    news["Fetched_At"] = (news.Date + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d") + "T01:00:00+08:00"
    with pytest.raises(ValueError, match="later than"):
        run_paired_research(predictions, panel, news)


def test_self_financing_rebalance_supports_cash_and_tax():
    cfg = ResearchConfig()
    current, cash = np.array([30000., 10000., 0., 0., 0.]), 60000.
    after, remaining, ledger = rebalance(current, cash, [.1, .2, .1, 0, 0], cfg)
    assert after.sum() + remaining + ledger["Total_Cost"] == pytest.approx(100000)
    assert after / (after.sum() + remaining) == pytest.approx([.1, .2, .1, 0, 0])
    assert ledger["Sell_Notional"] > 0 and ledger["Buy_Notional"] > 0


def test_daily_555_overlay_uses_exact_444_core_and_leaves_source_untouched():
    panel, _, _ = fixture_data()
    signal = panel.index[-1]
    recommendation = pd.DataFrame({"Ticker": TICKERS, "Signal_Date": signal,
                                   "Target_Date": signal + pd.offsets.BDay(1),
                                   "Previous_Close": 100.0, "Predicted_Close": [101, 99, 102, 103, 104]})
    news = pd.DataFrame({"Date": signal, "ticker": TICKERS, "Daily_Sentiment": [.9, 0, -.9, 0, 0]})
    original = recommendation.copy(deep=True)
    result = paired_daily_recommendation(recommendation, panel, news)
    pd.testing.assert_frame_equal(recommendation, original)
    assert len(result) == len(ARMS) * len(TICKERS)
    arm = result[result.Arm == "dynamic_news_risk555"].set_index("Ticker")
    assert arm.loc[TICKERS[0], "Target_Weight"] == pytest.approx(.3)
    assert arm.loc[TICKERS[1], "Target_Weight"] == 0
    assert arm.loc[TICKERS[2], "Target_Weight"] == 0
    assert result.groupby("Ticker").Predicted_Return_Pct.nunique().eq(1).all()
    assert result.Core_Model.eq("444").all()
    assert result.attrs["provenance"]["causal_validated"] is False


def test_missing_old_warmup_is_recorded_but_current_prices_are_required():
    panel, predictions, news = fixture_data()
    panel.loc[panel.index[0], MARKET] = np.nan
    result = run_paired_research(predictions, panel, news)
    assert result["protocol"]["price_history_missing_rows"] == [str(panel.index[0].date())]
    panel.loc[panel.index[101], TICKERS[0]] = np.nan
    with pytest.raises(ValueError, match="Missing signal/execution"):
        run_paired_research(predictions, panel, news)


def test_evening_acquisition_is_allowed_before_recorded_evening_decision():
    panel, predictions, news = fixture_data()
    news["Latest_News_Timestamp"] = news.Date.dt.strftime("%Y-%m-%d") + "T12:00:00+08:00"
    news["Fetched_At"] = news.Date.dt.strftime("%Y-%m-%d") + "T17:37:00+08:00"
    decisions = {day: day.tz_localize("Asia/Taipei") + pd.Timedelta(hours=22) for day in news.Date.unique()}
    result = run_paired_research(predictions, panel, news, decision_at=decisions)
    evidence = result["protocol"]["news_evidence"]
    assert evidence["decision_time_evidence_complete"] is True
    assert evidence["aggregate_timestamp_columns_complete"] is True
    assert evidence["news_point_in_time_verified"] is False
    # With no decision evidence the same data remain exploratory, not rejected at 13:30.
    replay = run_paired_research(predictions, panel, news)
    assert replay["protocol"]["news_evidence"]["decision_time_evidence_complete"] is False


def test_terminal_mark_only_mode_does_not_open_or_close_positions():
    panel, predictions, news = fixture_data()
    predictions.loc[predictions.Signal_Date == panel.index[102], "Predicted_Return_Pct"] = -5
    cfg = ResearchConfig(liquidate_at_end=False)
    result = run_paired_research(predictions, panel, news, config=cfg)
    assert result["fills"].Execution_Date.max() < panel.index[103]
    final = result["nav"].query("Date == @panel.index[103] and Arm != 'market_price_reference'")
    assert final.Transaction_Cost.eq(0).all()
    np.testing.assert_allclose(final.Equity_Weight, 1)
    assert "without liquidation" in result["protocol"]["terminal_policy"]


def test_daily_and_historical_inherit_same_baseline_annualization():
    panel, _, _ = fixture_data()
    signal = panel.index[-1]
    recommendations = pd.DataFrame({"Ticker": TICKERS, "Signal_Date": signal,
                                    "Target_Date": signal + pd.offsets.BDay(1), "Predicted_Return_Pct": 1.0})
    news = pd.DataFrame({"Date": signal, "ticker": TICKERS, "Daily_Sentiment": 0.0})
    result = paired_daily_recommendation(recommendations, panel, news, baseline_protocol={"annual_trading_days": 365})
    assert result.attrs["provenance"]["rules"]["annual_trading_days"] == 365
