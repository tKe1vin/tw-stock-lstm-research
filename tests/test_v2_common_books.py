"""Economic and calendar invariants of the pure-data V2.12 common books."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.paired_research import MARKET, TICKERS, run_paired_research
from scripts.v2_common_books import MODELS, NEW_ARMS, evaluate_common_books


def sample_data(test_days=17):
    calendar = pd.bdate_range("2024-01-02", periods=111 + test_days)
    x = np.arange(len(calendar), dtype=float)
    panel = pd.DataFrame(index=calendar)
    for j, ticker in enumerate(TICKERS):
        panel[ticker] = 100 * np.exp((j + 1) * .0004 * x + .01 * np.sin(x / (j + 2)))
    panel[MARKET] = 100 * np.exp(.0002 * x + .003 * np.sin(x))
    # Large move before the first fill is deliberately not attributable to it.
    panel.loc[calendar[111]:, list(TICKERS)] *= 2
    signals, dates = calendar[110:110 + test_days], calendar[111:111 + test_days]
    paired_predictions, news, predictions = [], [], []
    for signal, execution in zip(signals, dates):
        for ticker in TICKERS:
            paired_predictions.append({"Signal_Date": signal, "Target_Date": execution,
                                       "Ticker": ticker, "Predicted_Return_Pct": 1.0})
            news.append({"Date": signal, "ticker": ticker, "Daily_Sentiment": .4})
    for model_number, model in enumerate(MODELS):
        for split, offsets in (("validation", range(95, 101)), ("test", range(110, 110 + test_days))):
            for i in offsets:
                for j, ticker in enumerate(TICKERS):
                    label_end = calendar[i + 6] if i + 6 < len(calendar) else calendar[-1] + pd.offsets.BDay(i + 6 - len(calendar) + 1)
                    mature = i + 6 < len(calendar)
                    actual = (panel.iloc[i + 6][ticker] / panel.iloc[i + 1][ticker] - 1) * 100 if mature else np.nan
                    predictions.append({"Model": model, "Fold": 0, "Split": split,
                                        "Signal_Date": calendar[i], "Execution_Date": calendar[i + 1],
                                        "Label_End": label_end, "Label_State": "MATURED" if mature else "PENDING",
                                        "Ticker": ticker, "Predicted": float((j - 2) * (1 + model_number * .3) + (i % 3) * .15),
                                        "Actual_5d_Pct": actual})
    news = pd.DataFrame(news)
    canonical = run_paired_research(pd.DataFrame(paired_predictions), panel, news)
    return {"panel": panel, "sentiment": news, "predictions": pd.DataFrame(predictions),
            "paired_protocol": canonical["protocol"], "canonical_nav": canonical["nav"],
            "canonical_metrics": canonical["metrics"]}


@pytest.fixture(scope="module")
def source():
    return sample_data()


@pytest.fixture(scope="module")
def result(source):
    return evaluate_common_books(**source)


def test_next_close_entry_and_daily_cash_cost_identity(source, result):
    first = result["nav"].query("Arm in @NEW_ARMS").groupby("Arm").first()
    assert np.allclose(first.NAV + first.Transaction_Cost, 100_000, atol=1e-6)
    assert np.allclose(first.Holding_PnL, 0)
    for arm in NEW_ARMS:
        nav = result["nav"].query("Arm == @arm").sort_values("Date")
        previous = np.r_[100_000, nav.NAV.iloc[:-1]]
        assert np.allclose(nav.Pre_Trade_NAV, previous + nav.Holding_PnL, atol=1e-6)
        assert np.allclose(nav.NAV, nav.Pre_Trade_NAV - nav.Transaction_Cost, atol=1e-6)
        weight = result["weights"].query("Arm == @arm").groupby("Date").Market_Value.sum()
        assert np.allclose(nav.NAV, nav.Cash + weight.reindex(nav.Date).to_numpy(), atol=1e-6)
    # The 100% already completed first-close price jump does not double wealth.
    assert result["metrics"].query("Arm == 'full_equal_5'").ROI_Pct.iloc[0] < 5


def test_terminal_really_clears_shares_and_charges_all_sell_costs(result):
    final = result["nav"].query("Arm in @NEW_ARMS").groupby("Arm").tail(1)
    assert final.Terminal_Liquidation.all()
    assert not final.Rebalance.any()
    assert np.allclose(final.Cash, final.NAV)
    assert np.allclose(final.Equity_Weight, 0)
    assert (final.Sell_Notional > 0).all()
    assert np.allclose(final.Transaction_Cost, final.Sell_Notional * (.001425 + .003))
    weights = result["weights"].query("Terminal_Liquidation")
    assert weights.Shares.eq(0).all()
    assert weights.Actual_Weight.eq(0).all()
    assert weights.Requested_Weight.eq(0).all()
    fills = result["fills"].query("Terminal_Liquidation")
    assert fills.Notional_Change.lt(0).all()
    assert fills.Shares_After.eq(0).all()
    assert "Execution_Price" not in result["fills"]
    for arm in NEW_ARMS:
        nav = result["nav"].query("Arm == @arm")
        fills = result["fills"].query("Arm == @arm")
        assert fills.Transaction_Cost.sum() == pytest.approx(nav.Transaction_Cost.sum())


def test_terminal_on_scheduled_rebalance_date_never_opens_positions():
    result = evaluate_common_books(**sample_data(test_days=16))
    terminal = result["fills"].query("Terminal_Liquidation")
    assert terminal.Notional_Change.lt(0).all()
    assert terminal.Shares_After.eq(0).all()
    for arm in NEW_ARMS:
        nav = result["nav"].query("Arm == @arm")
        assert np.flatnonzero(nav.Rebalance).tolist() == [0, 5, 10]


def test_all_arms_have_same_calendar_and_source_values_are_unchanged(source, result):
    for arm, nav in result["nav"].groupby("Arm"):
        assert list(nav.Date) == list(pd.to_datetime(source["paired_protocol"]["shared_target_dates"]))
    for arm in ("static_equal444", "dynamic_news_risk555"):
        saved = source["canonical_nav"].query("Arm == @arm").sort_values("Date")
        copied = result["nav"].query("Arm == @arm").sort_values("Date")
        for column in ("NAV", "Cash", "Equity_Weight", "Transaction_Cost", "Turnover"):
            assert np.array_equal(saved[column], copied[column])
        saved_metric = source["canonical_metrics"].query("Arm == @arm").iloc[0]
        copied_metric = result["metrics"].query("Arm == @arm").iloc[0]
        for column in source["canonical_metrics"].columns:
            assert saved_metric[column] == copied_metric[column]
    assert len(result["metrics"]) == 8


def test_missing_signals_and_calendar_changes_are_rejected(source):
    broken = dict(source)
    broken["predictions"] = source["predictions"].drop(source["predictions"].query("Split == 'test'").index[0])
    with pytest.raises(ValueError, match="complete common test signal"):
        evaluate_common_books(**broken)
    broken = dict(source)
    broken["paired_protocol"] = deepcopy(source["paired_protocol"])
    broken["paired_protocol"]["shared_signal_dates"][0] = str(source["panel"].index[109].date())
    with pytest.raises(ValueError, match="next-session-close"):
        evaluate_common_books(**broken)
    broken = dict(source)
    broken["canonical_nav"] = source["canonical_nav"].iloc[1:]
    with pytest.raises(ValueError, match="common calendar"):
        evaluate_common_books(**broken)


def test_validation_tau_uses_only_fold_matured_validation_and_rejects_lookahead(source, result):
    for _, row in result["thresholds"].iterrows():
        part = source["predictions"].query("Model == @row.Model and Fold == @row.Fold and Ticker == @row.Ticker and Split == 'validation'")
        assert row.Tau == pytest.approx(np.median(np.abs(part.Predicted)))
        assert row.Latest_Validation_Label_End < row.Test_First_Signal
    broken = dict(source)
    altered = source["predictions"].copy()
    select = altered.Split.eq("validation") & altered.Signal_Date.eq(altered.query("Split == 'validation'").Signal_Date.max())
    panel = source["panel"]
    first = pd.Timestamp(source["paired_protocol"]["shared_signal_dates"][0])
    pos = panel.index.get_loc(first) - 6
    for index, row in altered[select].iterrows():
        altered.loc[index, ["Signal_Date", "Execution_Date", "Label_End", "Actual_5d_Pct"]] = [
            panel.index[pos], panel.index[pos + 1], first,
            (panel.loc[first, row.Ticker] / panel.iloc[pos + 1][row.Ticker] - 1) * 100]
    broken["predictions"] = altered
    with pytest.raises(ValueError, match="未成熟"):
        evaluate_common_books(**broken)


def test_validation_calendars_match_across_models(source):
    broken = dict(source)
    pred = source["predictions"]
    remove = pred.Model.eq("news_lstm") & pred.Split.eq("validation") & pred.Signal_Date.eq(
        pred.query("Split == 'validation'").Signal_Date.min())
    broken["predictions"] = pred[~remove]
    with pytest.raises(ValueError, match="identical validation signal calendars"):
        evaluate_common_books(**broken)


def test_later_fold_validation_cannot_change_earlier_fold_thresholds_or_fills(source):
    inputs = dict(source)
    pred = source["predictions"].copy()
    fold_boundary = source["panel"].index[119]
    pred.loc[pred.Split.eq("test") & pred.Signal_Date.ge(fold_boundary), "Fold"] = 1
    validation = pred[pred.Split.eq("validation")].copy()
    validation["Fold"] = 1
    pred = pd.concat([pred, validation], ignore_index=True)
    inputs["predictions"] = pred
    first = evaluate_common_books(**inputs)
    altered = pred.copy()
    later_validation = altered.Split.eq("validation") & altered.Fold.eq(1)
    altered.loc[later_validation, "Predicted"] *= 10
    inputs["predictions"] = altered
    second = evaluate_common_books(**inputs)
    prior = first["weights"].query("Arm == 'v2_news_lstm' and Signal_Date < @fold_boundary").reset_index(drop=True)
    prior_changed = second["weights"].query("Arm == 'v2_news_lstm' and Signal_Date < @fold_boundary").reset_index(drop=True)
    pd.testing.assert_frame_equal(prior, prior_changed)
    tau = second["thresholds"].query("Model == 'news_lstm'").pivot(index="Ticker", columns="Fold", values="Tau")
    assert np.allclose(tau[1], tau[0] * 10)
    after = second["weights"].query("Arm == 'v2_news_lstm' and Trade_Eligible and Signal_Date >= @fold_boundary")
    assert after.Fold.eq(1).all()
    assert np.allclose(after.Tau, after.Ticker.map(tau[1]))


def test_risk_comes_from_signal_day_not_execution_day(source, monkeypatch):
    import scripts.v2_common_books as books

    def controlled_risk(panel, config):
        risk = pd.DataFrame({"Market_Vol_20d": 2., "Risk_Threshold_Pct": 1., "Risk_Off": False}, index=panel.index)
        first = pd.Timestamp(source["paired_protocol"]["shared_signal_dates"][0])
        risk.loc[first, "Risk_Off"] = True
        return risk

    monkeypatch.setattr(books, "market_risk", controlled_risk)
    result = books.evaluate_common_books(**source)
    first = result["nav"].query("Arm == 'v2_news_lstm'").iloc[0]
    assert first.Risk_Off
    assert first.Equity_Weight == pytest.approx(.5)
    no_risk = result["nav"].query("Arm == 'v2_news_no_risk'").iloc[0]
    assert no_risk.Equity_Weight == pytest.approx(1)
    matched = result["nav"].query("Arm == 'v2_matched_equal'").iloc[0]
    assert matched.Equity_Weight == pytest.approx(first.Equity_Weight)


def test_target_requested_and_drifted_actual_weights_are_separate(result):
    weights = result["weights"].query("Arm == 'full_equal_5'")
    held = weights[~weights.Trade_Eligible & ~weights.Terminal_Liquidation]
    assert held.Requested_Weight.isna().all()
    assert np.allclose(held.Policy_Target_Weight, .2)
    assert not np.allclose(held.Actual_Weight, held.Policy_Target_Weight)
    rebalance = weights[weights.Trade_Eligible]
    assert np.allclose(rebalance.Requested_Weight, rebalance.Actual_Weight)
    nav = result["nav"].query("Arm == 'full_equal_5'")
    assert np.flatnonzero(nav.Rebalance).tolist() == [0, 5, 10, 15]


def test_pending_labels_are_counted_but_not_scored_or_filled(source, result):
    diagnostics = result["prediction_metrics"]
    summary = diagnostics[(diagnostics.Fold == "all") & diagnostics.Split.eq("test")]
    assert summary.Rows.eq(85).all()
    assert summary.Signal_Days.eq(17).all()
    assert summary.Matured_Rows.eq(60).all()
    assert summary.Matured_Signal_Days.eq(12).all()
    assert summary.Pending_Rows.eq(25).all()
    assert summary.Pending_Signal_Days.eq(5).all()
    assert summary.Independent_Sample_Count.isna().all()
    for model in MODELS:
        actual = source["predictions"].query("Model == @model and Split == 'test' and Label_State == 'MATURED'")
        report = summary[summary.Model.eq(model) & summary.Forecast.eq("model")].iloc[0]
        assert report.MAE_Pp == pytest.approx(np.mean(np.abs(actual.Predicted - actual.Actual_5d_Pct)))
        zero = summary[summary.Model.eq(model) & summary.Forecast.eq("zero_return_baseline")].iloc[0]
        assert zero.RMSE_Pp == pytest.approx(np.sqrt(np.mean(actual.Actual_5d_Pct ** 2)))
        assert pd.isna(zero.Direction_Accuracy_Pct)
        assert zero.Direction_Scored_Rows == 0
        assert zero.Direction_Coverage_Pct == 0
        nonzero = actual.Actual_5d_Pct.abs() > 1e-8
        assert report.Always_Up_Accuracy_Pct == pytest.approx(actual.loc[nonzero, "Actual_5d_Pct"].gt(0).mean() * 100)
    broken = dict(source)
    broken["predictions"] = source["predictions"].copy()
    pending = broken["predictions"].Label_State.eq("PENDING")
    broken["predictions"].loc[pending, "Actual_5d_Pct"] = 0
    with pytest.raises(ValueError, match="pending actuals must remain NaN"):
        evaluate_common_books(**broken)


def test_bootstrap_excludes_first_entry_and_curves_include_entry_cost(result):
    assert result["paired_effects"].N_Days.eq(16).all()
    assert result["paired_effects"].Block_Length.eq(20).all()
    assert result["paired_effects"].Usable_Draws.eq(2000).all()
    assert result["paired_effects"].Seed.eq(42).all()
    assert not result["entry_costs"].Included_In_Bootstrap.any()
    first_date = result["nav"].Date.min()
    assert result["daily_returns"].Date.min() > first_date
    for arm in NEW_ARMS:
        first = result["curves"].query("Arm == @arm").iloc[0]
        assert first.Wealth_Index < 100
        assert first.Drawdown_Pct < 0


def test_inputs_are_not_mutated_and_outputs_are_deterministic(source, result):
    copies = {name: value.copy(deep=True) if isinstance(value, pd.DataFrame) else deepcopy(value)
              for name, value in source.items()}
    second = evaluate_common_books(**source)
    for name, value in source.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(value, copies[name])
        else:
            assert value == copies[name]
    pd.testing.assert_frame_equal(second["paired_effects"], result["paired_effects"])
