"""Pure-data V2.12 historical exploration on the canonical 444/555 calendar.

The original 444/555 NAV and metric values are copied, never reconstructed.
The six new five-session policies share one next-close, fractional-share ledger
and ``paired_research.rebalance``. This module performs no fitting, I/O or
network access; the caller owns source verification and artifact publication.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd

from experiments.v2.books import threshold_weights, validation_tau
from scripts.paired_analysis import BootstrapConfig, circular_block_indices
from scripts.paired_research import MARKET, TICKERS, ResearchConfig, market_risk, rebalance


MODELS = ("price_lstm", "no_news_lstm", "news_lstm")
CANONICAL_ARMS = ("static_equal444", "dynamic_news_risk555")
MODEL_ARMS = {name: "v2_" + name for name in MODELS}
NEW_ARMS = (*MODEL_ARMS.values(), "v2_news_no_risk", "v2_matched_equal", "full_equal_5")
DISPLAY_NAMES = {
    "static_equal444": "444 固定股數（原共同帳本）",
    "dynamic_news_risk555": "555 每日新聞風控（原共同帳本）",
    "v2_price_lstm": "V2.12 價格 LSTM＋風控",
    "v2_no_news_lstm": "V2.12 五特徵無新聞 LSTM＋風控",
    "v2_news_lstm": "V2.12 六特徵新聞 LSTM＋風控",
    "v2_news_no_risk": "V2.12 新聞 LSTM（無風控）",
    "v2_matched_equal": "V2.12 同政策總曝險等權",
    "full_equal_5": "五交易日全額等權",
}
EFFECTS = (
    ("news_vs_no_news", "v2_news_lstm", "v2_no_news_lstm"),
    ("risk_with_news", "v2_news_lstm", "v2_news_no_risk"),
    ("selection_at_same_gross", "v2_news_lstm", "v2_matched_equal"),
)
DIRECTION_ZERO_TOLERANCE_PP = 1e-8


def _frame(value, name):
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame; loading belongs to the caller")
    return value.copy(deep=True)


def _dates(values):
    return pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).tz_localize(None).normalize()


def _bools(values):
    result = values.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if result.isna().any():
        raise ValueError("Risk_Off must contain booleans")
    return result.astype(bool)


def _config(protocol):
    rules = protocol.get("rules", {})
    fields = ResearchConfig.__dataclass_fields__
    cfg = ResearchConfig(**{key: value for key, value in rules.items() if key in fields})
    expected = {"initial_capital": 100_000.0, "broker_fee_rate": .001425, "sell_tax_rate": .003,
                "liquidate_at_end": True, "annual_trading_days": 252}
    if any(getattr(cfg, key) != value for key, value in expected.items()):
        raise ValueError("V2.12 requires the fixed canonical capital, fee, tax and terminal liquidation rules")
    if cfg.risk_off_gross_cap != .5 or cfg.max_stock_weight != .3:
        raise ValueError("V2.12 threshold policy requires the fixed 50% risk cap and 30% stock cap")
    return cfg


def _calendar(panel, protocol):
    panel = _frame(panel, "panel")
    if "Date" in panel:
        panel = panel.set_index("Date")
    panel.index = _dates(panel.index)
    if panel.index.duplicated().any() or not panel.index.is_monotonic_increasing:
        raise ValueError("Panel must use a sorted, unique, complete session calendar")
    required = list(TICKERS) + [MARKET]
    if not set(required).issubset(panel):
        raise ValueError("Panel must include all five stocks and ^TWII")
    panel = panel[required].astype(float)
    if np.isinf(panel.to_numpy()).any() or (panel <= 0).any().any():
        raise ValueError("Observed prices must be finite and positive; warm-up NaNs remain missing")
    dates = _dates(protocol.get("shared_target_dates", []))
    signals = _dates(protocol.get("shared_signal_dates", []))
    if (len(dates) < 3 or len(dates) != len(signals) or dates.duplicated().any()
            or not dates.is_monotonic_increasing or signals.duplicated().any()):
        raise ValueError("Canonical protocol must specify unique common signal/valuation calendars")
    locations = panel.index.get_indexer(dates)
    signal_locations = panel.index.get_indexer(signals)
    if ((locations <= 0).any() or (signal_locations < 0).any()
            or not np.array_equal(locations, signal_locations + 1)
            or not np.array_equal(np.diff(locations), np.ones(len(dates) - 1, dtype=int))):
        raise ValueError("Canonical calendar must be consecutive sessions, with next-session-close execution")
    if not np.isfinite(panel.loc[dates.union(signals)].to_numpy()).all():
        raise ValueError("Missing signal/execution prices; calendar rows cannot be dropped or filled")
    return panel, dates, signals


def _sentiment(sentiment, signals):
    raw = _frame(sentiment, "sentiment")
    if {"Date", "ticker", "Daily_Sentiment"}.issubset(raw):
        raw["Date"] = _dates(raw.Date)
        raw = raw[raw.ticker.isin(TICKERS) & raw.Date.isin(signals)]
        if raw.duplicated(["Date", "ticker"]).any():
            raise ValueError("Duplicate signal-day sentiment")
        scores = raw.pivot(index="Date", columns="ticker", values="Daily_Sentiment")
    else:
        if "Date" in raw:
            raw = raw.set_index("Date")
        raw.index = _dates(raw.index)
        if raw.index.duplicated().any():
            raise ValueError("Duplicate signal-day sentiment")
        scores = raw
    scores = scores.reindex(index=signals, columns=list(TICKERS)).astype(float)
    if not np.isfinite(scores.to_numpy()).all() or (scores.abs() > 1).any().any():
        raise ValueError("Missing or invalid signal-day sentiment")
    return scores


def _predictions(predictions, panel, dates, signals):
    pred = _frame(predictions, "predictions")
    required = {"Model", "Fold", "Split", "Signal_Date", "Execution_Date", "Label_End",
                "Label_State", "Ticker", "Predicted", "Actual_5d_Pct"}
    if not required.issubset(pred):
        raise ValueError("Predictions require the complete model/fold/split/label schema")
    if set(pred.Model) != set(MODELS) or not set(pred.Ticker).issubset(TICKERS):
        raise ValueError("Predictions must contain exactly the three fixed model candidates and five stocks")
    pred["Split"] = pred.Split.astype(str).str.lower()
    pred["Label_State"] = pred.Label_State.astype(str).str.lower()
    if not set(pred.Split).issubset({"validation", "test"}):
        raise ValueError("Only validation and test predictions belong in this evaluation")
    for name in ("Signal_Date", "Execution_Date", "Label_End"):
        pred[name] = _dates(pred[name])
    if pred[["Model", "Fold", "Split", "Signal_Date", "Execution_Date", "Ticker"]].isna().any().any():
        raise ValueError("Missing prediction identity or execution dates")
    if pred.duplicated(["Model", "Fold", "Split", "Signal_Date", "Ticker"]).any():
        raise ValueError("Duplicate prediction model/fold/split/signal/ticker")
    pred["Predicted"] = pd.to_numeric(pred.Predicted, errors="raise")
    pred["Actual_5d_Pct"] = pd.to_numeric(pred.Actual_5d_Pct, errors="raise")
    if not np.isfinite(pred.Predicted.to_numpy()).all():
        raise ValueError("All predictions must be finite, including pending-label rows")
    test = pred[pred.Split == "test"]
    assignments = None
    for model in MODELS:
        part = test[test.Model == model]
        if len(part) != len(signals) * len(TICKERS) or part.duplicated(["Signal_Date", "Ticker"]).any():
            raise ValueError("Each model needs the complete common test signal calendar; missing signals are rejected")
        if not _dates(sorted(part.Signal_Date.unique())).equals(signals):
            raise ValueError("Model test signal calendars differ from the canonical calendar")
        for signal, day in part.groupby("Signal_Date"):
            if set(day.Ticker) != set(TICKERS) or day.Fold.nunique() != 1:
                raise ValueError("Each signal needs five stocks and one fold")
        current = part.groupby("Signal_Date").Fold.first().reindex(signals)
        if assignments is not None and not current.equals(assignments):
            raise ValueError("The three models must share identical test fold assignments")
        assignments = current
    # Folds occupy contiguous date blocks; a later fold cannot reappear earlier.
    ordered_folds = list(dict.fromkeys(assignments.tolist()))
    compressed = assignments[assignments.ne(assignments.shift())].tolist()
    if compressed != ordered_folds:
        raise ValueError("Test folds must occupy contiguous, non-overlapping signal blocks")
    if set(pred.Fold) != set(ordered_folds):
        raise ValueError("Validation folds must correspond exactly to test folds")
    locations = panel.index.get_indexer(pred.Signal_Date)
    if (locations < 0).any() or (locations + 1 >= len(panel)).any():
        raise ValueError("Prediction signal/entry dates are missing from the full session calendar")
    if not np.array_equal(panel.index[locations + 1].to_numpy(), pred.Execution_Date.to_numpy()):
        raise ValueError("Forecast entry must be the next session close")
    available = locations + 6 < len(panel)
    expected_mature = np.zeros(len(pred), dtype=bool)
    expected_mature[available] = panel.index[locations[available] + 6] <= dates[-1]
    states = pred.Label_State.to_numpy()
    if not np.array_equal(states == "matured", expected_mature) or not set(states).issubset({"matured", "pending"}):
        raise ValueError("Label maturity must be determined at the common final valuation date")
    if (pred.loc[available, "Label_End"].isna().any() or not np.array_equal(
            pred.loc[available, "Label_End"].to_numpy(), panel.index[locations[available] + 6].to_numpy())):
        raise ValueError("Five-session label end must be signal position +6 on the complete calendar")
    if (pred.loc[~expected_mature, "Label_End"].notna()
            & pred.loc[~expected_mature, "Label_End"].le(dates[-1])).any():
        raise ValueError("Pending label ends cannot be on or before the valuation cutoff")
    actual = pred.Actual_5d_Pct.to_numpy(dtype=float)
    if not np.isfinite(actual[expected_mature]).all() or not np.isnan(actual[~expected_mature]).all():
        raise ValueError("Matured actuals must be finite and pending actuals must remain NaN")
    if (pred.Split.eq("validation") & ~expected_mature).any():
        raise ValueError("Validation thresholds cannot contain pending labels")
    for position in np.flatnonzero(expected_mature):
        row = pred.iloc[position]
        start = float(panel.loc[row.Execution_Date, row.Ticker])
        end = float(panel.loc[row.Label_End, row.Ticker])
        expected = (end / start - 1) * 100
        if not np.isfinite(expected) or not np.isclose(actual[position], expected, rtol=1e-7, atol=1e-7):
            raise ValueError("Matured actuals disagree with the shared five-session price label")
    thresholds = []
    tau_map = {}
    validation_calendars = {}
    for model in MODELS:
        for fold in ordered_folds:
            first = test.loc[(test.Model == model) & (test.Fold == fold), "Signal_Date"].min()
            validation = pred[(pred.Model == model) & (pred.Fold == fold) & (pred.Split == "validation")]
            if validation.empty or validation.duplicated(["Signal_Date", "Ticker"]).any():
                raise ValueError("Every model/fold requires unique matured validation predictions")
            counts = validation.groupby("Signal_Date").Ticker.agg(set)
            if any(value != set(TICKERS) for value in counts):
                raise ValueError("Every validation signal must have all five stocks")
            calendar = _dates(sorted(validation.Signal_Date.unique()))
            if fold in validation_calendars and not calendar.equals(validation_calendars[fold]):
                raise ValueError("The three models must share identical validation signal calendars per fold")
            validation_calendars[fold] = calendar
            cutoff = validation_tau(validation, fold=fold, as_of=first)
            tau_map[(model, fold)] = cutoff
            for j, ticker in enumerate(TICKERS):
                part = validation[validation.Ticker == ticker]
                thresholds.append({"Model": model, "Fold": fold, "Ticker": ticker, "Tau": cutoff[j],
                                   "Validation_Rows": len(part), "Validation_Signal_Days": part.Signal_Date.nunique(),
                                   "Latest_Validation_Label_End": part.Label_End.max(), "Test_First_Signal": first})
    return pred, test, tau_map, pd.DataFrame(thresholds)


def prediction_diagnostics(predictions):
    """Descriptive row errors; correlated stock rows are never independent trials."""
    rows = []
    groups = [(model, "all", split, group) for (model, split), group in predictions.groupby(["Model", "Split"], sort=False)]
    groups += [(model, fold, split, group) for (model, fold, split), group in predictions.groupby(
        ["Model", "Fold", "Split"], sort=False)]
    for model, fold, split, group in groups:
        matured = group[group.Label_State.str.lower() == "matured"]
        actual = matured.Actual_5d_Pct.to_numpy(dtype=float)
        predicted = matured.Predicted.to_numpy(dtype=float)
        directional_actual = np.abs(actual) > DIRECTION_ZERO_TOLERANCE_PP
        direction_eligible = int(directional_actual.sum())
        always_up = float((actual[directional_actual] > 0).mean() * 100) if direction_eligible else np.nan
        for forecast, values in (("model", predicted), ("zero_return_baseline", np.zeros(len(matured)))):
            error = values - actual
            scored = directional_actual & (np.abs(values) > DIRECTION_ZERO_TOLERANCE_PP)
            scored_count = int(scored.sum())
            direction = float((np.sign(values[scored]) == np.sign(actual[scored])).mean() * 100) if scored_count else np.nan
            rows.append({"Model": model, "Fold": fold, "Split": split, "Forecast": forecast,
                         "Rows": len(group), "Signal_Days": group.Signal_Date.nunique(),
                         "Matured_Rows": len(matured), "Matured_Signal_Days": matured.Signal_Date.nunique(),
                         "Pending_Rows": len(group) - len(matured),
                         "Pending_Signal_Days": group.loc[group.Label_State.str.lower() == "pending", "Signal_Date"].nunique(),
                         "MAE_Pp": float(np.abs(error).mean()) if len(error) else np.nan,
                         "RMSE_Pp": float(np.sqrt(np.mean(error ** 2))) if len(error) else np.nan,
                         "Direction_Accuracy_Pct": direction,
                         "Direction_Eligible_Rows": direction_eligible, "Direction_Scored_Rows": scored_count,
                         "Direction_Coverage_Pct": float(scored_count / direction_eligible * 100) if direction_eligible else np.nan,
                         "Always_Up_Accuracy_Pct": always_up,
                         "Direction_Interpretation": "Nonzero actual returns only; zero predictions abstain; zero-return baseline has no direction score; not a trade win rate",
                         "Independent_Sample_Count": np.nan,
                         "Inference_Unit": "overlapping five-session labels; stocks share signal dates"})
    return pd.DataFrame(rows)


def _policies(test, risk, tau_map, signals):
    policies = {}
    equal = np.full(len(TICKERS), 1 / len(TICKERS))
    for signal in signals:
        flag = bool(risk.loc[signal, "Risk_Off"])
        for model in MODELS:
            day = test[(test.Model == model) & (test.Signal_Date == signal)].set_index("Ticker").loc[list(TICKERS)]
            fold = day.Fold.iloc[0]
            predicted = day.Predicted.to_numpy(dtype=float)
            cutoff = tau_map[(model, fold)]
            weights = threshold_weights(predicted, cutoff, flag)
            policies[(MODEL_ARMS[model], signal)] = (weights, model, fold, predicted, cutoff)
            if model == "news_lstm":
                policies[("v2_news_no_risk", signal)] = (threshold_weights(predicted, cutoff, False), model, fold, predicted, cutoff)
                policies[("v2_matched_equal", signal)] = (equal * weights.sum(), model, fold, predicted, cutoff)
        policies[("full_equal_5", signal)] = (equal.copy(), None, None, np.full(len(TICKERS), np.nan), np.full(len(TICKERS), np.nan))
    return policies


def _next_close_books(panel, dates, signals, scores, risk, policies, cfg):
    nav_rows, weight_rows, fill_rows = [], [], []
    for arm in NEW_ARMS:
        shares = np.zeros(len(TICKERS))
        cash = cfg.initial_capital
        previous_nav = cfg.initial_capital
        previous_price = None
        holding_risk = None
        for number, (date, signal) in enumerate(zip(dates, signals)):
            price = panel.loc[date, list(TICKERS)].to_numpy(dtype=float)
            before_shares = shares.copy()
            values = shares * price
            pre_nav = float(values.sum() + cash)
            holding_pnl = 0.0 if previous_price is None else float((shares * (price - previous_price)).sum())
            terminal = number == len(dates) - 1
            due = number % 5 == 0 and not terminal
            target, model, fold, predicted, cutoff = policies[(arm, signal)]
            requested = np.zeros(len(TICKERS)) if terminal else target
            flag = bool(risk.loc[signal, "Risk_Off"])
            entering_policy_risk = holding_risk
            ledger = {"Buy_Notional": 0.0, "Sell_Notional": 0.0, "Total_Cost": 0.0, "Turnover": 0.0}
            if due or terminal:
                after, cash, ledger = rebalance(values, cash, requested, cfg)
                delta = after - values
                shares = after / price
                for j, ticker in enumerate(TICKERS):
                    if abs(delta[j]) > 1e-8:
                        buy, sell = max(delta[j], 0), max(-delta[j], 0)
                        fill_rows.append({"Arm": arm, "Display_Name": DISPLAY_NAMES[arm], "Signal_Date": signal,
                                          "Execution_Date": date, "Ticker": ticker,
                                          "Shares_Before": before_shares[j], "Share_Change": shares[j] - before_shares[j],
                                          "Shares_After": shares[j], "Notional_Change": delta[j],
                                          "Broker_Fee": (buy + sell) * cfg.broker_fee_rate,
                                          "Sell_Tax": sell * cfg.sell_tax_rate,
                                          "Transaction_Cost": buy * cfg.broker_fee_rate + sell * (cfg.broker_fee_rate + cfg.sell_tax_rate),
                                          "Terminal_Liquidation": terminal,
                                          "Reason": "terminal_liquidation" if terminal else "next_session_close_allocation"})
                values = after
                holding_risk = flag
            nav = float(values.sum() + cash)
            if (not np.isclose(pre_nav, previous_nav + holding_pnl, rtol=1e-12, atol=1e-6)
                    or not np.isclose(nav, pre_nav - ledger["Total_Cost"], rtol=1e-12, atol=1e-6)):
                raise AssertionError("Daily holdings/cash/cost NAV identity failed")
            nav_rows.append({"Date": date, "Arm": arm, "Display_Name": DISPLAY_NAMES[arm], "NAV": nav,
                             "Cash": cash, "Equity_Weight": float(values.sum() / nav), "Risk_Off": flag,
                             "Transaction_Cost": ledger["Total_Cost"], "Turnover": ledger["Turnover"],
                             "Trade": bool(ledger["Buy_Notional"] + ledger["Sell_Notional"] > 1e-8),
                             "Rebalance": due, "Signal_Date": signal, "Terminal_Liquidation": terminal,
                             "Pre_Trade_NAV": pre_nav, "Holding_PnL": holding_pnl,
                             "Buy_Notional": ledger["Buy_Notional"], "Sell_Notional": ledger["Sell_Notional"],
                             "Holding_Policy_Risk_Off": entering_policy_risk,
                             "After_Close_Policy_Risk_Off": holding_risk})
            for j, ticker in enumerate(TICKERS):
                weight_rows.append({"Date": date, "Target_Date": date, "Arm": arm, "Display_Name": DISPLAY_NAMES[arm],
                                    "Signal_Date": signal, "Ticker": ticker, "Model": model, "Fold": fold,
                                    "Predicted": predicted[j], "Tau": cutoff[j],
                                    "Daily_Sentiment": float(scores.loc[signal, ticker]), "Risk_Off": flag,
                                    "Market_Vol_20d": float(risk.loc[signal, "Market_Vol_20d"]),
                                    "Risk_Threshold_Pct": float(risk.loc[signal, "Risk_Threshold_Pct"]),
                                    "Policy_Target_Weight": float(target[j]),
                                    "Requested_Weight": float(requested[j]) if due or terminal else np.nan,
                                    "Actual_Weight": float(values[j] / nav), "Shares": shares[j],
                                    "Market_Value": values[j], "Trade_Eligible": due,
                                    "Terminal_Liquidation": terminal})
            previous_nav, previous_price = nav, price
    return pd.DataFrame(nav_rows), pd.DataFrame(weight_rows), pd.DataFrame(fill_rows)


def _canonical(nav, metrics, dates, signals, cfg):
    nav, metrics = _frame(nav, "canonical_nav"), _frame(metrics, "canonical_metrics")
    needed = {"Date", "Arm", "NAV", "Cash", "Equity_Weight", "Risk_Off", "Transaction_Cost", "Turnover", "Signal_Date"}
    if not needed.issubset(nav) or not {"Arm", "ROI_Pct", "MDD_Pct", "Sharpe", "Total_Transaction_Cost"}.issubset(metrics):
        raise ValueError("Canonical NAV/metrics schema is incomplete")
    nav = nav[nav.Arm.isin(CANONICAL_ARMS)].copy()
    metrics = metrics[metrics.Arm.isin(CANONICAL_ARMS)].copy()
    nav["Date"], nav["Signal_Date"] = _dates(nav.Date), _dates(nav.Signal_Date)
    nav["Risk_Off"] = _bools(nav.Risk_Off)
    if set(nav.Arm) != set(CANONICAL_ARMS) or set(metrics.Arm) != set(CANONICAL_ARMS) or metrics.Arm.duplicated().any():
        raise ValueError("Both canonical 444 and 555 arms must be preserved")
    for arm in CANONICAL_ARMS:
        part = nav[nav.Arm == arm].sort_values("Date")
        if not _dates(part.Date).equals(dates) or not _dates(part.Signal_Date).equals(signals):
            raise ValueError("Canonical NAV differs from the protocol's common calendar")
        if not np.isfinite(part[["NAV", "Cash", "Equity_Weight", "Transaction_Cost", "Turnover"]].to_numpy()).all():
            raise ValueError("Canonical NAV contains missing or invalid values")
        if not np.allclose(part.NAV - part.Cash, part.NAV * part.Equity_Weight, atol=1e-6, rtol=1e-12):
            raise ValueError("Canonical cash and equity do not reconcile to NAV")
        if not np.isclose(part.NAV.iloc[0] + part.Transaction_Cost.iloc[0], cfg.initial_capital, atol=1e-6):
            raise ValueError("Canonical first entry does not reconcile to initial cash")
        if not np.isclose(part.Cash.iloc[-1], part.NAV.iloc[-1], atol=1e-6) or abs(part.Equity_Weight.iloc[-1]) > 1e-10:
            raise ValueError("Canonical terminal day must be fully liquidated")
        metric = metrics[metrics.Arm == arm].iloc[0]
        if not np.isclose(metric.ROI_Pct, (part.NAV.iloc[-1] / cfg.initial_capital - 1) * 100, atol=1e-8):
            raise ValueError("Canonical ROI does not match its preserved NAV")
        if not np.isclose(metric.Total_Transaction_Cost, part.Transaction_Cost.sum(), atol=1e-6):
            raise ValueError("Canonical costs do not match its preserved NAV")
    nav["Display_Name"] = nav.Arm.map(DISPLAY_NAMES)
    nav["Terminal_Liquidation"] = nav.Date.eq(dates[-1])
    nav["Trade"] = nav.Transaction_Cost.gt(1e-8)
    nav["Rebalance"] = (~nav.Terminal_Liquidation) & ((nav.Arm == "dynamic_news_risk555") | nav.Date.eq(dates[0]))
    metrics["Display_Name"] = metrics.Arm.map(DISPLAY_NAMES)
    return nav, metrics


def _metrics(nav, canonical_metrics, cfg):
    rows = []
    for arm in NEW_ARMS:
        part = nav[nav.Arm == arm].sort_values("Date")
        values = np.r_[cfg.initial_capital, part.NAV.to_numpy(dtype=float)]
        returns = values[1:] / values[:-1] - 1
        std = returns.std(ddof=1)
        rows.append({"Arm": arm, "Display_Name": DISPLAY_NAMES[arm],
                     "Description": DISPLAY_NAMES[arm], "Result_Status": "historical_exploration",
                     "Return_Basis": "net_research_portfolio_fractional_shares",
                     "First_Execution_Date": str(part.Date.min().date()),
                     "Last_Valuation_Date": str(part.Date.max().date()), "Observations": len(part),
                     "ROI_Pct": float((values[-1] / cfg.initial_capital - 1) * 100),
                     "MDD_Pct": float((values / np.maximum.accumulate(values) - 1).min() * 100),
                     "Sharpe": float(returns.mean() / std * np.sqrt(cfg.annual_trading_days)) if std > 1e-12 else 0.0,
                     "Total_Transaction_Cost": float(part.Transaction_Cost.sum()),
                     "Total_Turnover": float(part.Turnover.sum()),
                     "Average_Equity_Weight_Pct": float(part.Equity_Weight.mean() * 100)})
    result = pd.concat([canonical_metrics, pd.DataFrame(rows)], ignore_index=True, sort=False)
    for position, row in result.iterrows():
        part = nav[nav.Arm == row.Arm].sort_values("Date")
        if "Average_Equity_Weight_Pct" not in result or pd.isna(row.get("Average_Equity_Weight_Pct")):
            result.loc[position, "Average_Equity_Weight_Pct"] = float(part.Equity_Weight.mean() * 100)
        result.loc[position, "Average_Cash_Weight_Pct"] = float((part.Cash / part.NAV).mean() * 100)
        result.loc[position, "Rebalance_Days"] = int(part.Rebalance.sum())
        result.loc[position, "Trade_Days"] = int(part.Trade.sum())
        result.loc[position, "Sessions"] = len(part)
    roi = result.set_index("Arm").ROI_Pct
    result["Gap_Vs_444_Pp"] = result.ROI_Pct - roi["static_equal444"]
    result["Gap_Vs_555_Pp"] = result.ROI_Pct - roi["dynamic_news_risk555"]
    result["Gap_Vs_Matched_Equal_Pp"] = result.ROI_Pct - roi["v2_matched_equal"]
    for column in ("Rebalance_Days", "Trade_Days", "Sessions"):
        result[column] = result[column].astype(int)
    return result


def _effects(nav, cfg):
    bootstrap = BootstrapConfig(block_length=20, draws=2000, seed=42)
    daily, entries = [], []
    for arm, group in nav.groupby("Arm", sort=False):
        group = group.sort_values("Date")
        values = group.NAV.to_numpy(dtype=float)
        entries.append({"Arm": arm, "Entry_Date": group.Date.iloc[0], "Initial_Capital": cfg.initial_capital,
                        "Entry_Transaction_Cost": float(group.Transaction_Cost.iloc[0]),
                        "Entry_Return_Pct": float((values[0] / cfg.initial_capital - 1) * 100),
                        "Included_In_Bootstrap": False})
        returns = (values[1:] / values[:-1] - 1) * 100
        for i, value in enumerate(returns, start=1):
            daily.append({"Arm": arm, "Date": group.Date.iloc[i], "Net_Return_Pct": float(value)})
    daily = pd.DataFrame(daily)
    matrix = daily.pivot(index="Date", columns="Arm", values="Net_Return_Pct").sort_index()
    indices = circular_block_indices(len(matrix), bootstrap)
    rows = []
    for name, arm_a, arm_b in EFFECTS:
        delta = (matrix[arm_a] - matrix[arm_b]).to_numpy(dtype=float)
        samples = delta[indices].mean(axis=1)
        lower, upper = np.quantile(samples, [.025, .975])
        rows.append({"Contrast": name, "Arm_A": arm_a, "Arm_B": arm_b,
                     "N_Days": len(delta), "Mean_Daily_Difference_Pp": float(delta.mean()),
                     "CI_Lower_Pp": float(lower), "CI_Upper_Pp": float(upper),
                     "Block_Length": 20, "Requested_Draws": 2000, "Usable_Draws": len(samples),
                     "Seed": 42, "Confidence_Level": .95, "Result_Status": "historical_exploration",
                     "Unit": "percentage points of mean arithmetic daily net return"})
    return pd.DataFrame(rows), daily, pd.DataFrame(entries), asdict(bootstrap)


def evaluate_common_books(panel, sentiment, predictions, paired_protocol, canonical_nav, canonical_metrics):
    """Evaluate fixed V2.12 policies and preserve the two canonical comparison arms.

    Every argument is an in-memory DataFrame except ``paired_protocol`` (dict).
    ``panel.index`` must retain the complete session calendar, including missing
    warm-up bars; predictions supply the three newly fitted models' validation
    and complete daily test signals. Pending actual labels must be NaN.

    Returns DataFrames ``nav``, ``metrics``, ``curves``, ``weights`` (new arms),
    ``fills`` (new arms), ``thresholds``, ``prediction_metrics``,
    ``prediction_timeline``, ``paired_effects``, ``daily_returns``, ``entry_costs``
    and a serializable ``protocol``. No input is mutated.
    """
    protocol = deepcopy(dict(paired_protocol))
    cfg = _config(protocol)
    panel, dates, signals = _calendar(panel, protocol)
    scores = _sentiment(sentiment, signals)
    pred, test, tau_map, thresholds = _predictions(predictions, panel, dates, signals)
    risk = market_risk(panel, cfg)
    if not np.isfinite(risk.loc[signals, ["Market_Vol_20d", "Risk_Threshold_Pct"]].to_numpy()).all():
        raise ValueError("Signal-day risk warm-up is incomplete")
    policies = _policies(test, risk, tau_map, signals)
    nav, weights, fills = _next_close_books(panel, dates, signals, scores, risk, policies, cfg)
    original_nav, original_metrics = _canonical(canonical_nav, canonical_metrics, dates, signals, cfg)
    nav = pd.concat([original_nav, nav], ignore_index=True, sort=False)
    nav = nav.sort_values(["Arm", "Date"], kind="stable").reset_index(drop=True)
    metrics = _metrics(nav, original_metrics, cfg)
    curves = nav[["Date", "Arm", "Display_Name"]].copy()
    curves["Wealth_Index"] = nav.NAV / cfg.initial_capital * 100
    peaks = nav.groupby("Arm", sort=False).NAV.cummax().clip(lower=cfg.initial_capital)
    curves["Drawdown_Pct"] = (nav.NAV / peaks - 1) * 100
    effects, daily, entries, bootstrap = _effects(nav, cfg)
    timeline = pred.groupby(["Model", "Fold", "Split", "Signal_Date", "Execution_Date", "Label_End", "Label_State"],
                            dropna=False, sort=True).agg(Rows=("Ticker", "size"), Actual_Count=("Actual_5d_Pct", "count")).reset_index()
    out_protocol = {
        "version": "V2.12", "status": "historical_exploration", "prospective_validated": False,
        "independent_holdout": False, "test_roi_used_for_selection": False,
        "rules": asdict(cfg), "tickers": list(TICKERS), "arms": DISPLAY_NAMES.copy(),
        "shared_target_dates": [str(day.date()) for day in dates],
        "shared_signal_dates": [str(day.date()) for day in signals], "sessions": len(dates),
        "v2_prediction_target": "100 * (Close[signal+6] / Close[signal+1] - 1)",
        "prediction_task_comparability": "Original 444/555 use next-day close forecasts and daily opportunities; V2 candidates share a five-session return target and five-session opportunities. Common valuation/costs do not make these prediction tasks identical.",
        "rebalance_schedule": "New arms at valuation offsets 0,5,10,... from the common first execution; terminal close liquidates only. Rebalance_Days excludes terminal liquidation.",
        "canonical_preservation": "The original static_equal444 and dynamic_news_risk555 NAV, cash, cost, turnover and existing metric values are copied unchanged. Their models are not retrained here.",
        "new_ledger": "paired_research.rebalance for every allocation and terminal liquidation; fractional shares; no minimum fee; zero cash yield, slippage and dividends",
        "entry_return": "Initial holdings are zero; first execution earns no already completed signal-close to execution-close return. Entry cost is included.",
        "terminal_policy": "Sell all holdings at the final common close, charge fee and sell tax, hold Cash=NAV and Equity_Weight=0; never open final-day positions",
        "tau": "Per model/fold/ticker median(abs(validation prediction)); every validation label ends strictly before that fold's first test signal",
        "risk": "scripts.paired_research.market_risk: signal-day known rolling volatility versus earlier expanding mean + sample std; never execution-day risk",
        "policy": "experiments.v2.books.threshold_weights fixed .20/.25/.15 base/up/down, normalized, .30 stock cap and .50 Risk-Off gross cap; news-no-risk omits only the risk cap",
        "matched_equal": "Exactly the V2 news risk policy target gross exposure and the same execution opportunities; actual weights drift between fills",
        "weights": "Policy_Target_Weight is that day's signal policy; Requested_Weight is populated only on a scheduled allocation or terminal liquidation; Actual_Weight reflects marked holdings after actual fills",
        "bootstrap": bootstrap, "bootstrap_method": "Same circular 20-session date blocks for all three fixed contrasts; first entry day excluded, including its separately reported cost; terminal liquidation costs included",
        "statistical_unit": "Aligned daily portfolio returns, not 150*5 independent observations; five-session forecast labels overlap and five stocks share dates",
        "direction_scoring": "Only nonzero actual/predicted pairs are scored; |return| <= 1e-8 percentage points is zero, zero predictions abstain, and coverage is reported. Zero-return forecast is an MAE/RMSE baseline with no direction score; always-up accuracy uses the same nonzero actuals. Direction accuracy is not a trade win rate.",
        "limitations": ["All results reuse examined history and remain exploratory; no superiority or causal claim follows from these intervals",
                        "Historical sentiment remains retrospective_unverified without independent acquisition/version evidence",
                        "News/no-news changes trained features within the same five-session task; original 444/555 have a different prediction task and frequency"],
    }
    return {"protocol": out_protocol, "nav": nav, "metrics": metrics, "curves": curves,
            "weights": weights, "fills": fills, "thresholds": thresholds,
            "prediction_metrics": prediction_diagnostics(pred), "prediction_timeline": timeline,
            "paired_effects": effects, "daily_returns": daily, "entry_costs": entries}
