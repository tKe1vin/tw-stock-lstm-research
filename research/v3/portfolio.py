"""Deterministic 0050 core/satellite research ledger.

Dividends are receivables from the ex-date and become cash on their actual
payment date when provided. Events without a payment date use an explicitly
reported ex-date cash proxy; such output is not a live executable result.
Prices must be genuinely unadjusted; applying splits to split-adjusted Yahoo
prices would count the action twice. Orders are integer shares, fixed after the
signal close, and attempted at the next session close. Corporate-action
fractional entitlements are retained and separately disclosed, never traded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TICKERS = ("2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW")


@dataclass(frozen=True)
class _Settings:
    benchmark: str
    stocks: tuple[str, ...]
    initial_capital: float
    mode: str
    core_floor: float
    satellite_cap: float
    stock_cap: float
    top_n: int
    frequency: int
    hold: int
    band: float
    edge_multiplier: float
    fee: float
    minimum_fee: float
    stock_tax: float
    etf_tax: float
    slip: float
    volume_fraction: float


def _settings(config: dict[str, Any], stress_bps: float | None) -> _Settings:
    allocation, execution = config.get("allocation", {}), config.get("execution", {})
    if ("satellite_tickers" in config and "initial_satellite_tickers" in config
            and config["satellite_tickers"] != config["initial_satellite_tickers"]):
        raise ValueError("Conflicting satellite ticker aliases")
    if ("hold_sessions" in config and "hold_sessions" in config.get("target", {})
            and config["hold_sessions"] != config["target"]["hold_sessions"]):
        raise ValueError("Conflicting holding-interval aliases")
    volume_aliases = ("maximum_previous_volume_participation", "maximum_previous_session_volume_fraction")
    if all(name in execution for name in volume_aliases) and execution[volume_aliases[0]] != execution[volume_aliases[1]]:
        raise ValueError("Conflicting volume-participation aliases")
    if allocation.get("long_only", True) is not True or allocation.get("leverage", 1.) != 1.:
        raise ValueError("The ledger supports only unlevered long-only portfolios")
    if execution.get("entry", "next_session_close") != "next_session_close":
        raise ValueError("Only orders after the signal close at the next session close are supported")
    benchmark = config.get("benchmark", {})
    benchmark = benchmark.get("ticker", "0050.TW") if isinstance(benchmark, dict) else benchmark
    settings = _Settings(
        benchmark=str(benchmark),
        stocks=tuple(config.get("satellite_tickers", config.get("initial_satellite_tickers", DEFAULT_TICKERS))),
        initial_capital=float(config.get("initial_capital", 1_000_000)),
        mode=str(config.get("allocation_mode", "model")),
        core_floor=float(allocation.get("minimum_benchmark_weight", .8)),
        satellite_cap=float(allocation.get("maximum_satellite_weight", .2)),
        stock_cap=float(allocation.get("maximum_direct_stock_weight", .1)),
        top_n=int(allocation.get("maximum_active_stocks", 2)),
        frequency=int(execution.get("rebalance_every_sessions", 5)),
        hold=int(config.get("hold_sessions", config.get("target", {}).get("hold_sessions", 5))),
        band=float(execution.get("weight_change_no_trade_band", .02)),
        edge_multiplier=float(execution.get("minimum_expected_edge_cost_multiplier", 1.5)),
        fee=float(execution.get("broker_fee_rate", .001425)),
        minimum_fee=float(execution.get("minimum_broker_fee", 20.)),
        stock_tax=float(execution.get("stock_sell_tax", .003)),
        etf_tax=float(execution.get("equity_etf_sell_tax", .001)),
        slip=float(stress_bps if stress_bps is not None else execution.get("slippage_bps_each_side", 5.)) / 10_000,
        volume_fraction=float(execution.get("maximum_previous_volume_participation",
                                            execution.get("maximum_previous_session_volume_fraction", .01))),
    )
    numbers = [settings.initial_capital, settings.core_floor, settings.satellite_cap,
               settings.stock_cap, settings.band, settings.edge_multiplier, settings.fee,
               settings.minimum_fee, settings.stock_tax, settings.etf_tax, settings.slip,
               settings.volume_fraction]
    if not np.isfinite(numbers).all() or settings.initial_capital <= 0:
        raise ValueError("Ledger settings must be finite with positive capital")
    if settings.mode not in {"benchmark", "equal", "model"}:
        raise ValueError("allocation_mode must be benchmark, equal, or model")
    if settings.benchmark != "0050.TW":
        raise ValueError("V3 predeclares 0050.TW as its investable benchmark")
    if (not settings.stocks or len(set(settings.stocks)) != len(settings.stocks)
            or settings.benchmark in settings.stocks):
        raise ValueError("Satellite tickers must be distinct from the benchmark")
    if (not .8 <= settings.core_floor <= 1 or not 0 <= settings.satellite_cap <= .2
            or settings.satellite_cap > 1 - settings.core_floor + 1e-12
            or not 0 < settings.stock_cap <= .1 or not 1 <= settings.top_n <= 2):
        raise ValueError("V3 requires >=80% core, <=20% satellite, <=10% per stock, top <=2")
    if settings.frequency != 5 or settings.hold != 5:
        raise ValueError("The registered execution/holding interval is five sessions")
    if (execution.get("rebalance_every_sessions", 5) != settings.frequency
            or config.get("hold_sessions", config.get("target", {}).get("hold_sessions", 5)) != settings.hold
            or allocation.get("maximum_active_stocks", 2) != settings.top_n):
        raise ValueError("Session intervals and position counts must be exact integers")
    if (settings.band < 0 or settings.edge_multiplier < 0 or settings.minimum_fee < 0
            or any(not 0 <= x < 1 for x in (settings.fee, settings.stock_tax, settings.etf_tax, settings.slip))
            or not 0 < settings.volume_fraction <= 1):
        raise ValueError("Invalid costs, no-trade band, or participation fraction")
    return settings


def _normalise_bars(bars: pd.DataFrame, settings: _Settings) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    required = {"Date", "Ticker", "Close", "Volume", "Dividends", "Stock_Splits"}
    if not required.issubset(bars.columns):
        raise ValueError(f"Missing raw-bar columns: {sorted(required - set(bars.columns))}")
    result = bars.copy()
    metadata = bars.attrs.get("metadata", {})
    if metadata and (metadata.get("close_basis") != "truly_raw_unadjusted"
                     or metadata.get("dividend_basis") != "cash_per_post_event_share"):
        raise ValueError("Raw-price and post-event dividend conventions must be verified together")
    old_payment, payment = "Capital_Return_Payment_Date", "Return_of_Capital_Payment_Date"
    if old_payment in result and payment in result:
        a, b = pd.to_datetime(result[old_payment]), pd.to_datetime(result[payment])
        if not (a.eq(b) | (a.isna() & b.isna())).all():
            raise ValueError("Conflicting returned-capital payment-date aliases")
    elif old_payment in result:
        result = result.rename(columns={old_payment: payment})
    result["Date"] = pd.to_datetime(result.Date, errors="raise").dt.normalize()
    if result.Date.isna().any() or result.Date.dt.tz is not None:
        raise ValueError("Bar dates must be finite, timezone-naive session dates")
    result["Ticker"] = result.Ticker.astype(str)
    if result.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate Date/Ticker raw bars")
    if "Tradable" not in result:
        result["Tradable"] = True
    if result.Tradable.isna().any() or not result.Tradable.isin([True, False]).all():
        raise ValueError("Tradable must be an explicit boolean")
    result["Tradable"] = result.Tradable.astype(bool)
    if "Return_of_Capital" not in result:
        result["Return_of_Capital"] = 0.
    tickers = [settings.benchmark, *settings.stocks]
    result = result[result.Ticker.isin(tickers)]
    dates = pd.DatetimeIndex(sorted(result.Date.unique()))
    if len(dates) < settings.hold + 2:
        raise ValueError("Insufficient sessions for next-close entry and a five-session hold")
    result = result.set_index(["Date", "Ticker"]).sort_index()
    expected = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    if len(expected.difference(result.index)):
        raise ValueError("Missing market sessions/assets; explicit suspension rows are required")
    result = result.reindex(expected)
    numeric = ["Close", "Volume", "Dividends", "Stock_Splits", "Return_of_Capital"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    for amount, payment in (("Dividends", "Dividend_Payment_Date"),
                            ("Return_of_Capital", "Return_of_Capital_Payment_Date")):
        if payment not in result:
            result[payment] = pd.NaT
        else:
            result[payment] = pd.to_datetime(result[payment], errors="raise").dt.normalize()
        event = result[amount].gt(0)
        date_values = pd.Series(result.index.get_level_values("Date"), index=result.index)
        if (event & result[payment].notna() & result[payment].lt(date_values)).any():
            raise ValueError("A corporate-action payment date precedes its ex-date")
    if not np.isfinite(result[numeric[1:]].to_numpy(float)).all():
        raise ValueError("Non-finite volume or corporate actions")
    if (result[["Volume", "Dividends", "Stock_Splits", "Return_of_Capital"]] < 0).any().any():
        raise ValueError("Negative volume or corporate action")
    tradable = result.Tradable.to_numpy(bool)
    close = result.Close.to_numpy(float)
    if not (np.isfinite(close[tradable]) & (close[tradable] > 0)).all():
        raise ValueError("Missing or invalid tradable close; no stale-price fills")
    present = np.isfinite(close)
    if (close[present] <= 0).any() or np.isinf(close).any():
        raise ValueError("Invalid close")
    if ((~result.Tradable) & result.Volume.ne(0)).any():
        raise ValueError("An explicit suspension must have zero traded volume")
    return result, dates


def _predictions(predictions: pd.DataFrame | None, stocks: tuple[str, ...]) -> dict[pd.Timestamp, pd.DataFrame]:
    if predictions is None or predictions.empty:
        return {}
    result = predictions.copy()
    if "Date" not in result and "Signal_Date" in result:
        result = result.rename(columns={"Signal_Date": "Date"})
    if not {"Date", "Ticker", "Predicted"}.issubset(result):
        raise ValueError("Predictions need Date, Ticker, and Predicted percentage points")
    result["Date"] = pd.to_datetime(result.Date, errors="raise").dt.normalize()
    if result.Date.isna().any() or result.Date.dt.tz is not None:
        raise ValueError("Prediction dates must be finite session dates")
    result["Ticker"] = result.Ticker.astype(str)
    if not set(result.Ticker).issubset(set(stocks)):
        raise ValueError("Prediction ticker is outside the registered satellite universe")
    if result.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate prediction Date/Ticker")
    if not np.isfinite(pd.to_numeric(result.Predicted, errors="raise")).all():
        raise ValueError("Non-finite predicted excess return")
    if "Eligible" in result and (result.Eligible.isna().any() or not result.Eligible.isin([True, False]).all()):
        raise ValueError("Prediction eligibility must contain explicit booleans")
    result["Predicted"] = result.Predicted.astype(float)
    return {pd.Timestamp(day): part.set_index("Ticker") for day, part in result.groupby("Date", sort=True)}


def _estimated_switch_cost_pct(notional: float, settings: _Settings) -> float:
    """A stock round-trip plus both ETF switch legs, in percentage points."""
    fee_fraction = max(settings.fee, settings.minimum_fee / max(notional, 1e-12))
    return 100 * (4 * fee_fraction + settings.stock_tax + settings.etf_tax + 4 * settings.slip)


def _target_weights(day, info, previous, nav, settings):
    weights = {ticker: 0. for ticker in settings.stocks}
    threshold = settings.edge_multiplier * _estimated_switch_cost_pct(
        nav * min(settings.stock_cap, settings.satellite_cap / settings.top_n), settings)
    reason = "benchmark_control"
    if settings.mode == "equal":
        # This fixed five-stock universe control deliberately has five positions;
        # top-two is a model policy, not an undocumented change to the control.
        each = min(settings.satellite_cap / len(settings.stocks), settings.stock_cap)
        weights = dict.fromkeys(settings.stocks, each)
        reason = "fixed_equal_satellite_control"
    elif settings.mode == "model":
        if info is None or set(info.index) != set(settings.stocks):
            reason = "incomplete_prediction_cross_section_fallback_to_0050"
        elif "Eligible" in info and (info.Eligible.isna().any() or not info.Eligible.astype(bool).all()):
            reason = "ineligible_information_fallback_to_0050"
        else:
            chosen = sorted(((float(info.loc[ticker, "Predicted"]), ticker)
                             for ticker in settings.stocks if float(info.loc[ticker, "Predicted"]) > threshold),
                            key=lambda item: (-item[0], item[1]))[:settings.top_n]
            for _, ticker in chosen:
                wanted = min(settings.stock_cap, settings.satellite_cap / settings.top_n)
                old = min(float(previous.get(ticker, 0.)), settings.stock_cap)
                weights[ticker] = old if abs(wanted - old) < settings.band and old > 0 else wanted
            reason = "net_edge_gate_passed" if chosen else "no_net_edge_fallback_to_0050"
    weights[settings.benchmark] = 1 - sum(weights.values())
    if weights[settings.benchmark] < settings.core_floor - 1e-10:
        raise AssertionError("Core policy weight below its floor")
    return weights, threshold, reason


def _max_affordable(quantity: int, price: float, cash: float, settings: _Settings) -> int:
    low, high = 0, quantity
    while low < high:
        mid = (low + high + 1) // 2
        gross = mid * price
        cost = gross + max(settings.minimum_fee, gross * settings.fee)
        if cost <= cash + 1e-9:
            low = mid
        else:
            high = mid - 1
    return low


def _engine(raw, dates, by_signal, signal_dates, settings, end_date):
    assets = [settings.benchmark, *settings.stocks]
    location = {day: i for i, day in enumerate(dates)}
    signals = pd.DatetimeIndex(pd.to_datetime(list(signal_dates))).normalize()
    if not len(signals) or signals.hasnans or signals.has_duplicates or not signals.is_monotonic_increasing:
        raise ValueError("Signal dates must be nonempty, unique, and chronological")
    if len(signals.difference(dates)):
        raise ValueError("A signal is outside the market-session calendar")
    anchor = location[signals[0]]
    final = dates[-1] if end_date is None else pd.Timestamp(end_date).normalize()
    if final not in location:
        raise ValueError("The fixed terminal session is not present")
    eligible = [day for day in signals if (location[day] - anchor) % settings.frequency == 0
                and location[day] + settings.hold + 1 <= location[final]]
    if not eligible:
        raise ValueError("No scheduled signal has a complete five-session holding horizon")
    if end_date is None:
        final = dates[location[eligible[-1]] + settings.hold + 1]
    start = dates[location[eligible[0]] + 1]
    scheduled = set(eligible)
    shares = dict.fromkeys(assets, 0.)
    marks = dict.fromkeys(assets, np.nan)
    cash = settings.initial_capital
    cumulative_cost = 0.
    receivables = []
    unknown_payment_events = []
    pending = None
    rows, fills, policies = [], [], []

    def fill_orders(day, day_raw, order, terminal=False):
        nonlocal cash, cumulative_cost
        cost_today, rejected = 0., 0
        split_ratios = {asset: (float(day_raw.loc[asset, "Stock_Splits"]) or 1.) for asset in assets}
        target = dict.fromkeys(assets, 0.) if terminal else {
            asset: order["shares"][asset] * split_ratios[asset] for asset in assets}
        # Sell before buy; ETF buys have deterministic first priority, avoiding
        # accidental debt when the next close is above the signal close.
        for side in ("SELL", "BUY"):
            for asset in assets:
                delta = shares[asset] - target[asset] if side == "SELL" else target[asset] - shares[asset]
                wanted = int(np.floor(max(delta, 0.) + 1e-9))
                if wanted <= 0:
                    continue
                row = day_raw.loc[asset]
                prior_volume = float(order["volume"][asset]) * split_ratios[asset]
                limit = max(0, int(np.floor(prior_volume * settings.volume_fraction)))
                quantity = min(wanted, limit)
                price = float(row.Close)
                reason = "terminal_liquidation" if terminal else "next_session_close"
                if not bool(row.Tradable) or not np.isfinite(price) or row.Volume <= 0:
                    quantity, reason = 0, "untradable_at_execution_no_carry_forward"
                elif limit == 0:
                    quantity, reason = 0, "previous_known_volume_limit"
                fill_price = price * (1 + settings.slip if side == "BUY" else 1 - settings.slip)
                if side == "BUY" and quantity:
                    quantity = _max_affordable(quantity, fill_price, cash, settings)
                    if not quantity:
                        reason = "insufficient_cash"
                gross = quantity * fill_price if quantity else 0.
                fee = max(settings.minimum_fee, gross * settings.fee) if quantity else 0.
                tax = gross * (settings.etf_tax if asset == settings.benchmark else settings.stock_tax) if side == "SELL" else 0.
                if side == "SELL" and quantity and cash + gross - fee - tax < -1e-8:
                    quantity, gross, fee, tax, reason = 0, 0., 0., 0., "minimum_fee_exceeds_available_cash"
                slippage = quantity * abs(fill_price - price) if quantity else 0.
                if quantity:
                    if side == "SELL":
                        shares[asset] -= quantity
                        cash += gross - fee - tax
                    else:
                        shares[asset] += quantity
                        cash -= gross + fee
                    if quantity < wanted and reason in {"next_session_close", "terminal_liquidation"}:
                        reason = "partial_volume_or_cash_limit_no_carry_forward"
                else:
                    rejected += 1
                cost = fee + tax + slippage
                cost_today += cost
                cumulative_cost += cost
                fills.append({"Signal_Date": order["signal"], "Execution_Date": day,
                              "Ticker": asset, "Side": side, "Requested_Shares": wanted,
                              "Shares": quantity, "Close": price, "Fill_Price": fill_price if quantity else np.nan,
                              "Gross": gross, "Fee": fee, "Tax": tax, "Slippage": slippage,
                              "Total_Cost": cost, "Cash_After": cash,
                              "Status": "filled" if quantity == wanted else "partial" if quantity else "rejected",
                              "Reason": reason})
        if cash < -1e-7 or min(shares.values()) < -1e-7:
            raise AssertionError("Ledger borrowed cash or sold shares it did not own")
        return cost_today, rejected

    for day in dates[dates <= final]:
        day_raw = raw.loc[day]
        dividend_cash, capital_cash = 0., 0.
        paid = [event for event in receivables if event["payment"] <= day]
        receivables = [event for event in receivables if event["payment"] > day]
        dividend_cash += sum(event["amount"] for event in paid if event["kind"] == "dividend")
        capital_cash += sum(event["amount"] for event in paid if event["kind"] == "capital")
        new_receivable = 0.
        stale = []
        for asset in assets:
            row = day_raw.loc[asset]
            ratio = float(row.Stock_Splits) or 1.
            before = shares[asset]
            capital_amount = before * float(row.Return_of_Capital)
            shares[asset] *= ratio
            dividend_amount = shares[asset] * float(row.Dividends)
            for kind, amount, payment_column in (("dividend", dividend_amount, "Dividend_Payment_Date"),
                                                  ("capital", capital_amount, "Return_of_Capital_Payment_Date")):
                if amount <= 0:
                    continue
                payment = row[payment_column]
                if pd.isna(payment):
                    unknown_payment_events.append({"Date": str(day.date()), "Ticker": asset, "Kind": kind})
                    if kind == "dividend":
                        dividend_cash += amount
                    else:
                        capital_cash += amount
                elif payment <= day:
                    if kind == "dividend":
                        dividend_cash += amount
                    else:
                        capital_cash += amount
                else:
                    receivables.append({"payment": payment, "amount": amount, "kind": kind})
                    new_receivable += amount
            if np.isfinite(marks[asset]):
                # A known distribution detaches value from the share even when
                # trading is suspended. Carrying the old share mark unchanged
                # while adding the cash receivable would manufacture wealth.
                marks[asset] = (marks[asset] - float(row.Return_of_Capital)) / ratio - float(row.Dividends)
                if marks[asset] <= 0:
                    raise ValueError("Corporate action makes the mechanical ex-rights reference nonpositive")
            if np.isfinite(row.Close):
                marks[asset] = float(row.Close)
            elif shares[asset] > 0 and not np.isfinite(marks[asset]):
                raise ValueError("A suspended held asset has no previous verified valuation")
            if not row.Tradable:
                stale.append(asset)
        cash += dividend_cash + capital_cash
        day_cost, rejected = 0., 0
        if pending is not None:
            day_cost, rejected = fill_orders(day, day_raw, pending)
            pending = None  # Expired partial/unfilled orders are never rolled forward.
        if day == final:
            prior = dates[location[day] - 1]
            liquidation = {"signal": pd.NaT, "volume": raw.loc[prior].Volume.to_dict()}
            extra, failed = fill_orders(day, day_raw, liquidation, terminal=True)
            day_cost += extra
            rejected += failed
        held_value = sum(shares[asset] * marks[asset] for asset in assets if shares[asset] > 0)
        receivable_value = sum(event["amount"] for event in receivables)
        nav = cash + held_value + receivable_value
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("Nonpositive or nonfinite NAV")
        actual = {asset: shares[asset] * marks[asset] / nav if shares[asset] > 0 else 0. for asset in assets}
        if day >= start:
            rows.append({"Date": day, "NAV": nav, "Cash": cash, "Holdings_Value": held_value,
                         "Dividend_And_Capital_Receivables": receivable_value,
                         "New_Receivables": new_receivable,
                         "Dividend_Cash": dividend_cash, "Return_of_Capital_Cash": capital_cash,
                         "Total_Cost": day_cost, "Cumulative_Cost": cumulative_cost,
                         "Core_Weight": actual[settings.benchmark],
                         "Satellite_Weight": sum(actual[asset] for asset in settings.stocks),
                         "Cash_Weight": cash / nav, "Receivable_Weight": receivable_value / nav,
                         "Stale_Valuations": ",".join(stale),
                         "Unfilled_Orders": rejected})
        if day in scheduled:
            weights, threshold, reason = _target_weights(day, by_signal.get(day), actual, nav, settings)
            target_shares = {}
            for asset in assets:
                signal_price = float(day_raw.loc[asset, "Close"])
                if weights[asset] > 0 and (not bool(day_raw.loc[asset, "Tradable"]) or not np.isfinite(signal_price)):
                    # The entire paired strategy opportunity is unavailable.
                    weights = {ticker: float(ticker == settings.benchmark) for ticker in assets}
                    reason = "untradable_signal_asset_fallback_to_0050"
                    break
            for asset in assets:
                signal_price = float(day_raw.loc[asset, "Close"])
                target_shares[asset] = int(np.floor(nav * weights[asset] / signal_price)) if weights[asset] > 0 and np.isfinite(signal_price) else 0
                predicted = by_signal.get(day)
                estimate = float(predicted.loc[asset, "Predicted"]) if predicted is not None and asset in predicted.index else np.nan
                policies.append({"Signal_Date": day, "Execution_Date": dates[location[day] + 1],
                                 "Ticker": asset, "Policy_Target_Weight": weights[asset],
                                 "Target_Shares_At_Signal": target_shares[asset],
                                 "Predicted_Excess_Pct": estimate,
                                 "Minimum_Net_Edge_Pct": threshold, "Reason": reason})
            pending = {"signal": day, "shares": target_shares, "volume": day_raw.Volume.to_dict()}
    nav = pd.DataFrame(rows)
    residual = sum(shares[asset] * marks[asset] for asset in assets if shares[asset] > 0)
    fractional = sum((shares[asset] - np.floor(shares[asset] + 1e-8)) * marks[asset]
                     for asset in assets if shares[asset] > 0)
    returns = nav.NAV.div(nav.NAV.shift(1).fillna(settings.initial_capital)).sub(1)
    peak = np.maximum.accumulate(np.r_[settings.initial_capital, nav.NAV.to_numpy(float)])
    drawdowns = np.r_[settings.initial_capital, nav.NAV.to_numpy(float)] / peak - 1
    vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.
    fills_frame = pd.DataFrame(fills)
    metrics = {
        "ROI_Pct": float((nav.NAV.iloc[-1] / settings.initial_capital - 1) * 100),
        "Final_NAV": float(nav.NAV.iloc[-1]), "Initial_Capital": settings.initial_capital,
        "Sharpe": float(np.sqrt(252) * returns.mean() / vol) if vol > 1e-14 else 0.,
        "MDD_Pct": float(drawdowns.min() * 100), "Total_Transaction_Cost": cumulative_cost,
        "Start": str(start.date()), "End": str(final.date()), "Sessions": len(nav),
        "Mean_Core_Weight": float(nav.Core_Weight.mean()),
        "Mean_Satellite_Weight": float(nav.Satellite_Weight.mean()),
        "Mean_Cash_Weight": float(nav.Cash_Weight.mean()),
        "Terminal_Residual_Holdings_Value": float(residual),
        "Terminal_Fractional_Entitlement_Value": float(max(fractional, 0.)),
        "Terminal_Unliquidated_Whole_Shares_Value": float(max(residual - max(fractional, 0.), 0.)),
        "Unfilled_Order_Count": int(nav.Unfilled_Orders.sum()),
        "Partial_Order_Count": int(fills_frame.Status.eq("partial").sum()) if len(fills_frame) else 0,
        "Broker_Fees": float(fills_frame.Fee.sum()) if len(fills_frame) else 0.,
        "Sell_Taxes": float(fills_frame.Tax.sum()) if len(fills_frame) else 0.,
        "Slippage_Cost": float(fills_frame.Slippage.sum()) if len(fills_frame) else 0.,
        "Gross_Trading_Notional": float(fills_frame.Gross.sum()) if len(fills_frame) else 0.,
        "Terminal_Receivables": float(sum(event["amount"] for event in receivables)),
        "Unknown_Payment_Date_Event_Count": len(unknown_payment_events),
        "Valuation_Protocol": "retrospective_ex_date_cash_wealth_proxy" if unknown_payment_events else "raw_price_corporate_action_receivable_ledger",
        "Execution_Protocol": "integer_orders_fixed_at_signal_close_next_session_close_proxy",
        "Live_Execution_Proven": False,
    }
    return {"metrics": metrics, "nav": nav, "fills": fills_frame,
            "weights": pd.DataFrame(policies), "unknown_payment_events": unknown_payment_events}


def run_backtest(bars: pd.DataFrame, predictions: pd.DataFrame | None, signal_dates,
                 config: dict[str, Any], end_date=None, stress_bps: float | None = None) -> dict[str, Any]:
    """Run one registered arm and compare its aligned net wealth with 0050.

    ``Predicted`` is the five-session *excess return in percentage points*, not
    a rank or probability. Missing model cross-sections fall back to 0050.
    ``signal_dates`` fixes the schedule independent of model availability. The
    input calendar must retain exchange sessions on which an asset is suspended.
    An explicit ``end_date`` is shared by every arm; without one the book ends
    after the last scheduled complete five-session holding period.
    """
    settings = _settings(config, stress_bps)
    raw, dates = _normalise_bars(bars, settings)
    by_signal = _predictions(predictions, settings.stocks)
    signals = list(signal_dates)
    result = _engine(raw, dates, by_signal, signals, settings, end_date)
    benchmark_settings = _Settings(**{**settings.__dict__, "mode": "benchmark"})
    benchmark = result if settings.mode == "benchmark" else _engine(
        raw, dates, {}, signals, benchmark_settings, result["metrics"]["End"])
    nav, reference = result["nav"], benchmark["nav"]
    if not nav.Date.equals(reference.Date):
        raise AssertionError("Strategy and 0050 do not have identical evaluation dates")
    strategy_return = nav.NAV.div(nav.NAV.shift(1).fillna(settings.initial_capital)).sub(1)
    benchmark_return = reference.NAV.div(reference.NAV.shift(1).fillna(settings.initial_capital)).sub(1)
    active = strategy_return - benchmark_return
    active_std = float(active.std(ddof=1)) if len(active) > 1 else 0.
    variance = float(benchmark_return.var(ddof=1)) if len(active) > 1 else 0.
    metrics = result["metrics"]
    metrics.update({
        "Benchmark": settings.benchmark, "Allocation_Mode": settings.mode,
        "Benchmark_ROI_Pct": benchmark["metrics"]["ROI_Pct"],
        "Benchmark_Total_Transaction_Cost": benchmark["metrics"]["Total_Transaction_Cost"],
        "Net_Excess_ROI_Pp": metrics["ROI_Pct"] - benchmark["metrics"]["ROI_Pct"],
        "Net_Relative_Wealth_Pct": float((nav.NAV.iloc[-1] / reference.NAV.iloc[-1] - 1) * 100),
        "Annualized_Tracking_Error": float(np.sqrt(252) * active_std),
        "Information_Ratio": float(np.sqrt(252) * active.mean() / active_std) if active_std > 1e-14 else 0.,
        "Beta_To_0050": float(strategy_return.cov(benchmark_return) / variance) if variance > 1e-14 else None,
        "Slippage_Bps_Each_Side": settings.slip * 10_000,
    })
    metrics["Tracking_Error_Annualized"] = metrics["Annualized_Tracking_Error"]
    metrics["Relative_Wealth_Excess_Pct"] = metrics["Net_Relative_Wealth_Pct"]
    nav["Benchmark_NAV"] = reference.NAV.to_numpy(float)
    nav["Net_Relative_Wealth"] = nav.NAV / nav.Benchmark_NAV - 1
    result["protocol"] = {
        "price_convention": "truly_raw_unadjusted_with_explicit_actions",
        "dividend_cash_timing": "receivable_on_ex_date_cash_on_payment_date_when_available",
        "return_of_capital_cash_timing": "receivable_on_ex_date_cash_on_payment_date_when_available",
        "unknown_payment_events": result.pop("unknown_payment_events"),
        "integer_order_shares": True, "corporate_action_fractional_entitlements_traded": False,
        "liquidity_limit": "1% of signal-date volume, mechanically adjusted for execution-date split",
        "current_zero_volume_prevents_fill": True, "unfilled_orders_carried_forward": False,
        "signal_date_sizing": True, "minimum_fee_per_order": settings.minimum_fee,
        "terminal_liquidation_subject_to_same_liquidity_and_fee_rules": True,
        "benchmark_cash_policy": "hold_0050_shares_and_sweep_available_cash_into_0050_on_the_same_five_session_schedule",
        "suspension_valuation": "mechanical_ex_rights_reference_then_last_known_mark_until_a_real_quote_returns",
        "execution_is_historical_proxy_not_guaranteed_auction_fill": True,
        "equal_control_has_five_stocks_model_has_at_most_two": True,
        "performance_claim": "exploratory_only",
        "source_price_basis_metadata_present": bool(bars.attrs.get("metadata")),
    }
    return result
