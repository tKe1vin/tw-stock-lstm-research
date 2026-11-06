"""Next-close books. Risk-off stays on. Missing predictions do not move the calendar."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.calendar_rules import apply_rule, cap_weights
from experiments.horizon_experiment import TICKERS
from experiments.risk_experiment import (
    BROKER_FEE_RATE,
    INITIAL_CAPITAL,
    MARKET,
    SELL_TAX_RATE,
    max_drawdown,
    rebalance,
    sharpe_ratio,
)

BASE = 0.20
UP = 0.25
DOWN = 0.15
FLOOR = 0.05
CAP = 0.30
RISK = 0.50


def validation_tau(validation):
    """Per-ticker median absolute validation prediction. Test rows are ignored."""
    part = validation[validation.Split == "validation"] if "Split" in validation.columns else validation
    tau = []
    for ticker in TICKERS:
        values = part.loc[part.Ticker == ticker, "Predicted"].to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"{ticker} 沒有可用的驗證預測來決定門檻")
        tau.append(float(np.median(np.abs(values))))
    return np.asarray(tau, dtype=float)


def threshold_weights(predictions, tau, risk_off):
    pred = np.asarray(predictions, dtype=float)
    cutoff = np.asarray(tau, dtype=float)
    raw = np.full(len(pred), BASE)
    raw[pred > cutoff] = UP
    raw[pred < -cutoff] = DOWN
    raw = np.clip(raw, FLOOR, CAP)
    raw *= 1.0 / raw.sum()
    weights = cap_weights(raw, risk_off, RISK)
    if weights.max() > CAP + 1e-8:
        raise AssertionError("單一股票權重超過預先規定的 30%")
    return weights


def run_book(panel, risk, predictions, sentiment, start, end, frequency, weight_name, tau=None):
    """Fill at the close being marked, which is the next close after the signal."""
    if frequency not in (1, 5):
        raise ValueError("再平衡間隔只接受事先規定的每日或每五個交易日")
    index = panel.index[(panel.index >= start) & (panel.index <= end)]
    by_signal = {}
    if predictions is not None and len(predictions):
        for signal, day in predictions.groupby("Signal_Date", sort=True):
            by_signal[pd.Timestamp(signal)] = day.set_index("Ticker").loc[TICKERS]
    location = {pd.Timestamp(day): i for i, day in enumerate(panel.index)}
    shares = np.zeros(len(TICKERS))
    cash = INITIAL_CAPITAL
    previous = None
    rows = []
    anchor = location[pd.Timestamp(index[0])]
    for day in index:
        day = pd.Timestamp(day)
        price = panel.loc[day, TICKERS].to_numpy(dtype=float)
        held = shares.copy()
        change = np.zeros(len(TICKERS)) if previous is None else held * (price - previous)
        old_values = held * price
        pre_nav = float(old_values.sum() + cash)
        flag = bool(risk.loc[day, "Risk_Off"])
        due = (location[day] - anchor) % frequency == 0
        signal = panel.index[location[day] - 1] if location[day] > 0 else None
        info = by_signal.get(pd.Timestamp(signal)) if signal is not None else None
        traded = False
        skipped = False
        ledger = {"Total_Cost": 0.0, "Turnover": 0.0}
        weights = old_values / pre_nav if pre_nav else np.zeros(len(TICKERS))
        if due and weight_name == "buy_hold" and previous is None:
            weights = np.full(len(TICKERS), BASE)
            target_values, cash, ledger = rebalance(old_values, cash, weights)
            shares = target_values / price
            traded = True
        elif due and weight_name != "buy_hold":
            if info is None and weight_name != "equal":
                skipped = True
            else:
                pred = np.zeros(len(TICKERS))
                if info is not None:
                    pred = info.Predicted.to_numpy(dtype=float)
                if weight_name == "equal":
                    weights = apply_rule("B", pred, np.zeros(len(TICKERS)), flag, np.ones(len(TICKERS)))
                elif weight_name == "liquidation":
                    sent = sentiment.loc[pd.Timestamp(signal), TICKERS].to_numpy(dtype=float)
                    if not np.isfinite(sent).all():
                        skipped = True
                    else:
                        weights = apply_rule("A", pred, sent, flag, np.ones(len(TICKERS)))
                elif weight_name == "threshold":
                    weights = threshold_weights(pred, tau, flag)
                else:
                    raise ValueError(weight_name)
                if not skipped:
                    target_values, cash, ledger = rebalance(old_values, cash, weights)
                    shares = target_values / price
                    traded = True
        post_values = shares * price
        nav = float(post_values.sum() + cash)
        liquidation_cost = 0.0
        if day == index[-1]:
            liquidation_cost = float(post_values.sum() * (BROKER_FEE_RATE + SELL_TAX_RATE))
            nav -= liquidation_cost
        denominator = float(post_values.sum() + cash)
        row = {
            "Date": day,
            "Trade": traded,
            "Skipped_For_Missing_Prediction": skipped,
            "NAV_Reported": nav,
            "Rebalance_Cost": ledger["Total_Cost"],
            "Turnover": ledger.get("Turnover", 0.0),
            "Liquidation_Cost": liquidation_cost,
            "Equity_Weight": float(post_values.sum() / denominator) if denominator else 0.0,
            "Cash_Weight": float(cash / denominator) if denominator else 1.0,
            "Stock_PnL": float(change.sum()),
        }
        for j, ticker in enumerate(TICKERS):
            row[f"Target_{ticker}"] = float(weights[j])
            row[f"PnL_{ticker}"] = float(change[j])
        rows.append(row)
        previous = price
    daily = pd.DataFrame(rows)
    identity = (
        INITIAL_CAPITAL
        + float(daily.Stock_PnL.sum())
        - float(daily.Rebalance_Cost.sum())
        - float(daily.Liquidation_Cost.sum())
        - float(daily.NAV_Reported.iloc[-1])
    )
    if abs(identity) > 1e-4:
        raise AssertionError(f"淨值恆等式不成立，差額 {identity}")
    weights = daily[[f"Target_{ticker}" for ticker in TICKERS]].to_numpy(dtype=float)
    if np.any(weights < -1e-8) or np.any(weights.sum(axis=1) > 1 + 1e-6):
        raise AssertionError("權重為負或總和超過 1")
    return daily


def metrics(daily, name, schedule):
    values = np.concatenate([[INITIAL_CAPITAL], daily.NAV_Reported.to_numpy(dtype=float)])
    traded = daily[daily.Trade]
    exclusions = 0
    if len(traded):
        exclusions = int(sum((traded[f"Target_{ticker}"] <= 1e-12).sum() for ticker in TICKERS))
    return {
        "Name": name,
        "Schedule": schedule,
        "Study": "exploratory_not_a_new_independent_test",
        "Execution": "next_session_close",
        "Same_Close_Execution": False,
        "ROI_Pct": float((values[-1] / INITIAL_CAPITAL - 1) * 100),
        "Sharpe": sharpe_ratio(values),
        "MDD_Pct": max_drawdown(values),
        "Total_Transaction_Cost": float(daily.Rebalance_Cost.sum() + daily.Liquidation_Cost.sum()),
        "Average_Equity_Weight_Pct": float(daily.Equity_Weight.mean() * 100),
        "Average_Cash_Weight_Pct": float(daily.Cash_Weight.mean() * 100),
        "Exclusion_Count": exclusions,
        "Rebalance_Days": int(daily.Trade.sum()),
        "Skipped_Rebalance_Days": int(daily.Skipped_For_Missing_Prediction.sum()),
        "Sessions": int(len(daily)),
        "Average_Turnover": float(daily.Turnover.mean()),
    }


def contributions(daily, name):
    rows = []
    for ticker in TICKERS:
        rows.append({
            "Name": name,
            "Ticker": ticker,
            "PnL": float(daily[f"PnL_{ticker}"].sum()),
        })
    return pd.DataFrame(rows)


def index_metrics(panel, start, end):
    dates = panel.index[(panel.index >= start) & (panel.index <= end)]
    entry = float(panel.loc[dates[0], MARKET])
    shares = (INITIAL_CAPITAL / (1 + BROKER_FEE_RATE)) / entry
    cash = INITIAL_CAPITAL - shares * entry * (1 + BROKER_FEE_RATE)
    nav = []
    for i, day in enumerate(dates):
        value = shares * float(panel.loc[day, MARKET]) + cash
        if i == len(dates) - 1:
            value -= shares * float(panel.loc[day, MARKET]) * (BROKER_FEE_RATE + SELL_TAX_RATE)
        nav.append(value)
    values = np.concatenate([[INITIAL_CAPITAL], np.asarray(nav, dtype=float)])
    return {
        "Name": "twii_price_index_with_one_round_trip",
        "Schedule": "buy_and_hold_index",
        "Study": "exploratory_not_a_new_independent_test",
        "Execution": "next_session_close_window",
        "Same_Close_Execution": False,
        "ROI_Pct": float((values[-1] / INITIAL_CAPITAL - 1) * 100),
        "Price_ROI_Pct": float(panel.loc[dates[-1], MARKET] / entry - 1.0) * 100,
        "Sharpe": sharpe_ratio(values),
        "MDD_Pct": max_drawdown(values),
        "Total_Transaction_Cost": float(
            shares * entry * BROKER_FEE_RATE
            + shares * float(panel.loc[dates[-1], MARKET]) * (BROKER_FEE_RATE + SELL_TAX_RATE)
        ),
        "Average_Equity_Weight_Pct": 100.0,
        "Average_Cash_Weight_Pct": 0.0,
        "Note": "^TWII 是價格指數，不是可交易 ETF",
    }
