"""Paired 444-core allocation research; no model fitting or test-based rule selection.

All arms read the *same* saved 444 predictions. ``dynamic_news_risk555`` is
an allocation overlay, not a claim that the historical multivariate 555 LSTM
has been retrained with the same core. Historical news without both publication
and acquisition-time evidence remains exploratory. A next-close entry never earns
the already completed signal-close to execution-close return.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

TICKERS = ("2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW")
MARKET = "^TWII"
ARMS = (
    "static_equal444", "dynamic_no_news", "dynamic_news_no_risk", "dynamic_no_news_risk", "dynamic_news_risk555",
    "equal_rebalanced_reference", "risk_matched_equal_reference",
)
ARM_LABELS = {
    "static_equal444": "444 initial equal allocation, fixed shares",
    "dynamic_no_news": "444 core + prediction-based dynamic allocation",
    "dynamic_news_no_risk": "444 core + dynamic allocation + news",
    "dynamic_news_risk555": "444 core + dynamic allocation + news + market risk",
    "dynamic_no_news_risk": "444 core + dynamic allocation + market risk, no news",
    "equal_rebalanced_reference": "Daily equal allocation, same fills and costs",
    "risk_matched_equal_reference": "Daily equal allocation, same gross exposure as news+risk arm",
}


@dataclass(frozen=True)
class ResearchConfig:
    initial_capital: float = 100_000.0
    broker_fee_rate: float = 0.001425
    sell_tax_rate: float = 0.003
    annual_trading_days: int = 252
    sentiment_bull: float = 0.3
    sentiment_bear: float = -0.3
    max_stock_weight: float = 0.30
    risk_off_gross_cap: float = 0.50
    volatility_window: int = 20
    risk_min_history: int = 60
    baseline_mode: str = "fixed_initial_shares"
    liquidate_at_end: bool = True
    information_cutoff: str = "13:30:00"

    def __post_init__(self):
        numeric = [self.initial_capital, self.broker_fee_rate, self.sell_tax_rate,
                   self.sentiment_bull, self.sentiment_bear, self.max_stock_weight,
                   self.risk_off_gross_cap]
        if not np.isfinite(numeric).all() or self.initial_capital <= 0:
            raise ValueError("Invalid capital or rule parameters")
        if not 0 <= self.broker_fee_rate < 1 or not 0 <= self.sell_tax_rate < 1:
            raise ValueError("Invalid transaction cost rates")
        if not 0 <= self.risk_off_gross_cap <= 1 or not 0 < self.max_stock_weight <= 1:
            raise ValueError("Invalid exposure caps")
        if self.sentiment_bear >= self.sentiment_bull:
            raise ValueError("Sentiment thresholds must be ordered")
        if self.baseline_mode not in {"fixed_initial_shares", "daily_equal_weight"}:
            raise ValueError("Unsupported baseline mode")
        if min(self.volatility_window, self.risk_min_history) < 2 or self.annual_trading_days < 1:
            raise ValueError("Risk windows and annualization must be positive")
        pd.Timestamp("2000-01-01 " + self.information_cutoff)


def _resolved_config(config=None, baseline_protocol=None):
    if config is not None:
        return config
    baseline = dict(baseline_protocol or {})
    keys = {"initial_capital", "broker_fee_rate", "sell_tax_rate", "annual_trading_days", "liquidate_at_end"}
    return ResearchConfig(**{key: baseline[key] for key in keys if key in baseline})


def _frame(value):
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def _dates(values):
    # CSV price dates are date labels, not UTC timestamps to be shifted in time.
    result = pd.to_datetime(values, errors="raise")
    return result.dt.tz_localize(None).dt.normalize()


def _digest(frame):
    return hashlib.sha256(frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


def price_panel(prices, tickers=TICKERS):
    raw = _frame(prices)
    if {"Date", "ticker", "Close"}.issubset(raw.columns):
        raw["Date"] = _dates(raw.Date)
        raw = raw[raw.ticker.isin(tuple(tickers) + (MARKET,))]
        if raw.duplicated(["Date", "ticker"]).any():
            raise ValueError("Duplicate price date/ticker")
        panel = raw.pivot(index="Date", columns="ticker", values="Close").sort_index()
    else:
        if "Date" in raw:
            raw["Date"] = _dates(raw.Date)
            raw = raw.set_index("Date")
        panel = raw.copy().sort_index()
        panel.index = pd.DatetimeIndex(panel.index).tz_localize(None).normalize()
    columns = list(tickers) + [MARKET]
    if panel.index.duplicated().any() or not set(columns).issubset(panel.columns):
        raise ValueError("Prices need unique dates, all selected stocks and ^TWII")
    panel = panel[columns].astype(float)
    if np.isinf(panel.to_numpy()).any() or (panel <= 0).any().any():
        raise ValueError("Observed prices must be finite and positive")
    # Missing warm-up history stays missing and is reported. No row is dropped
    # or forward-filled. Signal and evaluation rows are checked strictly below.
    panel.index.name = "Date"
    return panel


def market_risk(panel, config=None):
    """Use volatility through signal close and a threshold based on earlier closes."""
    cfg = config or ResearchConfig()
    market = panel[MARKET]
    vol = market.pct_change(fill_method=None).rolling(cfg.volatility_window).std(ddof=1)
    vol = vol * np.sqrt(cfg.annual_trading_days) * 100
    prior = vol.shift(1)
    threshold = prior.expanding(min_periods=cfg.risk_min_history).mean() + prior.expanding(
        min_periods=cfg.risk_min_history).std(ddof=1)
    return pd.DataFrame({"Market_Vol_20d": vol, "Risk_Threshold_Pct": threshold,
                         "Risk_Off": vol > threshold}, index=panel.index)


def allocation_weights(predictions, sentiments=None, *, use_news=False, risk_off=False,
                       use_risk=False, config=None):
    """Fixed rules: base 1/N; positive+news>0.3 raises to .3; negative signal/news exits.

    Unallocated weight remains cash. Caps only scale weights down; no stock is
    selected by realized return, and the test set cannot tune these thresholds.
    """
    cfg = config or ResearchConfig()
    pred = np.asarray(predictions, dtype=float)
    if pred.ndim != 1 or len(pred) == 0 or not np.isfinite(pred).all():
        raise ValueError("Predictions must be a finite vector")
    weights = np.full(len(pred), min(1 / len(pred), cfg.max_stock_weight), dtype=float)
    weights[pred < 0] = 0.0
    if use_news:
        news = np.asarray(sentiments, dtype=float)
        if news.shape != pred.shape or not np.isfinite(news).all() or (np.abs(news) > 1).any():
            raise ValueError("News must provide a finite [-1, 1] score for every stock")
        weights[(pred > 0) & (news > cfg.sentiment_bull)] = cfg.max_stock_weight
        weights[(pred < 0) | (news < cfg.sentiment_bear)] = 0.0
    cap = cfg.risk_off_gross_cap if use_risk and risk_off else 1.0
    if weights.sum() > cap:
        weights *= cap / weights.sum()
    return weights


def _sentiment_table(sentiment, dates, tickers, cfg, decision_at=None):
    raw = _frame(sentiment)
    if not {"Date", "ticker", "Daily_Sentiment"}.issubset(raw.columns):
        raise ValueError("News requires Date, ticker, Daily_Sentiment")
    raw["Date"] = _dates(raw.Date)
    raw = raw[raw.Date.isin(dates) & raw.ticker.isin(tickers)].copy()
    if raw.duplicated(["Date", "ticker"]).any():
        raise ValueError("Duplicate sentiment date/ticker")
    scores = raw.pivot(index="Date", columns="ticker", values="Daily_Sentiment").reindex(
        index=dates, columns=tickers).astype(float)
    if not np.isfinite(scores.to_numpy()).all() or (scores.abs() > 1).any().any():
        raise ValueError("Missing or invalid signal-day news; unknown news is not silently neutral")
    publication = next((c for c in ("Latest_News_Timestamp", "Published_At", "published_at") if c in raw), None)
    acquired = next((c for c in ("Available_At", "Retrieved_At", "Fetched_At", "available_at", "retrieved_at", "fetched_at") if c in raw), None)
    complete = publication is not None and acquired is not None
    decision_map = {}
    if decision_at is not None:
        if isinstance(decision_at, Mapping):
            source_decisions = decision_at.items()
        else:
            if len(dates) != 1:
                raise ValueError("Historical decision_at requires one timestamp per signal date")
            source_decisions = [(dates[0], decision_at)]
        for day, value in source_decisions:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None:
                raise ValueError("Decision timestamps must contain a timezone")
            label = pd.Timestamp(day).normalize().tz_localize(None)
            if stamp.tz_convert("Asia/Taipei").date() != label.date():
                raise ValueError("Decision timestamp must be on its signal date")
            decision_map[label] = stamp.tz_convert("UTC")
    for column in (publication, acquired):
        if column is None:
            continue
        populated = raw[column].notna() & raw[column].astype(str).str.strip().ne("")
        complete = complete and bool(populated.all())
        if populated.any():
            values = raw.loc[populated, column]
            if any(pd.Timestamp(value).tzinfo is None for value in values):
                raise ValueError(f"News {column} timestamps require an explicit timezone")
            stamps = pd.to_datetime(values, utc=True, errors="raise")
            if column == publication:
                cutoff = pd.to_datetime(raw.loc[populated, "Date"].dt.strftime("%Y-%m-%d") + " " + cfg.information_cutoff)
                cutoff = cutoff.dt.tz_localize("Asia/Taipei").dt.tz_convert("UTC")
                reason = "publication cutoff"
            else:
                # Acquisition can occur after the 13:30 publication mapping cutoff,
                # provided it is available by the actual evening decision. Without
                # decision evidence only the signal-day end is enforced; not PIT proof.
                cutoff = raw.loc[populated, "Date"].map(lambda day: decision_map.get(
                    day, (day + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)).tz_localize("Asia/Taipei").tz_convert("UTC")))
                reason = "recorded decision time or signal-day end"
            if (stamps > cutoff).any():
                raise ValueError(f"News {column} is later than the {reason}")
    evidence = {
        "publication_timestamp_column": publication,
        "publication_cutoff_local_time": cfg.information_cutoff,
        "acquisition_timestamp_column": acquired,
        "aggregate_timestamp_columns_complete": bool(complete),
        "recorded_decision_times": {str(day.date()): value.isoformat() for day, value in decision_map.items()},
        "decision_time_evidence_complete": all(day in decision_map for day in dates),
        "unknown_decision_policy": "Check acquisition no later than signal-day end; this is not evidence it existed at the actual decision",
        "news_point_in_time_verified": False,
        "reason": "Aggregate timestamps do not prove per-article membership, retrieval or immutable historical scoring; exploratory unless separately audited",
    }
    return scores, evidence, raw.sort_values(["Date", "ticker"])


def _prediction_table(predictions, panel, tickers):
    pred = _frame(predictions)
    if "Target_Date" not in pred and "Date" in pred:
        pred["Target_Date"] = pred["Date"]
    required = {"Signal_Date", "Target_Date", "Ticker"}
    if not required.issubset(pred.columns):
        raise ValueError("Predictions require Signal_Date, Target_Date/Date, Ticker")
    pred = pred[pred.Ticker.isin(tickers)].copy()
    pred["Signal_Date"] = _dates(pred.Signal_Date)
    pred["Target_Date"] = _dates(pred.Target_Date)
    if pred.empty or pred.duplicated(["Target_Date", "Ticker"]).any():
        raise ValueError("Empty or duplicate predictions")
    expected_pairs = None
    for ticker in tickers:
        part = pred[pred.Ticker == ticker].sort_values("Target_Date")
        pairs = list(zip(part.Signal_Date, part.Target_Date))
        if expected_pairs is None:
            expected_pairs = pairs
        if not pairs or pairs != expected_pairs:
            raise ValueError("All arms/stocks must share the exact prediction signal and target dates")
    dates = pd.DatetimeIndex([x[1] for x in expected_pairs])
    if dates.duplicated().any():
        raise ValueError("More than one signal per execution date")
    for signal, target in expected_pairs:
        if signal not in panel.index or target not in panel.index:
            raise ValueError("Prediction date absent from same-source price panel")
        loc = panel.index.get_loc(signal)
        if loc + 1 >= len(panel) or panel.index[loc + 1] != target:
            raise ValueError("Target must be the next observed common trading session after signal")
    # Missing prediction days must not silently stretch a position's holding period.
    expected_dates = panel.index[(panel.index >= dates.min()) & (panel.index <= dates.max())]
    if not dates.equals(expected_dates):
        raise ValueError("Predictions skip trading sessions inside the evaluation period")
    previous = np.asarray([panel.loc[s, t] for s, t in zip(pred.Signal_Date, pred.Ticker)])
    actual = np.asarray([panel.loc[d, t] for d, t in zip(pred.Target_Date, pred.Ticker)])
    for column, truth in (("Previous_Close", previous), ("Actual_Close", actual)):
        if column in pred and not np.allclose(pd.to_numeric(pred[column]), truth, rtol=1e-6, atol=1e-4):
            raise ValueError(f"Prediction {column} differs from the supplied price snapshot")
    if "Predicted_Close" in pred:
        core = (pd.to_numeric(pred.Predicted_Close).to_numpy() / previous - 1) * 100
        if "Predicted_Return_Pct" in pred and not np.allclose(
                core, pd.to_numeric(pred.Predicted_Return_Pct), rtol=2e-5, atol=1e-5):
            raise ValueError("Stored price and return predictions disagree")
        pred["Core_Predicted_Return_Pct"] = core
    elif "Predicted_Return_Pct" in pred:
        pred["Core_Predicted_Return_Pct"] = pd.to_numeric(pred.Predicted_Return_Pct)
    else:
        raise ValueError("No predicted price or return")
    if not np.isfinite(pred.Core_Predicted_Return_Pct).all() or (pred.Core_Predicted_Return_Pct <= -100).any():
        raise ValueError("Invalid core predictions")
    return pred.sort_values(["Target_Date", "Ticker"]), dates


def rebalance(current_values, cash, target_weights, config=None):
    """Post-fee target weights solved with cash conservation and sell tax."""
    cfg = config or ResearchConfig()
    current = np.asarray(current_values, dtype=float)
    weights = np.asarray(target_weights, dtype=float)
    total = float(current.sum() + cash)
    if (not np.isfinite(current).all() or not np.isfinite(weights).all() or not np.isfinite(cash)
            or current.shape != weights.shape or total <= 0 or cash < -1e-7
            or (current < 0).any() or (weights < 0).any() or weights.sum() > 1 + 1e-10):
        raise ValueError("Invalid holdings, cash or weights")
    low, high = 0.0, total
    for _ in range(64):
        after = (low + high) / 2
        delta = weights * after - current
        cost = delta.clip(min=0).sum() * cfg.broker_fee_rate + (-delta.clip(max=0)).sum() * (
            cfg.broker_fee_rate + cfg.sell_tax_rate)
        if after + cost > total:
            high = after
        else:
            low = after
    after = (low + high) / 2
    target = weights * after
    delta = target - current
    buy, sell = float(delta.clip(min=0).sum()), float((-delta.clip(max=0)).sum())
    cost = buy * cfg.broker_fee_rate + sell * (cfg.broker_fee_rate + cfg.sell_tax_rate)
    remaining = float(after - target.sum())
    if not np.isclose(target.sum() + remaining + cost, total, atol=1e-6):
        raise AssertionError("Cash conservation failed")
    return target, remaining, {"Buy_Notional": buy, "Sell_Notional": sell,
                               "Total_Cost": cost, "Turnover": (buy + sell) / total}


def _arm_weights(core, news, risk_off, cfg):
    equal = np.full(len(core), 1 / len(core))
    full = allocation_weights(core, news, use_news=True, use_risk=True, risk_off=risk_off, config=cfg)
    return {
        "static_equal444": equal,
        "dynamic_no_news": allocation_weights(core, config=cfg),
        "dynamic_news_no_risk": allocation_weights(core, news, use_news=True, config=cfg),
        "dynamic_news_risk555": full,
        "dynamic_no_news_risk": allocation_weights(core, use_risk=True, risk_off=risk_off, config=cfg),
        "equal_rebalanced_reference": equal,
        "risk_matched_equal_reference": equal * full.sum(),
    }


def _metric(arm, daily, cfg):
    values = np.r_[cfg.initial_capital, daily.NAV.to_numpy(dtype=float)]
    returns = values[1:] / values[:-1] - 1
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    return {
        "Arm": arm, "Description": ARM_LABELS.get(arm, arm),
        "ROI_Pct": float((values[-1] / cfg.initial_capital - 1) * 100),
        "Sharpe": float(returns.mean() / std * np.sqrt(cfg.annual_trading_days)) if std > 1e-12 else 0.0,
        "MDD_Pct": float((values / np.maximum.accumulate(values) - 1).min() * 100),
        "Total_Transaction_Cost": float(daily.Transaction_Cost.sum()),
        "Total_Turnover": float(daily.Turnover.sum()),
        "Average_Equity_Weight_Pct": float(daily.Equity_Weight.mean() * 100),
        "First_Execution_Date": str(daily.Date.min().date()),
        "Last_Valuation_Date": str(daily.Date.max().date()),
        "Observations": len(daily), "Result_Status": "exploratory",
        "Return_Basis": "net_research_portfolio_fractional_shares",
    }


def run_paired_research(predictions, prices, sentiment, baseline_protocol=None, config=None, tickers=TICKERS, decision_at=None):
    """Return protocol, metrics, NAV, weights and fills for a paired saved-prediction replay.

    DataFrames or CSV paths are accepted. Every arm shares dates and model core.
    Warm-up history must be supplied for causal risk thresholds. End liquidation
    is common: no new entry is opened on the terminal valuation date.
    """
    baseline = dict(baseline_protocol or {})
    cfg = _resolved_config(config, baseline)
    tickers = tuple(tickers)
    if len(set(tickers)) != len(tickers) or not tickers:
        raise ValueError("Tickers must be unique")
    panel = price_panel(prices, tickers)
    pred, dates = _prediction_table(predictions, panel, tickers)
    if len(dates) < 2:
        raise ValueError("At least two target dates are required to measure post-fill returns")
    used_dates = dates.union(pd.DatetimeIndex(pred.Signal_Date))
    if not np.isfinite(panel.loc[used_dates].to_numpy()).all():
        raise ValueError("Missing signal/execution price; evaluation rows are never dropped or filled")
    if "test_target_dates" in baseline and list(baseline["test_target_dates"]) != [str(x.date()) for x in dates]:
        raise ValueError("Saved 444 protocol target dates differ from the replay predictions")
    signals = pd.DatetimeIndex(sorted(pred.Signal_Date.unique()))
    news, news_evidence, news_raw = _sentiment_table(sentiment, signals, tickers, cfg, decision_at)
    risk = market_risk(panel, cfg)
    if not np.isfinite(risk.loc[signals, ["Market_Vol_20d", "Risk_Threshold_Pct"]].to_numpy()).all():
        raise ValueError("Risk warm-up incomplete on one or more signal dates")
    states = {arm: {"shares": np.zeros(len(tickers)), "cash": cfg.initial_capital} for arm in ARMS}
    nav_rows, weight_rows, fill_rows = [], [], []
    for number, date in enumerate(dates):
        part = pred[pred.Target_Date == date].set_index("Ticker").loc[list(tickers)]
        signal = pd.Timestamp(part.Signal_Date.iloc[0])
        core = part.Core_Predicted_Return_Pct.to_numpy(dtype=float)
        scores = news.loc[signal, list(tickers)].to_numpy(dtype=float)
        flag = bool(risk.loc[signal, "Risk_Off"])
        targets = _arm_weights(core, scores, flag, cfg)
        prices_today = panel.loc[date, list(tickers)].to_numpy(dtype=float)
        for arm in ARMS:
            state = states[arm]
            values = state["shares"] * prices_today
            final_day = number == len(dates) - 1
            terminal = final_day and cfg.liquidate_at_end
            should_trade = terminal or (not final_day and (arm != "static_equal444" or number == 0 or cfg.baseline_mode == "daily_equal_weight"))
            requested = np.zeros(len(tickers)) if terminal else targets[arm]
            ledger = {"Buy_Notional": 0.0, "Sell_Notional": 0.0, "Total_Cost": 0.0, "Turnover": 0.0}
            if should_trade:
                after, cash, ledger = rebalance(values, state["cash"], requested, cfg)
                delta = after - values
                for j, ticker in enumerate(tickers):
                    if abs(delta[j]) > 1e-8:
                        fill_rows.append({"Arm": arm, "Signal_Date": signal, "Execution_Date": date,
                                          "Ticker": ticker, "Execution_Price": prices_today[j],
                                          "Share_Change": delta[j] / prices_today[j],
                                          "Notional_Change": delta[j],
                                          "Reason": "terminal_liquidation" if terminal else "next_session_close_allocation"})
                state["shares"], state["cash"] = after / prices_today, cash
                values = after
            nav = float(values.sum() + state["cash"])
            equity = float(values.sum() / nav)
            nav_rows.append({"Arm": arm, "Date": date, "Signal_Date": signal, "NAV": nav,
                             "Cash": state["cash"], "Equity_Weight": equity,
                             "Risk_Off": flag, "Transaction_Cost": ledger["Total_Cost"],
                             "Turnover": ledger["Turnover"]})
            for j, ticker in enumerate(tickers):
                weight_rows.append({"Arm": arm, "Signal_Date": signal, "Target_Date": date,
                                    "Ticker": ticker, "Core_Predicted_Return_Pct": core[j],
                                    "Daily_Sentiment": scores[j], "Risk_Off": flag,
                                    "Market_Vol_20d": float(risk.loc[signal, "Market_Vol_20d"]),
                                    "Risk_Threshold_Pct": float(risk.loc[signal, "Risk_Threshold_Pct"]),
                                    "Policy_Target_Weight": float(targets[arm][j]),
                                    "Requested_Weight": float(requested[j]) if should_trade else np.nan,
                                    "Actual_Weight": float(values[j] / nav),
                                    "Terminal_Liquidation": terminal})
    nav = pd.DataFrame(nav_rows)
    metrics = pd.DataFrame([_metric(arm, nav[nav.Arm == arm], cfg) for arm in ARMS])
    index_values = panel.loc[dates, MARKET] / float(panel.loc[dates[0], MARKET]) * cfg.initial_capital
    reference = pd.DataFrame({"Arm": "market_price_reference", "Date": dates,
                              "Signal_Date": pd.NaT, "NAV": index_values.to_numpy(), "Cash": np.nan,
                              "Equity_Weight": np.nan, "Risk_Off": False, "Transaction_Cost": 0.0, "Turnover": 0.0})
    index_metric = _metric("market_price_reference", reference, cfg)
    index_metric.update({"Description": "^TWII theoretical price-index reference; not an investable net benchmark",
                         "Return_Basis": "theoretical_price_index_no_costs_no_dividends",
                         "Average_Equity_Weight_Pct": np.nan})
    metrics = pd.concat([metrics, pd.DataFrame([index_metric])], ignore_index=True)
    if cfg.baseline_mode == "daily_equal_weight":
        metrics.loc[metrics.Arm == "static_equal444", "Description"] = "444 daily equal allocation"
    nav = pd.concat([nav, reference], ignore_index=True)
    protocol = {
        "protocol_version": "paired-444-core-allocation-v1", "result_status": "exploratory",
        "causal_validated": False, "official_555_replacement": False,
        "common_core": "Identical saved 444 price predictions converted to next-day percent returns",
        "model_training_executed": False,
        "model_difference_isolated": "Allocation/news/risk overlays only; not a multivariate LSTM feature ablation",
        "tickers": list(tickers), "rules": asdict(cfg), "arms": {arm: ARM_LABELS[arm] for arm in ARMS},
        "execution": "News publication mapped by signal-day 13:30 cutoff; acquisition available by recorded decision, or unverified signal-day end upper bound; execute at next-session close",
        "entry_return": "No position earns signal-close to first execution-close return",
        "terminal_policy": "Liquidate all arms at final close without opening final-day new entries" if cfg.liquidate_at_end else "Mark existing holdings at final close without liquidation",
        "rebalance_frequency": "Every next-session close; static_equal444 initial only unless baseline_mode=daily_equal_weight",
        "risk_threshold": "Signal-day annualized rolling sample volatility > earlier volatility expanding mean + sample std",
        "risk_matched_reference": "Equal per-stock weights at exactly the news+risk policy target gross exposure, same execution/costs",
        "news_evidence": news_evidence,
        "selection": "Rules are fixed in configuration; no test-period optimization. Existing observed history is exploratory, not a fresh preregistered test",
        "shared_target_dates": [str(x.date()) for x in dates],
        "shared_signal_dates": [str(x.date()) for x in signals],
        "source_sha256": {"predictions_canonical_csv": _digest(pred), "prices_canonical_csv": _digest(panel.reset_index()),
                          "sentiment_canonical_csv": _digest(news_raw)},
        "source_444_evaluation_protocol": baseline,
        "paired_contrasts": {
            "prediction_dynamic_rule": ["dynamic_no_news", "equal_rebalanced_reference"],
            "news_without_risk": ["dynamic_news_no_risk", "dynamic_no_news"],
            "news_with_risk": ["dynamic_news_risk555", "dynamic_no_news_risk"],
            "risk_without_news": ["dynamic_no_news_risk", "dynamic_no_news"],
            "risk_with_news": ["dynamic_news_risk555", "dynamic_news_no_risk"],
            "stock_selection_at_same_gross": ["dynamic_news_risk555", "risk_matched_equal_reference"],
        },
        "original_six_feature_555_notes": [
            "Original six-feature 555 is a separate secondary model extension, not the shared 444-core overlay",
            "Its raw metrics versus this study mix model features, trained weights, targets and allocation effects; they cannot identify the incremental effect of news",
            "For model-feature attribution, refit all models on identical target dates, training and validation splits, scaler fit rows, seeds and selection rules, then replay their predictions through the same policy and next-close ledger",
            "Compare news inclusion within the same risk setting and risk inclusion within the same news setting; report both 2x2 contrasts and their interaction",
            "Shared dates and costs alone do not validate historical point-in-time data or create an independent test",
        ],
        "price_history_missing_rows": [str(day.date()) for day in panel.index[panel.isna().any(axis=1)]],
        "price_history_missing_policy": "Keep missing warm-up bars as missing; do not fill or drop. Require every signal/execution price and derived risk threshold to be finite",
        "limitations": ["News publication, acquisition, membership and scoring history require a separate point-in-time audit",
                        "Saved predictions are reused; independent forward evidence is required for superiority claims",
                        "Fractional shares, close fills, no slippage/market impact/dividend reinvestment; ^TWII is a price reference only"],
    }
    return {"protocol": protocol, "metrics": metrics, "nav": nav, "weights": pd.DataFrame(weight_rows),
            "fills": pd.DataFrame(fill_rows)}


def paired_daily_recommendation(recommendation, prices, sentiment, baseline_protocol=None, config=None, tickers=TICKERS, decision_at=None):
    """Use today's 444 forecast as the shared core; return all arms, without orders.

    Target_Date may lie beyond the last observed close. Prices must include the
    signal close and warm-up history. No old 555 model output is overwritten.
    """
    cfg = _resolved_config(config, baseline_protocol)
    tickers = tuple(tickers)
    raw = _frame(recommendation)
    raw = raw[raw.Ticker.isin(tickers)].copy()
    if raw.empty or raw.Ticker.duplicated().any() or set(raw.Ticker) != set(tickers):
        raise ValueError("Daily recommendation must have exactly one row per selected stock")
    raw["Signal_Date"], raw["Target_Date"] = _dates(raw.Signal_Date), _dates(raw.Target_Date)
    if raw.Signal_Date.nunique() != 1 or raw.Target_Date.nunique() != 1:
        raise ValueError("Daily recommendation dates must agree")
    signal, target = pd.Timestamp(raw.Signal_Date.iloc[0]), pd.Timestamp(raw.Target_Date.iloc[0])
    panel = price_panel(prices, tickers)
    if not np.isfinite(panel.loc[signal].to_numpy()).all():
        raise ValueError("Missing latest signal close")
    if signal != panel.index[-1] or target <= signal:
        raise ValueError("Daily core must use the latest available price close and a later target")
    from scripts.common import sessions
    future = sessions(signal + pd.Timedelta(days=1), signal + pd.Timedelta(days=30))
    if not len(future) or target != future[0]:
        raise ValueError("Daily target is not the next XTAI trading session")
    raw = raw.set_index("Ticker").loc[list(tickers)]
    previous = panel.loc[signal, list(tickers)].to_numpy(dtype=float)
    if "Previous_Close" in raw and not np.allclose(raw.Previous_Close, previous, rtol=1e-6, atol=1e-4):
        raise ValueError("Daily core and price snapshot differ")
    core = ((raw.Predicted_Close.to_numpy(dtype=float) / previous - 1) * 100
            if "Predicted_Close" in raw else raw.Predicted_Return_Pct.to_numpy(dtype=float))
    if not np.isfinite(core).all() or (core <= -100).any():
        raise ValueError("Invalid daily core prediction")
    if decision_at is None and "Created_At" in raw:
        created = pd.to_datetime(raw.Created_At, utc=True, errors="raise")
        if created.nunique() != 1:
            raise ValueError("Daily core needs one shared Created_At decision timestamp")
        decision_at = created.iloc[0]
    news, evidence, news_raw = _sentiment_table(sentiment, pd.DatetimeIndex([signal]), tickers, cfg, decision_at)
    risk = market_risk(panel, cfg).loc[signal]
    if not np.isfinite([risk.Market_Vol_20d, risk.Risk_Threshold_Pct]).all():
        raise ValueError("Daily risk warm-up incomplete")
    scores = news.loc[signal, list(tickers)].to_numpy(dtype=float)
    targets = _arm_weights(core, scores, bool(risk.Risk_Off), cfg)
    digest = _digest(raw.reset_index())
    rows = []
    for arm, weights in targets.items():
        for j, ticker in enumerate(tickers):
            rows.append({"Arm": arm, "Ticker": ticker, "Signal_Date": signal, "Target_Date": target,
                         "Predicted_Return_Pct": float(core[j]), "Daily_Sentiment": float(scores[j]),
                         "Target_Weight": float(weights[j]), "Cash_Weight": float(1 - weights.sum()),
                         "Risk_Off": bool(risk.Risk_Off), "Market_Vol_20d": float(risk.Market_Vol_20d),
                         "Risk_Threshold_Pct": float(risk.Risk_Threshold_Pct),
                         "Core_Model": "444", "Target_Date_Source": "444 recommendation; verified next XTAI session", "Core_Recommendation_SHA256": digest,
                         "Protocol": "paired-444-core-allocation-v1", "Status": "research_target_only_no_order",
                         "News_Point_In_Time_Verified": evidence["news_point_in_time_verified"],
                         "Note": "Static baseline is a reference target; fixed-share book requires existing holdings. Target is a close-fill approximation, not a forecast of return after the fill."})
    result = pd.DataFrame(rows)
    result.attrs["provenance"] = {"source_444_protocol": baseline_protocol or {}, "news_evidence": evidence,
                                  "rules": asdict(cfg), "causal_validated": False,
                                  "source_sha256": {"core_recommendation_canonical_csv": digest,
                                                    "prices_canonical_csv": _digest(panel.reset_index()),
                                                    "news_canonical_csv": _digest(news_raw)}}
    return result


def write_research_outputs(result, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(json.dumps(result["protocol"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for key in ("metrics", "nav", "weights", "fills"):
        result[key].to_csv(output / f"{key}.csv", index=False, encoding="utf-8-sig")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--sentiment", required=True)
    parser.add_argument("--baseline-protocol")
    parser.add_argument("--output", required=True)
    parser.add_argument("--recommendation")
    parser.add_argument("--decision-times", help="JSON mapping signal dates to timezone-aware actual decision timestamps")
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline_protocol).read_text(encoding="utf-8")) if args.baseline_protocol else {}
    decisions = json.loads(Path(args.decision_times).read_text(encoding="utf-8")) if args.decision_times else None
    result = run_paired_research(args.predictions, args.prices, args.sentiment, baseline, decision_at=decisions)
    output = write_research_outputs(result, args.output)
    if args.recommendation:
        daily = paired_daily_recommendation(args.recommendation, args.prices, args.sentiment, baseline)
        daily.to_csv(output / "daily_recommendation.csv", index=False, encoding="utf-8-sig")
        (output / "daily_recommendation.provenance.json").write_text(json.dumps(daily.attrs["provenance"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(result["metrics"].to_string(index=False))
    print(f"Research outputs: {output}")


if __name__ == "__main__":
    main()
