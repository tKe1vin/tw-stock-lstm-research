"""Walk-forward comparison of 555 risk-control variants.

The official notebook and its saved backtests are read-only inputs.
Parameter choice uses only out-of-sample target dates through 2026-03-25.
The already observed 2026-03-26 to 2026-11-04 window is reported separately
and is not an input to selection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "555" / "runs" / "20261104T185517400806"
RESULT_DIR = Path(__file__).resolve().parent / "results"

TICKERS = ["2330.TW", "2317.TW", "2603.TW", "2002.TW", "2891.TW"]
MARKET = "^TWII"
FEATURES = ["Close", "MA20", "RSI_14", "MACD_Hist", "Market_Vol_20d", "Daily_Sentiment"]
TIME_STEP = 10
VALIDATION_DAYS = 60
MIN_TRAIN_SAMPLES = 252
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 32
SEED = 42
INITIAL_CAPITAL = 100_000.0
BROKER_FEE_RATE = 0.001425
SELL_TAX_RATE = 0.003
SENTIMENT_BULL = 0.3
SENTIMENT_BEAR = -0.3
MAX_WEIGHT_LIMIT = 0.30
ANNUAL_TRADING_DAYS = 252
# Last target date allowed into parameter selection. The next session,
# 2026-03-26, is the first day of the already observed test window.
SELECTION_END = pd.Timestamp("2026-03-25")
OBSERVED_TEST_START = pd.Timestamp("2026-03-26")
MDD_BUDGET_PP = 5.0
BUFFER = 0.05
WEEKLY_GAP = 5

OFFICIAL = {
    "ROI_Pct": 11.953686872710323,
    "Sharpe": 1.6955502211150597,
    "MDD_Pct": -4.607245667559345,
    "Average_Cash_Weight_Pct": 37.0,
    "Risk_Off_Days": 107,
    "Total_Transaction_Cost": 5735.507730504729,
    "Total_Turnover": 18.10801391071648,
}


def technical_features(close):
    result = pd.DataFrame({"Close": close.astype(float)})
    result["MA20"] = close.rolling(20).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    result["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    result.loc[(loss == 0) & (gain > 0), "RSI_14"] = 100.0
    result.loc[(loss == 0) & (gain == 0), "RSI_14"] = 50.0
    macd = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(
        span=26, adjust=False, min_periods=26
    ).mean()
    result["MACD_Hist"] = macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()
    return result


def build_market_risk(market):
    risk = pd.DataFrame({"Market_Close": market})
    risk["Market_Return"] = market.pct_change(fill_method=None)
    risk["Market_Vol_20d"] = risk.Market_Return.rolling(20).std(ddof=1) * np.sqrt(252) * 100
    previous = risk.Market_Vol_20d.shift(1)
    risk["Risk_Threshold_Pct"] = previous.expanding(min_periods=60).mean() + previous.expanding(
        min_periods=60
    ).std(ddof=1)
    risk["Risk_Off"] = risk.Market_Vol_20d > risk.Risk_Threshold_Pct
    return risk


def load_panel(path):
    raw = pd.read_csv(path)
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize()
    panel = raw.pivot(index="Date", columns="ticker", values="Close").sort_index()
    return panel[TICKERS + [MARKET]].dropna().astype(float)


def load_sentiment(path, index):
    raw = pd.read_csv(path)
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize()
    table = raw.pivot(index="Date", columns="ticker", values="Daily_Sentiment").sort_index()
    return table.reindex(index)


def build_frames(panel, sentiment):
    risk = build_market_risk(panel[MARKET])
    risk.index.name = "Date"
    next_dates = pd.Series(panel.index, index=panel.index).shift(-1)
    frames = {}
    for ticker in TICKERS:
        frame = technical_features(panel[ticker])
        frame["Daily_Sentiment"] = sentiment[ticker].reindex(panel.index)
        frame["Market_Vol_20d"] = risk.Market_Vol_20d
        frame["Risk_Threshold_Pct"] = risk.Risk_Threshold_Pct
        frame["Risk_Off"] = risk.Risk_Off.fillna(False)
        frame["Target_Date"] = next_dates
        frame["Target_Next_Return"] = panel[ticker].pct_change(fill_method=None).shift(-1) * 100
        frames[ticker] = frame
    return frames, risk


def allocate(predictions, sentiments, risk_off, exposure):
    pred = np.asarray(predictions, dtype=float)
    sent = np.asarray(sentiments, dtype=float)
    if not np.isfinite(pred).all() or not np.isfinite(sent).all():
        raise ValueError("配置輸入含有非有限值")
    raw = np.full(len(pred), 0.2)
    raw[(pred > 0) & (sent > SENTIMENT_BULL)] = MAX_WEIGHT_LIMIT
    raw[(pred < 0) | (sent < SENTIMENT_BEAR)] = 0.0
    cap = exposure if risk_off else 1.0
    if raw.sum() > cap:
        raw *= cap / raw.sum()
    if (raw < 0).any() or raw.sum() > cap + 1e-10 or raw.max() > MAX_WEIGHT_LIMIT + 1e-10:
        raise ValueError("配置超出風控上限")
    return raw


def rebalance(current_values, cash, target_weights, fee=BROKER_FEE_RATE, tax=SELL_TAX_RATE):
    current = np.asarray(current_values, dtype=float)
    weights = np.asarray(target_weights, dtype=float)
    total = float(current.sum() + cash)
    if total <= 0 or cash < -1e-7 or (current < 0).any() or (weights < 0).any() or weights.sum() > 1 + 1e-10:
        raise ValueError("調倉資金或權重無效")
    low, high = 0.0, total
    for _ in range(70):
        mid = (low + high) / 2
        delta = weights * mid - current
        costs = delta.clip(min=0).sum() * fee + (-delta.clip(max=0)).sum() * (fee + tax)
        if mid + costs > total:
            high = mid
        else:
            low = mid
    after = (low + high) / 2
    target = weights * after
    delta = target - current
    buys = float(delta.clip(min=0).sum())
    sells = float((-delta.clip(max=0)).sum())
    ledger = {
        "Buy_Notional": buys,
        "Sell_Notional": sells,
        "Buy_Fee": buys * fee,
        "Sell_Fee": sells * fee,
        "Sell_Tax": sells * tax,
        "Total_Cost": buys * fee + sells * (fee + tax),
        "Turnover": (buys + sells) / total,
    }
    remaining = float(after - target.sum())
    if not np.isclose(target.sum() + remaining + ledger["Total_Cost"], total, atol=1e-6):
        raise AssertionError("調倉資金未守恆")
    return target, remaining, ledger


def sharpe_ratio(values):
    series = np.asarray(values, dtype=float)
    returns = series[1:] / series[:-1] - 1
    std = returns.std(ddof=1) if len(returns) > 1 else 0.0
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(ANNUAL_TRADING_DAYS))


def max_drawdown(values):
    series = np.asarray(values, dtype=float)
    return float((series / np.maximum.accumulate(series) - 1).min() * 100)


def strategies():
    return [
        {"id": "A", "name": "A 原始曝險 50%", "family": "exposure", "exposure": 0.50, "trigger": "same_day", "rebalance": "daily"},
        {"id": "B", "name": "B 曝險 65%", "family": "exposure", "exposure": 0.65, "trigger": "same_day", "rebalance": "daily"},
        {"id": "C", "name": "C 曝險 75%", "family": "exposure", "exposure": 0.75, "trigger": "same_day", "rebalance": "daily"},
        {"id": "D", "name": "D 曝險 85%", "family": "exposure", "exposure": 0.85, "trigger": "same_day", "rebalance": "daily"},
        {"id": "E", "name": "E 連續兩日才 Risk-Off", "family": "trigger", "exposure": 0.50, "trigger": "two_day", "rebalance": "daily"},
        {"id": "F", "name": "F 權重緩衝 5 個百分點", "family": "rebalance", "exposure": 0.50, "trigger": "same_day", "rebalance": "buffer"},
        {"id": "G", "name": "G 每 5 個交易日再平衡", "family": "rebalance", "exposure": 0.50, "trigger": "same_day", "rebalance": "weekly"},
    ]


def _flags_for(spec, day, risk):
    signal = pd.Timestamp(day.Signal_Date.iloc[0])
    location = risk.index.get_loc(signal)
    if isinstance(location, slice) or not isinstance(location, (int, np.integer)):
        raise ValueError(f"找不到唯一的訊號日 {signal.date()}")
    same = bool(risk.Risk_Off.iloc[location])
    if spec["trigger"] == "same_day":
        return same
    if location == 0:
        return False
    return bool(same and risk.Risk_Off.iloc[location - 1])


def run_portfolio(predictions, panel, risk, spec, liquidate=True):
    ordered = []
    for target_date, day in predictions.groupby("Date", sort=True):
        day = day.set_index("Ticker").loc[TICKERS].reset_index()
        if len(day) != len(TICKERS):
            raise ValueError(f"{pd.Timestamp(target_date).date()} 預測股票不完整")
        ordered.append(day)
    shares = np.zeros(len(TICKERS))
    cash = INITIAL_CAPITAL
    first_signal = pd.Timestamp(ordered[0].Signal_Date.iloc[0])
    nav_rows = [{"Date": first_signal, "NAV": INITIAL_CAPITAL, "Cash": cash}]
    cash_weights = []
    costs = []
    risk_flags = []
    rebalanced = []
    days_since_rebalance = WEEKLY_GAP
    for index, day in enumerate(ordered):
        signal = pd.Timestamp(day.Signal_Date.iloc[0])
        target = pd.Timestamp(day.Date.iloc[0])
        price_before = panel.loc[signal, TICKERS].to_numpy(dtype=float)
        price_after = panel.loc[target, TICKERS].to_numpy(dtype=float)
        old_values = shares * price_before
        pre_nav = float(old_values.sum() + cash)
        flag = _flags_for(spec, day, risk)
        target_weights = allocate(
            day.Predicted_Return_Pct.to_numpy(),
            day.Daily_Sentiment.to_numpy(),
            flag,
            spec["exposure"],
        )
        current_weights = old_values / pre_nav
        if spec["rebalance"] == "daily":
            trade = True
        elif spec["rebalance"] == "buffer":
            trade = index == 0 or float(np.max(np.abs(target_weights - current_weights))) >= BUFFER
        elif spec["rebalance"] == "weekly":
            trade = days_since_rebalance >= WEEKLY_GAP
        else:
            raise ValueError(spec["rebalance"])
        if trade:
            target_values, cash, ledger = rebalance(old_values, cash, target_weights)
            shares = target_values / price_before
            held_cash_weight = float(1 - target_weights.sum())
            days_since_rebalance = 1
        else:
            ledger = {
                "Buy_Notional": 0.0,
                "Sell_Notional": 0.0,
                "Buy_Fee": 0.0,
                "Sell_Fee": 0.0,
                "Sell_Tax": 0.0,
                "Total_Cost": 0.0,
                "Turnover": 0.0,
            }
            held_cash_weight = float(cash / pre_nav)
            days_since_rebalance += 1
        ending = float((shares * price_after).sum() + cash)
        if liquidate and index == len(ordered) - 1:
            liquidation = float((shares * price_after).sum())
            last_cost = liquidation * (BROKER_FEE_RATE + SELL_TAX_RATE)
            ending -= last_cost
            cash = ending
            shares[:] = 0
            costs.append({
                "Total_Cost": last_cost,
                "Turnover": liquidation / (ending + last_cost),
                "Stage": "Liquidation",
            })
        costs.append({**ledger, "Stage": "Rebalance"})
        nav_rows.append({"Date": target, "NAV": ending, "Cash": cash})
        cash_weights.append(held_cash_weight)
        risk_flags.append(flag)
        rebalanced.append(trade)
    nav = pd.DataFrame(nav_rows)
    cost_frame = pd.DataFrame(costs)
    return {
        "nav": nav,
        "ROI_Pct": float((nav.NAV.iloc[-1] / INITIAL_CAPITAL - 1) * 100),
        "Sharpe": sharpe_ratio(nav.NAV),
        "MDD_Pct": max_drawdown(nav.NAV),
        "Average_Cash_Weight_Pct": float(np.mean(cash_weights) * 100),
        "Average_Invested_Weight_Pct": float((1 - np.mean(cash_weights)) * 100),
        "Risk_Off_Days": int(np.sum(risk_flags)),
        "Rebalance_Days": int(np.sum(rebalanced)),
        "Total_Transaction_Cost": float(cost_frame.Total_Cost.sum()),
        "Total_Turnover": float(cost_frame.Turnover.sum()),
        "Test_Days": len(ordered),
    }


def reproduce_official():
    predictions = pd.read_csv(RUN / "test_predictions.csv", parse_dates=["Date", "Signal_Date"])
    predictions["Date"] = predictions.Date.dt.normalize()
    predictions["Signal_Date"] = predictions.Signal_Date.dt.normalize()
    panel = load_panel(RUN / "historical_prices.csv")
    saved_risk = pd.read_csv(RUN / "market_risk_features.csv", parse_dates=["Date"]).set_index("Date")
    saved_risk.index = saved_risk.index.tz_localize(None).normalize()
    rebuilt, risk = build_frames(panel, load_sentiment(RUN / "sentiment_snapshot.csv", panel.index))
    common = saved_risk.index.intersection(risk.index)
    mismatch = (saved_risk.loc[common, "Risk_Off"].astype(bool) != risk.loc[common, "Risk_Off"].astype(bool)).sum()
    if mismatch:
        raise AssertionError(f"重建的 Risk-Off 與正式輸出有 {mismatch} 日不同")
    vol_gap = (saved_risk.loc[common, "Market_Vol_20d"] - risk.loc[common, "Market_Vol_20d"]).abs()
    if vol_gap.dropna().max() > 1e-8:
        raise AssertionError("重建的波動與正式輸出不一致")
    # A same-day signal must be computable from that day's close and earlier closes only.
    shifted = risk.Market_Vol_20d.shift(1)
    threshold = shifted.expanding(min_periods=60).mean() + shifted.expanding(min_periods=60).std(ddof=1)
    if not np.allclose(
        threshold.dropna().to_numpy(),
        risk.loc[threshold.dropna().index, "Risk_Threshold_Pct"].to_numpy(),
        equal_nan=True,
    ):
        raise AssertionError("風險門檻使用了當日波動")
    result = run_portfolio(predictions, panel, risk, strategies()[0])
    checks = {
        "ROI_Pct": result["ROI_Pct"],
        "Sharpe": result["Sharpe"],
        "MDD_Pct": result["MDD_Pct"],
        "Average_Cash_Weight_Pct": result["Average_Cash_Weight_Pct"],
        "Risk_Off_Days": result["Risk_Off_Days"],
        "Total_Transaction_Cost": result["Total_Transaction_Cost"],
        "Total_Turnover": result["Total_Turnover"],
    }
    for key, expected in OFFICIAL.items():
        got = checks[key]
        tolerance = 1e-6 if key == "Risk_Off_Days" else 1e-4
        if abs(got - expected) > tolerance:
            raise AssertionError(f"正式 555 重現失敗 {key}: 計算 {got}，輸出 {expected}")
    signal_close_used = predictions.groupby("Signal_Date").Previous_Close.mean()
    return {"reproduction": checks, "frames_ready": True, "risk_days_checked": int(len(common)), "signal_rows": int(len(signal_close_used))}


def _valid_ends(frame):
    values = frame[FEATURES].to_numpy(dtype=float)
    ends = []
    returns = frame.Target_Next_Return.to_numpy(dtype=float)
    for end in range(TIME_STEP - 1, len(frame) - 1):
        start = end - TIME_STEP + 1
        if np.isfinite(values[start : end + 1]).all() and np.isfinite(returns[end]):
            ends.append(end)
    return np.asarray(ends, dtype=int)


def _prepare_windows(frame):
    ends = _valid_ends(frame)
    label_dates = pd.DatetimeIndex(pd.to_datetime(frame.Target_Date.iloc[ends])).tz_localize(None).normalize()
    values = frame[FEATURES].to_numpy(dtype=float)
    return ends, label_dates, values


def walk_forward_predictions(frames, selection_end=SELECTION_END):
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    prepared = {ticker: _prepare_windows(frames[ticker]) for ticker in TICKERS}
    common = prepared[TICKERS[0]][1]
    for ticker in TICKERS[1:]:
        if not prepared[ticker][1].equals(common):
            raise ValueError(f"{ticker} 的有效標籤日與其他股票不同")
    eligible = common[common <= selection_end]
    if len(eligible) <= MIN_TRAIN_SAMPLES + VALIDATION_DAYS:
        raise ValueError("選參期間的有效樣本不足")
    cursor = MIN_TRAIN_SAMPLES + VALIDATION_DAYS
    folds = []
    rows = []
    while cursor < len(eligible):
        block = eligible[cursor : cursor + VALIDATION_DAYS]
        val_dates = eligible[cursor - VALIDATION_DAYS : cursor]
        train_dates = eligible[: cursor - VALIDATION_DAYS]
        if len(train_dates) < MIN_TRAIN_SAMPLES or len(val_dates) != VALIDATION_DAYS or len(block) == 0:
            raise ValueError("walk-forward 切分無效")
        if set(block).intersection(train_dates.union(val_dates)):
            raise ValueError("預測區塊與訓練或驗證重疊")
        print(
            f"[WF] 預測 {block[0].date()} 至 {block[-1].date()}，"
            f"訓練 {train_dates[0].date()} 至 {train_dates[-1].date()}",
            flush=True,
        )
        for ticker in TICKERS:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(SEED)
            ends, label_dates, values = prepared[ticker]
            train_index = np.where(label_dates.isin(train_dates))[0]
            val_index = np.where(label_dates.isin(val_dates))[0]
            test_index = np.where(label_dates.isin(block))[0]
            if len(train_index) != len(train_dates) or len(val_index) != len(val_dates) or len(test_index) != len(block):
                raise ValueError(f"{ticker} 視窗日期無法對齊")
            train_rows = np.unique(np.concatenate([
                np.arange(ends[item] - TIME_STEP + 1, ends[item] + 1) for item in train_index
            ]))
            scaler = MinMaxScaler().fit(values[train_rows])
            valid_rows = np.isfinite(values).all(axis=1)
            scaled = np.full_like(values, np.nan)
            scaled[valid_rows] = scaler.transform(values[valid_rows])
            windows = np.stack([scaled[end - TIME_STEP + 1 : end + 1] for end in ends])
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(TIME_STEP, len(FEATURES))),
                tf.keras.layers.LSTM(50, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(50),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            history = model.fit(
                windows[train_index],
                frames[ticker].Target_Next_Return.iloc[ends[train_index]].to_numpy(dtype=float),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=False,
                validation_data=(
                    windows[val_index],
                    frames[ticker].Target_Next_Return.iloc[ends[val_index]].to_numpy(dtype=float),
                ),
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
                )],
                verbose=0,
            )
            predicted = model.predict(windows[test_index], verbose=0).ravel()
            if not np.isfinite(predicted).all():
                raise ValueError(f"{ticker} 產生非有限預測")
            chosen = frames[ticker].iloc[ends[test_index]]
            part = pd.DataFrame({
                "Date": label_dates[test_index],
                "Signal_Date": chosen.index,
                "Ticker": ticker,
                "Predicted_Return_Pct": predicted,
                "Daily_Sentiment": chosen.Daily_Sentiment.to_numpy(dtype=float),
                "Actual_Return_Pct": frames[ticker].Target_Next_Return.iloc[ends[test_index]].to_numpy(dtype=float),
            })
            rows.append(part)
            folds.append({
                "ticker": ticker,
                "oos_start": str(block[0].date()),
                "oos_end": str(block[-1].date()),
                "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                "epochs_ran": len(history.history["val_loss"]),
            })
            del model
        cursor += VALIDATION_DAYS
    predictions = pd.concat(rows, ignore_index=True)
    predictions["Date"] = pd.to_datetime(predictions.Date).dt.normalize()
    predictions["Signal_Date"] = pd.to_datetime(predictions.Signal_Date).dt.normalize()
    if predictions.Date.max() > selection_end:
        raise AssertionError("選參預測進入了已觀察測試期")
    return predictions, pd.DataFrame(folds)


def evaluate(predictions, panel, risk, sample_name):
    baseline = None
    records = []
    navs = {}
    for spec in strategies():
        result = run_portfolio(predictions, panel, risk, spec)
        navs[spec["id"]] = result["nav"].assign(Strategy=spec["id"])
        if spec["id"] == "A":
            baseline = result
        records.append({
            "Sample": sample_name,
            "Strategy_ID": spec["id"],
            "Strategy": spec["name"],
            "Family": spec["family"],
            "Risk_Off_Exposure": spec["exposure"],
            "Trigger": spec["trigger"],
            "Rebalance": spec["rebalance"],
            "ROI_Pct": result["ROI_Pct"],
            "Sharpe": result["Sharpe"],
            "MDD_Pct": result["MDD_Pct"],
            "Average_Cash_Weight_Pct": result["Average_Cash_Weight_Pct"],
            "Average_Invested_Weight_Pct": result["Average_Invested_Weight_Pct"],
            "Risk_Off_Days": result["Risk_Off_Days"],
            "Rebalance_Days": result["Rebalance_Days"],
            "Test_Days": result["Test_Days"],
            "Total_Transaction_Cost": result["Total_Transaction_Cost"],
            "Total_Turnover": result["Total_Turnover"],
        })
    table = pd.DataFrame(records)
    table["ROI_Diff_vs_A_pp"] = table.ROI_Pct - baseline["ROI_Pct"]
    table["Sharpe_Diff_vs_A"] = table.Sharpe - baseline["Sharpe"]
    table["MDD_Diff_vs_A_pp"] = table.MDD_Pct - baseline["MDD_Pct"]
    table["Cost_Diff_vs_A"] = table.Total_Transaction_Cost - baseline["Total_Transaction_Cost"]
    table["Cash_Diff_vs_A_pp"] = table.Average_Cash_Weight_Pct - baseline["Average_Cash_Weight_Pct"]
    return table, pd.concat(navs.values(), ignore_index=True)


def select_exposure(table):
    """Pre-specified rule: exposure family only, selection sample only."""
    subset = table[(table.Sample.str.startswith("walk_forward")) & (table.Family == "exposure")].copy()
    base = subset.loc[subset.Strategy_ID == "A"].iloc[0]
    subset["Eligible"] = subset.MDD_Pct >= base.MDD_Pct - MDD_BUDGET_PP
    eligible = subset[subset.Eligible]
    best_sharpe = float(eligible.Sharpe.max())
    band = eligible[eligible.Sharpe >= best_sharpe - 0.02]
    winner = band.sort_values(
        ["Total_Transaction_Cost", "ROI_Pct"],
        ascending=[True, False],
    ).iloc[0]
    return {
        "rule": (
            f"只在 walk-forward、目標日不晚於 {SELECTION_END.date()} 的 A–D 中選擇。"
            f"MDD 比 A 惡化超過 {MDD_BUDGET_PP:.0f} 個百分點者剔除；其餘取 Sharpe 最高者。"
            "Sharpe 差距小於 0.02 時改取交易成本較低者。"
            "已觀察測試期不進入這個規則。"
        ),
        "selected_id": winner.Strategy_ID,
        "selected_name": winner.Strategy,
        "selected_sharpe": float(winner.Sharpe),
        "selected_roi": float(winner.ROI_Pct),
        "selected_mdd": float(winner.MDD_Pct),
        "candidates": subset[
            ["Strategy_ID", "Sharpe", "ROI_Pct", "MDD_Pct", "Eligible"]
        ].to_dict(orient="records"),
    }


def regime_summary(predictions, panel, risk):
    spec = strategies()[0]
    # Re-run is unnecessary for regime math: use same-day flags and official exposure path metrics
    # are already in the comparison table. This splits the selection predictions by stored risk flag.
    flags = []
    returns = []
    for _, day in predictions.groupby("Date", sort=True):
        signal = pd.Timestamp(day.Signal_Date.iloc[0])
        target = pd.Timestamp(day.Date.iloc[0])
        flags.append(bool(risk.loc[signal, "Risk_Off"]))
        market_return = panel.loc[target, MARKET] / panel.loc[signal, MARKET] - 1
        returns.append(float(market_return * 100))
    frame = pd.DataFrame({"Risk_Off": flags, "Market_Return_Pct": returns})
    summary = []
    for flag, name in [(True, "Risk-Off"), (False, "Risk-On")]:
        part = frame[frame.Risk_Off == flag]
        summary.append({
            "Regime": name,
            "Days": int(len(part)),
            "Sum_Market_Return_pp": float(part.Market_Return_Pct.sum()),
            "Mean_Market_Return_pp": float(part.Market_Return_Pct.mean()),
        })
    return summary


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("[1] 重現正式 555 測試期", flush=True)
    reproduction = reproduce_official()
    panel = load_panel(RUN / "historical_prices.csv")
    sentiment = load_sentiment(RUN / "sentiment_snapshot.csv", panel.index)
    frames, risk = build_frames(panel, sentiment)
    print("[2] walk-forward 訓練，目標日止於 2026-03-25", flush=True)
    predictions, folds = walk_forward_predictions(frames)
    predictions.to_csv(RESULT_DIR / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(RESULT_DIR / "walk_forward_folds.csv", index=False, encoding="utf-8-sig")
    selection, selection_nav = evaluate(predictions, panel, risk, "walk_forward_through_2026-03-25")
    print("[3] 已觀察測試期只做探索性重播，不參與選參", flush=True)
    observed = pd.read_csv(RUN / "test_predictions.csv", parse_dates=["Date", "Signal_Date"])
    observed["Date"] = observed.Date.dt.normalize()
    observed["Signal_Date"] = observed.Signal_Date.dt.normalize()
    if observed.Date.min() < OBSERVED_TEST_START:
        raise AssertionError("探索性樣本含有測試期之前的目標日")
    exploratory, exploratory_nav = evaluate(observed, panel, risk, "exploratory_observed_test_2026-03-26_2026-11-04")
    comparison = pd.concat([selection, exploratory], ignore_index=True)
    decision = select_exposure(selection)
    regimes = regime_summary(predictions, panel, risk)
    comparison.to_csv(RESULT_DIR / "comparison.csv", index=False, encoding="utf-8-sig")
    selection_nav.to_csv(RESULT_DIR / "walk_forward_nav.csv", index=False, encoding="utf-8-sig")
    exploratory_nav.to_csv(RESULT_DIR / "exploratory_nav.csv", index=False, encoding="utf-8-sig")
    payload = {
        "reproduction": reproduction["reproduction"],
        "selection_rule": decision,
        "walk_forward_regimes": regimes,
        "execution_assumption": "沿用正式 555：訊號日收盤價與當日波動可用，並假設能以該收盤價成交，持有至下一交易日收盤。",
        "close_to_close": True,
        "threshold_excludes_current_vol": True,
        "observed_test_used_for_selection": False,
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    print("SELECTED", decision["selected_id"], decision["selected_name"], flush=True)
    return comparison, decision


if __name__ == "__main__":
    main()
